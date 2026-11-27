"""Referência Python: VAD em PCM e MFCC por janela independente de 2 s.

WAVs devem conter PCM pós-DC-blocker do capturador. Não reaplicamos o
DC blocker: isso duplicaria o filtro nas gravações do INMP441.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from .voice_prior import VOICE_PRIOR

TARGET_SR = 16000
VOICE_LOW, VOICE_HIGH = 80, 8000
N_FFT, HOP_LENGTH, N_MFCC, N_MELS = 1024, 256, 13, 40
WIN_SAMPLES, VAD_CHUNK = 32000, 1600
NAME = re.compile(r"^locutor_(?P<speaker>[a-zA-Z0-9_]+)_(?P<session>s\d+)\.wav$")


def load_audio(path):
    y, sr = sf.read(path, dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f"Áudio vazio ou não finito: {path}")
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    # Mesmo domínio int16 da entrada do MFCC embarcado.
    return np.clip(np.rint(y * 32768), -32768, 32767).astype("<i2")


def bandpass(y):
    sos = signal.butter(5, [80, 7920], btype="bandpass", fs=TARGET_SR, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(y, dtype=np.float64))


def power(y):
    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                              window="hann", center=True, pad_mode="constant")) ** 2


def noise_profile(pcm):
    """Primeira calibração; perfil fixo para comparação offline controlada."""
    if len(pcm) != 8192:
        raise ValueError("O perfil exige exatamente 8192 amostras de silêncio")
    return power(bandpass(np.asarray(pcm, dtype=np.float64) / 32768)).mean(axis=1)


def auto_noise_profile(pcm, threshold=300):
    """Estima o perfil de ruído da sessão a partir dos blocos silenciosos (<= 2/3 do VAD)."""
    chunks = [pcm[i:i + VAD_CHUNK] for i in range(0, len(pcm) - VAD_CHUNK + 1, VAD_CHUNK)]
    if not chunks:
        return None
    scored = sorted(((int(np.sqrt(np.mean(c.astype(np.float64) ** 2))), idx, c)
                     for idx, c in enumerate(chunks)), key=lambda x: (x[0], x[1]))
    quiet = [c for r, _, c in scored[:6] if r <= threshold * 2 // 3]
    if len(quiet) * VAD_CHUNK < 8192:
        return None
    return noise_profile(np.concatenate(quiet)[:8192])


def pipeline(y, noise=None, alpha=0.8, floor=0.05):
    """Float mono normalizado -> (MFCC 39×T, passa-banda).

    Piso de 80 dB aplicado por frame, como frame_mfcc do firmware.
    """
    yf = bandpass(y)
    p = power(yf)
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float64)
        if noise.shape != (513,) or not np.isfinite(noise).all() or (noise < 0).any():
            raise ValueError("Perfil de ruído inválido")
        e_n = float(noise.sum())
        if e_n > 0.0:
            e_sig = np.maximum(p.sum(axis=0) - e_n, 0.05 * e_n)
            s_hat = VOICE_PRIOR[:, None] * e_sig[None, :]
            denom = s_hat + alpha * noise[:, None]
            g = np.divide(s_hat, denom, out=np.zeros_like(s_hat), where=denom > 1e-12)
            g2 = np.maximum(g * g, floor)
            p = g2 * p
    mel = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS) @ p
    db = 10 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max(axis=0, keepdims=True) - 80)
    m = librosa.feature.mfcc(S=db, sr=TARGET_SR, n_mfcc=N_MFCC)
    return np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]), yf


def voiced_windows(pcm, threshold=300):
    """Descarta blocos silenciosos e caudas; nunca cruza sessões.

    Reinicia após 100 blocos sem completar a janela, como capture_voiced.
    O gatilho inicial de 300 ms é de interface e não é simulado aqui.
    """
    if not 1 <= threshold <= 32767:
        raise ValueError("VAD fora de [1, 32767]")
    chunks, attempts = [], 0
    for i in range(0, len(pcm) - VAD_CHUNK + 1, VAD_CHUNK):
        block = pcm[i:i + VAD_CHUNK]
        attempts += 1
        rms = int(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        if rms > threshold:
            chunks.append(block)
        if len(chunks) == WIN_SAMPLES // VAD_CHUNK:
            yield np.concatenate(chunks)
            chunks, attempts = [], 0
        elif attempts == 100:
            chunks, attempts = [], 0


@dataclass
class Dataset:
    vectors: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray
    frames: list
    sources: list


def load_dataset(directory, threshold=300, noise=None):
    vectors, labels, sessions, frames, sources = [], [], [], [], []
    files = sorted(Path(directory).glob("locutor_*.wav"))
    if not files:
        raise ValueError(f"Nenhum locutor_*.wav em {directory}")
    for path in files:
        match = NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Nome inválido: {path.name}; use locutor_NOME_sN.wav")
        speaker = match['speaker'].upper()
        if len(speaker) >= 16:
            raise ValueError(f"Nome deve ter até 15 caracteres: {speaker}")
        pcm = load_audio(path)
        sess_noise = noise if noise is not None else auto_noise_profile(pcm, threshold)
        count = 0
        for window in voiced_windows(pcm, threshold):
            m, _ = pipeline(window.astype(np.float64) / 32768, noise=sess_noise)
            frames.append(m)
            vectors.append(m.mean(axis=1))
            labels.append(speaker)
            sessions.append(speaker + '/' + match['session'])
            sources.append(path.name)
            count += 1
        if not count:
            raise ValueError(f"Sessão sem janela completa de voz: {path.name}")
    return Dataset(np.asarray(vectors, dtype=np.float32), np.array(labels),
                   np.array(sessions), frames, sources)
