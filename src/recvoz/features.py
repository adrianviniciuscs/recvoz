"""Referência Python pura (NumPy/SciPy, sem librosa): VAD em PCM e MFCC 34-dim (17c + Δ).

WAVs devem conter PCM pós-DC-blocker do capturador. Não reaplicamos o
DC blocker: isso duplicaria o filtro nas gravações do INMP441.
"""
from dataclasses import dataclass
from math import gcd
from pathlib import Path
import re

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct
from .voice_prior import VOICE_PRIOR

TARGET_SR = 16000
VOICE_LOW, VOICE_HIGH = 80, 8000
N_FFT, HOP_LENGTH, N_MFCC, N_MELS = 1024, 256, 17, 40
MFCC_DIM = 2 * N_MFCC  # 34: c0..c16 + Δc0..Δc16 (sem Δ²)
WIN_SAMPLES, VAD_CHUNK = 32000, 1600
NAME = re.compile(r"^locutor_(?P<speaker>[a-zA-Z0-9_]+)_(?P<session>s\d+)\.wav$")

_HANN_WIN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT, dtype=np.float64) / N_FFT)
_SG1_W = np.arange(-4, 5, dtype=np.float64) / 60.0


def mel_filterbank(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    """Banco de filtros triangulares na escala Mel HTK com normalização de área."""
    if fmax is None:
        fmax = sr / 2.0
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    f_pts = mel2hz(m_pts)
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(n_mels):
        fl, fc, fr = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        up = (freqs - fl) / max(fc - fl, 1e-12)
        dn = (fr - freqs) / max(fr - fc, 1e-12)
        fb[m] = np.maximum(0.0, np.minimum(up, dn)) * (2.0 / max(fr - fl, 1e-12))
    return fb


_MEL_FB = mel_filterbank()


def load_audio(path):
    y, sr = sf.read(path, dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f"Áudio vazio ou não finito: {path}")
    if sr != TARGET_SR:
        g = gcd(int(TARGET_SR), int(sr))
        y = signal.resample_poly(y, int(TARGET_SR) // g, int(sr) // g)
    # Mesmo domínio int16 da entrada do MFCC embarcado.
    return np.clip(np.rint(y * 32768), -32768, 32767).astype("<i2")


def bandpass(y):
    sos = signal.butter(5, [80, 7920], btype="bandpass", fs=TARGET_SR, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(y, dtype=np.float64))


def power(y):
    """Densidade espectral de potência |STFT|^2 (Hann periódica, center=True, zero-pad)."""
    y = np.asarray(y, dtype=np.float64)
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="constant")
    n_frames = 1 + (len(yp) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + np.arange(n_frames)[:, None] * HOP_LENGTH
    spec = np.fft.rfft(yp[idx] * _HANN_WIN[None, :], n=N_FFT, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).T


def delta_sg(m):
    """Derivada 1a Savitzky-Golay (largura 9) com replicação das janelas de borda."""
    n_coef, n_fr = m.shape
    out = np.empty_like(m)
    for t in range(n_fr):
        base = 0 if t < 4 else (n_fr - 9 if t > n_fr - 5 else t - 4)
        out[:, t] = m[:, base : base + 9] @ _SG1_W
    return out


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
    """Float mono normalizado -> (MFCC 34×T [17c + Δ], passa-banda).

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
    mel = _MEL_FB @ p
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max(axis=0, keepdims=True) - 80.0)
    m = dct(db, type=2, axis=0, norm="ortho")[:N_MFCC]
    return np.vstack([m, delta_sg(m)]), yf


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
