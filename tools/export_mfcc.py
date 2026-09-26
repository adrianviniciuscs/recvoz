#!/usr/bin/env python3
"""Exporta tabelas e vetores de teste do MFCC p/ o firmware (F1).

Pipeline IDENTICA a eda_speaker_recognition.ipynb, rodada com o librosa
deste venv (1.x — delta via Savitzky-Golay, STFT center+zero-pad, rfft sem
escala, mel slaney, power_to_db ref=1.0/top_db=80, dct ortho):

    int16 -> /32768 -> butter 80-8000 (filtfilt) -> STFT 1024/256 Hann
    -> |.|^2 -> mel128 -> 10*log10 -> DCT13 -> deltas SG -> 39-dim

Gera:
  hardware/recvoz-esp/components/mfcc/mfcc_tables.h  (mel + SOS + zi + SG)
  tools/mfcc_test/test_pcm.bin      (32000 int16 LE, 2 s de adrian_s1)
  tools/mfcc_test/expected_bp.bin   (filtrado float32, 32000)
  tools/mfcc_test/expected_mfcc.bin (39x126 float32 C-order)
  tools/mfcc_test/expected_mean.bin (39 float32)
  tools/mfcc_test/enroll_vectors.npz (p/ F2: mean-vecs 125-frame chunks + labels)

Uso:  .venv/bin/python tools/export_mfcc.py
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MFCC_DIR = ROOT / "hardware/recvoz-esp/components/mfcc"
DB_DIR = ROOT / "hardware/recvoz-esp/components/speaker_db"
TEST_DIR = ROOT / "tools/mfcc_test"


def C(v):
    s = f"{v:.9g}"
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return s + "f"

# Parâmetros do notebook (não mexer sem re-treinar)
TARGET_SR = 16000
VOICE_LOW, VOICE_HIGH = 80, 8000
N_FFT, HOP_LENGTH = 1024, 256
N_MFCC = 13
WIN_SAMPLES = 32000  # 2 s

import librosa

assert librosa.__version__.startswith("1."), f"rode neste venv (librosa 1.x), achado {librosa.__version__}"
from scipy import signal


def pipeline(y):
    """y: float64 mono 16 kHz. Retorna (mfcc39xT float64, y_filtrado)."""
    nyq = TARGET_SR / 2.0
    b, a = signal.butter(5, [VOICE_LOW / nyq, min(VOICE_HIGH / nyq, 0.99)], btype="bandpass")
    yf = signal.filtfilt(b, a, y)
    mfcc = librosa.feature.mfcc(y=yf, sr=TARGET_SR, n_mfcc=N_MFCC,
                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, d1, d2]), yf


def main():
    import soundfile as sf

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    MFCC_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. tabelas fixas ---
    nyq = TARGET_SR / 2.0
    sos = signal.butter(5, [VOICE_LOW / nyq, min(VOICE_HIGH / nyq, 0.99)],
                        btype="bandpass", output="sos")
    zi = signal.sosfilt_zi(sos)
    assert sos.shape == (5, 6), sos.shape

    mel = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=128,
                              fmin=0.0, fmax=8000.0, htk=False, norm="slaney")
    assert mel.shape == (128, 513), mel.shape
    # Pesos exatos (CSR): triângulos amostrados não zeram nas bordas,
    # então copiamos os valores em vez de reconstruir rampas.
    rows = []
    for m in range(128):
        nz = np.nonzero(mel[m] > 0)[0]
        rows.append((int(nz[0]), mel[m][nz[0]:nz[-1] + 1].astype(np.float64)))
    stride = max(len(w) for _, w in rows)
    print("mel stride:", stride)

    w2 = signal.savgol_coeffs(9, 2, 2)  # device usa correlacao; janela simetrica
    assert np.allclose(w2, w2[::-1]), "SG order2 deveria ser simetrico"
    print("SG order2 interior:", np.round(w2, 6))

    with open(MFCC_DIR / "mfcc_tables.h", "w") as f:
        f.write("// GERADO por tools/export_mfcc.py — não editar à mão.\n")
        f.write(f"// librosa {librosa.__version__} | sr=16000 bandpass 80-7920 SOS(5) mel128 slaney\n#pragma once\n")
        f.write("#include <stdint.h>\n\n#define MFCC_N_SOS 5\n");
        f.write("static const float MFCC_SOS[5][6] = {\n")
        for row in sos:
            f.write("    {%s},\n" % ", ".join(C(v) for v in row))
        f.write("};\nstatic const float MFCC_SOS_ZI[5][2] = {\n")
        for row in zi:
            f.write("    {%s},\n" % ", ".join(C(v) for v in row))
        f.write("};\n\n#define MFCC_N_MELS 128\n")
        f.write(f"#define MFCC_MEL_STRIDE {stride}\n")
        f.write("static const uint16_t MFCC_MEL_START[128] = {\n    ")
        f.write(", ".join(str(s) for s, _ in rows))
        f.write("\n};\nstatic const uint8_t MFCC_MEL_LEN[128] = {\n    ")
        f.write(", ".join(str(len(w)) for _, w in rows))
        f.write("\n};\nstatic const float MFCC_MEL_W[128][MFCC_MEL_STRIDE] = {\n")
        for s, w in rows:
            vals = ", ".join(C(v) for v in w)
            vals += ", " + ", ".join(["0.0f"] * (stride - len(w)))
            f.write(f"    {{{vals}}},\n")
        f.write("};\n\n// Savitzky-Golay order=2 width=9 interior (simetrico; bordas = mesma janela nos 9 1os/ultimos)\n")
        f.write("static const float MFCC_SG2[9] = {%s};\n" % ", ".join(C(v) for v in w2))
    print("wrote mfcc_tables.h")

    # --- 2. vetores de teste: 2 s de adrian_s1, caminho int16 igual ao device ---
    pcm, sr = sf.read(ROOT / "audio_exemplo/locutor_adrian_s1.wav", dtype="int16")
    assert sr == TARGET_SR, sr
    pcm = np.asarray(pcm[:WIN_SAMPLES], dtype=np.int16)
    assert len(pcm) == WIN_SAMPLES
    y = pcm.astype(np.float64) / 32768.0
    m, yf = pipeline(y)
    print("expected mfcc shape:", m.shape, "dtype:", m.dtype)
    assert m.shape[1] == 1 + WIN_SAMPLES // HOP_LENGTH, m.shape  # center+zero-pad: 126

    pcm.tofile(TEST_DIR / "test_pcm.bin")
    yf.astype(np.float32).tofile(TEST_DIR / "expected_bp.bin")
    m.astype(np.float32).tofile(TEST_DIR / "expected_mfcc.bin")
    m.mean(axis=1).astype(np.float32).tofile(TEST_DIR / "expected_mean.bin")

    # estágios do frame 37 p/ debug estágio-a-estágio no host test
    T = 37
    D = librosa.stft(yf, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann",
                     center=True, pad_mode="constant")
    P37 = (np.abs(D[:, T]) ** 2).astype(np.float64)
    P37.astype(np.float32).tofile(TEST_DIR / "expected_pow37.bin")
    melmat = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=128,
                                 fmin=0.0, fmax=8000.0, htk=False, norm="slaney")
    M37 = (melmat @ P37).astype(np.float64)
    M37.astype(np.float32).tofile(TEST_DIR / "expected_mel37.bin")
    librosa.power_to_db(M37).astype(np.float32).tofile(TEST_DIR / "expected_db37.bin")
    print("wrote tools/mfcc_test/*.bin")

    # --- 3. enroll vectors p/ F2 (chunks de 125 frames como no notebook) ---
    import re
    from pathlib import Path as P
    LOCUTOR_RE = re.compile(r"^locutor_(?P<nome>.+?)_(?P<sessao>s\d+)$")
    vecs, labels = [], []
    by_sp = {}
    for wav in sorted(P(ROOT / "audio_exemplo").glob("locutor_*.wav")):
        mt = LOCUTOR_RE.match(wav.stem)
        if not mt:
            continue
        raw, srr = sf.read(str(wav), dtype="float32")
        if srr != TARGET_SR:
            raw = librosa.resample(raw, orig_sr=srr, target_sr=TARGET_SR)
        # Só blocos de 100 ms acima do VAD (== capture_voiced do firmware):
        # a base precisa da mesma distribuição das janelas ao vivo.
        blk = len(raw) // 1600 * 1600
        fr = raw[:blk].reshape(-1, 1600)
        hot = fr[np.sqrt((fr ** 2).mean(axis=1)) * 32768 > 300]
        kept = hot.reshape(-1) if len(hot) else np.array([], dtype=np.float32)
        print(f"{wav.name}: {len(raw)/TARGET_SR:.1f}s -> {len(kept)/TARGET_SR:.1f}s com voz")
        if len(kept) >= 32000:
            by_sp.setdefault(mt.group("nome"), []).append(kept.astype(np.float64))
    for sp in sorted(by_sp):
        mm, _ = pipeline(np.concatenate(by_sp[sp]))  # concatena por locutor, igual ao notebook
        fseg = int(round(2.0 * TARGET_SR / HOP_LENGTH))  # 125, igual ao notebook
        for i in range(0, mm.shape[1] - fseg + 1, fseg):
            vecs.append(mm[:, i:i + fseg].mean(axis=1))
            labels.append(sp)
    vecs = np.array(vecs, dtype=np.float32)
    print("enroll:", vecs.shape, sorted(set(labels)))
    np.savez(TEST_DIR / "enroll_vectors.npz",
             vecs=vecs, labels=np.array(labels),
             meta=np.array(["librosa " + librosa.__version__ + " | chunks 125 frames | mean float32"]))
    print("wrote enroll_vectors.npz")

    # --- 4. defaults de fábrica p/ o firmware (F2): ADRIAN + PEDRO ---
    DB_DIR.mkdir(parents=True, exist_ok=True)
    order = np.argsort(labels, kind="stable")
    with open(DB_DIR / "enroll_default.h", "w") as f:
        f.write("// GERADO por tools/export_mfcc.py — não editar à mão.\n#pragma once\n\n")
        f.write(f"#define ENROLL_DEFAULT_NSPK 2\n#define ENROLL_DEFAULT_NVEC {len(vecs)}\n")
        names = [str(labels[i]) for i in order]
        f.write("static const char *ENROLL_DEFAULT_NAMES[2] = {%s};\n"
                % ", ".join(f'"{n.upper()}"' for n in sorted(set(names))))
        f.write("static const float ENROLL_DEFAULT_VECS[%d][39] = {\n" % len(vecs))
        for i in order:
            f.write("    {%s},\n" % ", ".join(C(v) for v in vecs[i]))
        f.write("};\nstatic const uint8_t ENROLL_DEFAULT_LABELS[%d] = {\n    " % len(vecs))
        name_to_idx = {n: k for k, n in enumerate(sorted(set(n.upper() for n in names)))}
        f.write(", ".join(str(name_to_idx[n.upper()]) for n in (str(labels[i]) for i in order)))
        f.write("\n};\n")
    print("wrote enroll_default.h")


if __name__ == "__main__":
    sys.exit(main())
