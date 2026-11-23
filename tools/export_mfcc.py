#!/usr/bin/env python3
"""Exporta em staging após validar dados; publique com --output-root ."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import librosa
import numpy as np
from scipy import signal
from sklearn.mixture import GaussianMixture
from recvoz.features import (TARGET_SR, VOICE_LOW, VOICE_HIGH, N_FFT, HOP_LENGTH,
                             WIN_SAMPLES, pipeline, load_audio, load_dataset)


def C(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Coeficiente não finito")
    s = f"{value:.9g}"
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"


def generate(staging, pcm, dataset):
    mfcc_dir = staging / "hardware/recvoz-esp/components/mfcc"
    db_dir = staging / "hardware/recvoz-esp/components/speaker_db"
    test_dir = staging / "tests/fixtures"
    for d in (mfcc_dir, db_dir, test_dir):
        d.mkdir(parents=True)
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

    # DCT-II ortonormal 13x128 (== scipy.fft.dct(type=2, norm='ortho')):
    # evita 209k cos() em double (emulado, ~4 s) no S3.
    n = np.arange(128)[None, :]
    k = np.arange(13)[:, None]
    raw = 2 * np.cos(np.pi * (2 * n + 1) * k / 256.0)
    dct = raw / np.sqrt(2 * 128)
    dct[0] /= np.sqrt(2)

    with open(mfcc_dir / "mfcc_tables.h", "w") as f:
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
        f.write("\n// DCT-II ortho 13x128 (dot float no firmware; sem cos emulado)\n")
        f.write("static const float MFCC_DCT[13][128] = {\n")
        for row in dct:
            f.write("    {%s},\n" % ", ".join(C(v) for v in row))
        f.write("};\n")
    print("wrote mfcc_tables.h")

    m, yf = pipeline(pcm.astype(np.float64) / 32768)
    pcm.astype("<i2").tofile(test_dir / "test_pcm.bin")
    for name, data in [("expected_bp", yf), ("expected_mfcc", m), ("expected_mean", m.mean(axis=1))]:
        data.astype("<f4").tofile(test_dir / (name + ".bin"))
    p37 = np.abs(librosa.stft(yf, n_fft=N_FFT, hop_length=HOP_LENGTH,
                               center=True, pad_mode="constant")[:, 37]) ** 2
    mel37 = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=128) @ p37
    for name, data in [("expected_pow37", p37), ("expected_mel37", mel37),
                       ("expected_db37", librosa.power_to_db(mel37))]:
        data.astype("<f4").tofile(test_dir / (name + ".bin"))
    names = sorted(set(dataset.labels))
    labels = [names.index(n) for n in dataset.labels]
    with (db_dir / "enroll_default.h").open('w') as f:
        f.write('// GERADO por tools/export_mfcc.py; requer recalibrar tau.\n#pragma once\n#include <stdint.h>\n')
        f.write(f'#define ENROLL_DEFAULT_NSPK {len(names)}\n#define ENROLL_DEFAULT_NVEC {len(labels)}\n')
        f.write(f'static const char *ENROLL_DEFAULT_NAMES[{len(names)}] = {{' + ', '.join(json.dumps(n) for n in names) + '};\n')
        f.write(f'static const float ENROLL_DEFAULT_VECS[{len(labels)}][39] = {{\n')
        for row in dataset.vectors:
            f.write('    {' + ', '.join(C(v) for v in row) + '},\n')
        f.write('};\n')
        f.write(f'static const uint8_t ENROLL_DEFAULT_LABELS[{len(labels)}] = {{' + ', '.join(map(str, labels)) + '};\n')
    np.savez(test_dir / 'enroll_vectors.npz', vecs=dataset.vectors, labels=dataset.labels,
             sessions=dataset.sessions, sources=dataset.sources)

    # --- 3. GMM-UBM + CMN (8 Gaussianas diagonais em 24 dims: c1..c12 + Δc1..Δc12) ---
    idx24 = np.r_[1:13, 14:26]
    def cmn24(m39):
        X = m39[idx24, :].T.astype(np.float64)
        return X - X.mean(axis=0, keepdims=True)

    X_all = np.vstack([cmn24(m) for m in dataset.frames])
    ubm = GaussianMixture(n_components=8, covariance_type='diag', random_state=42, max_iter=100).fit(X_all)
    ubm_logw = np.log(ubm.weights_ + 1e-12)
    ubm_invvar = 1.0 / np.maximum(ubm.covariances_, 1e-6)
    ubm_logconst = -0.5 * np.sum(np.log(2.0 * np.pi * np.maximum(ubm.covariances_, 1e-6)), axis=1)

    r_map = 8.0
    spk_means = []
    for spk_name in names:
        Xs = np.vstack([cmn24(m) for m, lbl in zip(dataset.frames, dataset.labels) if lbl == spk_name])
        resp = ubm.predict_proba(Xs)
        nk = resp.sum(axis=0) + 1e-12
        ek = (resp.T @ Xs) / nk[:, None]
        alpha = nk / (nk + r_map)
        spk_means.append(alpha[:, None] * ek + (1.0 - alpha[:, None]) * ubm.means_)

    with (db_dir / 'gmm_default.h').open('w') as f:
        f.write('// GERADO por tools/export_mfcc.py — GMM-UBM (M=8, D=24, CMN).\n#pragma once\n')
        f.write('#define GMM_M 8\n#define GMM_D 24\n#define GMM_R 8.0f\n')
        f.write(f'#define GMM_NSPK {len(names)}\n')
        f.write('static const float GMM_UBM_LOGW[GMM_M] = {%s};\n' % ', '.join(C(v) for v in ubm_logw))
        f.write('static const float GMM_UBM_LOGCONST[GMM_M] = {%s};\n' % ', '.join(C(v) for v in ubm_logconst))
        f.write('static const float GMM_UBM_MEANS[GMM_M][GMM_D] = {\n')
        for row in ubm.means_:
            f.write('    {' + ', '.join(C(v) for v in row) + '},\n')
        f.write('};\nstatic const float GMM_UBM_INVVAR[GMM_M][GMM_D] = {\n')
        for row in ubm_invvar:
            f.write('    {' + ', '.join(C(v) for v in row) + '},\n')
        f.write('};\nstatic const float GMM_SPK_MEANS[GMM_NSPK][GMM_M][GMM_D] = {\n')
        for mat in spk_means:
            f.write('    {\n')
            for row in mat:
                f.write('        {' + ', '.join(C(v) for v in row) + '},\n')
            f.write('    },\n')
        f.write('};\n')
    print("wrote gmm_default.h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--fixture-wav', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--vad', type=int, default=300)
    args = parser.parse_args()
    try:
        if librosa.__version__ != '1.0.0':
            raise ValueError('Export exige librosa==1.0.0')
        pcm = load_audio(args.fixture_wav)[:WIN_SAMPLES]
        if len(pcm) != WIN_SAMPLES:
            raise ValueError('Fixture exige pelo menos 2 s de áudio')
        dataset = load_dataset(args.audio_dir, args.vad)
        if not 1 <= len(set(dataset.labels)) <= 8:
            raise ValueError('Firmware suporta 1..8 locutores de fábrica (máscara uint8)')
        if not np.isfinite(dataset.vectors).all():
            raise ValueError('Features não finitas')
        with tempfile.TemporaryDirectory(prefix='recvoz-export-') as tmp:
            staging = Path(tmp)
            generate(staging, pcm, dataset)
            manifest = {'pipeline': 'pcm-window-32000-v1', 'librosa': librosa.__version__,
                        'vad': args.vad, 'noise': 'auto-session-profile', 'skip_c0': True,
                        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(args.audio_dir.glob('locutor_*.wav'))},
                        'fixture_sha256': hashlib.sha256(args.fixture_wav.read_bytes()).hexdigest()}
            (staging / 'tests/fixtures/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            # Todos os dados foram validados e gerados antes de publicar.
            for source in sorted(staging.rglob('*')):
                if source.is_file():
                    dest = args.output_root / source.relative_to(staging)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, dest.with_suffix(dest.suffix + '.tmp'))
                    dest.with_suffix(dest.suffix + '.tmp').replace(dest)
        print(f'Exportados {len(dataset.vectors)} vetores em {args.output_root}. Recalibre tau antes de flashar.')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
