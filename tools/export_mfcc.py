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
                             N_MELS, WIN_SAMPLES, pipeline, load_audio, load_dataset)


def C(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Coeficiente não finito")
    s = f"{value:.9g}"
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"


def generate(staging, pcm, dataset, ubm_frames=None):
    mfcc_dir = staging / "hardware/recvoz-esp/components/mfcc"
    gmm_dir = staging / "hardware/recvoz-esp/components/gmm_ubm/include"
    db_dir = staging / "hardware/recvoz-esp/components/speaker_db"
    test_dir = staging / "tests/fixtures"
    for d in (mfcc_dir, gmm_dir, db_dir, test_dir):
        d.mkdir(parents=True)
    # --- 1. tabelas fixas ---
    nyq = TARGET_SR / 2.0
    sos = signal.butter(5, [VOICE_LOW / nyq, min(VOICE_HIGH / nyq, 0.99)],
                        btype="bandpass", output="sos")
    zi = signal.sosfilt_zi(sos)
    assert sos.shape == (5, 6), sos.shape

    mel = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS,
                              fmin=0.0, fmax=8000.0, htk=False, norm="slaney")
    assert mel.shape == (N_MELS, 513), mel.shape
    # Pesos exatos (CSR): triângulos amostrados não zeram nas bordas,
    # então copiamos os valores em vez de reconstruir rampas.
    rows = []
    for m in range(N_MELS):
        nz = np.nonzero(mel[m] > 0)[0]
        rows.append((int(nz[0]), mel[m][nz[0]:nz[-1] + 1].astype(np.float64)))
    stride = max(len(w) for _, w in rows)
    print("mel stride:", stride)

    w2 = signal.savgol_coeffs(9, 2, 2)  # device usa correlacao; janela simetrica
    assert np.allclose(w2, w2[::-1]), "SG order2 deveria ser simetrico"
    print("SG order2 interior:", np.round(w2, 6))

    # DCT-II ortonormal 13xN_MELS (== scipy.fft.dct(type=2, norm='ortho')):
    # evita cos() em double (emulado) no ESP32-S3.
    n = np.arange(N_MELS)[None, :]
    k = np.arange(13)[:, None]
    raw = 2 * np.cos(np.pi * (2 * n + 1) * k / (2.0 * N_MELS))
    dct = raw / np.sqrt(2 * N_MELS)
    dct[0] /= np.sqrt(2)

    with open(mfcc_dir / "mfcc_tables.h", "w") as f:
        f.write("// GERADO por tools/export_mfcc.py — não editar à mão.\n")
        f.write(f"// librosa {librosa.__version__} | sr=16000 bandpass 80-7920 SOS(5) mel{N_MELS} slaney\n#pragma once\n")
        f.write("#include <stdint.h>\n\n#define MFCC_N_SOS 5\n");
        f.write("static const float MFCC_SOS[5][6] = {\n")
        for row in sos:
            f.write("    {%s},\n" % ", ".join(C(v) for v in row))
        f.write("};\nstatic const float MFCC_SOS_ZI[5][2] = {\n")
        for row in zi:
            f.write("    {%s},\n" % ", ".join(C(v) for v in row))
        f.write(f"}};\n\n#define MFCC_N_MELS {N_MELS}\n")
        f.write(f"#define MFCC_MEL_STRIDE {stride}\n")
        f.write(f"static const uint16_t MFCC_MEL_START[{N_MELS}] = {{\n    ")
        f.write(", ".join(str(s) for s, _ in rows))
        f.write(f"\n}};\nstatic const uint8_t MFCC_MEL_LEN[{N_MELS}] = {{\n    ")
        f.write(", ".join(str(len(w)) for _, w in rows))
        f.write(f"\n}};\nstatic const float MFCC_MEL_W[{N_MELS}][MFCC_MEL_STRIDE] = {{\n")
        for s, w in rows:
            vals = ", ".join(C(v) for v in w)
            if stride > len(w):
                vals += ", " + ", ".join(["0.0f"] * (stride - len(w)))
            f.write(f"    {{{vals}}},\n")
        f.write("};\n\n// Savitzky-Golay order=2 width=9 interior (simetrico; bordas = mesma janela nos 9 1os/ultimos)\n")
        f.write("static const float MFCC_SG2[9] = {%s};\n" % ", ".join(C(v) for v in w2))
        f.write(f"\n// DCT-II ortho 13x{N_MELS} (dot float no firmware; sem cos emulado)\n")
        f.write(f"static const float MFCC_DCT[13][{N_MELS}] = {{\n")
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
    mel37 = librosa.filters.mel(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS) @ p37
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

    # --- 3. GMM-UBM + Frame-VAD (top 70% c0) + CMN-c1 (8 Gaussianas diagonais em 24 dims) ---
    idx24 = np.r_[1:13, 14:26]
    def cmn_c1(m39):
        c0 = m39[0, :]
        k_keep = max(10, int(round(len(c0) * 0.70)))
        idx_keep = np.sort(np.argsort(c0)[-k_keep:])
        X = m39[idx24, :][:, idx_keep].T.astype(np.float64).copy()
        X[:, 0] -= X[:, 0].mean()
        return X

    pool = ubm_frames if ubm_frames is not None else dataset.frames
    X_all = np.vstack([cmn_c1(m) for m in pool])
    ubm = GaussianMixture(n_components=8, covariance_type='diag', reg_covar=1e-4, random_state=42, max_iter=200).fit(X_all)
    ubm_logw = np.log(ubm.weights_ + 1e-12)
    ubm_invvar = 1.0 / np.maximum(ubm.covariances_, 1e-6)
    ubm_logconst = -0.5 * np.sum(np.log(2.0 * np.pi * np.maximum(ubm.covariances_, 1e-6)), axis=1)

    r_map = 8.0
    spk_means = []
    for spk_name in names:
        Xs = np.vstack([cmn_c1(m) for m, lbl in zip(dataset.frames, dataset.labels) if lbl == spk_name])
        resp = ubm.predict_proba(Xs)
        nk = resp.sum(axis=0) + 1e-12
        ek = (resp.T @ Xs) / nk[:, None]
        alpha = nk / (nk + r_map)
        spk_means.append(alpha[:, None] * ek + (1.0 - alpha[:, None]) * ubm.means_)

    with (gmm_dir / 'gmm_default.h').open('w') as f:
        src_tag = 'FLEURS PT-BR' if ubm_frames is not None else 'local'
        f.write(f'// GERADO por tools/export_mfcc.py — GMM-UBM (M=8, D=24, CMN-c1, UBM={src_tag}).\n#pragma once\n')
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
    parser.add_argument('--ubm-dir', type=Path, help='Diretório opcional com WAVs externos (ex.: FLEURS PT-BR) para o UBM')
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
        ubm_frames = None
        if args.ubm_dir:
            from recvoz.features import auto_noise_profile, voiced_windows
            ubm_frames = []
            for wp in sorted(args.ubm_dir.glob('*.wav')):
                upcm = load_audio(wp)
                uns = auto_noise_profile(upcm, args.vad)
                for win in voiced_windows(upcm, args.vad):
                    mf, _ = pipeline(win.astype(np.float64) / 32768.0, noise=uns, alpha=0.8)
                    ubm_frames.append(mf)
                    if len(ubm_frames) >= 350:
                        break
                if len(ubm_frames) >= 350:
                    break
        with tempfile.TemporaryDirectory(prefix='recvoz-export-') as tmp:
            staging = Path(tmp)
            generate(staging, pcm, dataset, ubm_frames=ubm_frames)
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
