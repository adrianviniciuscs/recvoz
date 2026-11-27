"""Sessões inteiras em treino/calibração/teste; limiar nunca vê o teste."""
import argparse
import json
import hashlib
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from .features import load_dataset, load_audio, noise_profile


def session_split(labels, sessions, seed=42, calibration=False):
    rng = np.random.default_rng(seed)
    train, cal, test = [], [], []
    for speaker in np.unique(labels):
        groups = np.unique(sessions[labels == speaker])
        minimum = 3 if calibration else 2
        if len(groups) < minimum:
            raise ValueError(f"{speaker}: precisa de {minimum} sessões, tem {len(groups)}")
        groups = rng.permutation(groups)
        test.extend(np.flatnonzero((labels == speaker) & (sessions == groups[0])))
        if calibration:
            cal.extend(np.flatnonzero((labels == speaker) & (sessions == groups[1])))
        train.extend(np.flatnonzero((labels == speaker) & np.isin(sessions, groups[minimum-1:])))
    return tuple(np.array(a, dtype=int) for a in (train, cal, test))


def distances(x, reference):
    x, reference = np.asarray(x, dtype=float), np.asarray(reference, dtype=float)
    norms = np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(reference, axis=1)[None, :]
    cosine = np.divide(x @ reference.T, norms, out=np.zeros_like(norms), where=norms > 0)
    return 1 - np.clip(cosine, -1, 1)


def classify(train_x, train_y, query, tau=2):
    d = distances(query[:, 1:], train_x[:, 1:])  # remove c0
    model = KNeighborsClassifier(n_neighbors=min(3, len(train_y)), metric='precomputed', weights='distance')
    model.fit(distances(train_x[:, 1:], train_x[:, 1:]), train_y)
    predicted = model.predict(d).astype(object)
    nearest = d.min(axis=1)
    predicted[nearest > tau] = 'DESCONHECIDO'
    return predicted, nearest


def closed_set(dataset, seed=42):
    train, _, test = session_split(dataset.labels, dataset.sessions, seed)
    pred, _ = classify(dataset.vectors[train], dataset.labels[train], dataset.vectors[test])
    names = sorted(set(dataset.labels))
    return {'seed': seed, 'accuracy': float(accuracy_score(dataset.labels[test], pred)),
            'labels': names, 'confusion_matrix': confusion_matrix(dataset.labels[test], pred, labels=names).tolist(),
            'train_sessions': sorted(set(dataset.sessions[train])),
            'test_sessions': sorted(set(dataset.sessions[test])), 'test_windows': len(test)}


def open_set(dataset, unknown_cal, unknown_test, seed=42):
    known = set(dataset.labels)
    if known & set(unknown_cal.labels) or known & set(unknown_test.labels) or set(unknown_cal.labels) & set(unknown_test.labels):
        raise ValueError('Locutores conhecidos, desconhecidos de calibração e desconhecidos de teste devem ser distintos')
    train, cal, test = session_split(dataset.labels, dataset.sessions, seed, calibration=True)
    x, y = dataset.vectors[train], dataset.labels[train]
    _, kc = classify(x, y, dataset.vectors[cal])
    _, uc = classify(x, y, unknown_cal.vectors)
    candidates = np.unique(np.clip(np.r_[kc, uc, 1e-7, np.nextafter(np.float32(1), np.float32(0))],
                                   1e-7, np.nextafter(np.float32(1), np.float32(0))))
    # Minimiza erro balanceado; empate favorece o limiar mais restritivo.
    losses = [((kc > t).mean() + (uc <= t).mean()) / 2 for t in candidates]
    tau = float(candidates[np.argmin(losses)])
    pred, kd = classify(x, y, dataset.vectors[test], tau)
    _, ud = classify(x, y, unknown_test.vectors, tau)
    return {'tau': tau, 'calibration_balanced_error': float(min(losses)),
            'false_accept_rate': float((ud <= tau).mean()),
            'false_reject_rate': float((kd > tau).mean()),
            'known_identification_accuracy': float((pred == dataset.labels[test]).mean()),
            'known_test_windows': len(test), 'unknown_test_windows': len(ud),
            'train_sessions': sorted(set(dataset.sessions[train])),
            'calibration_sessions': sorted(set(dataset.sessions[cal])),
            'test_sessions': sorted(set(dataset.sessions[test])),
            'unknown_calibration_sessions': sorted(set(unknown_cal.sessions)),
            'unknown_test_sessions': sorted(set(unknown_test.sessions))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--unknown-cal', type=Path)
    parser.add_argument('--unknown-test', type=Path)
    parser.add_argument('--noise-wav', type=Path, help='Silêncio independente: primeiras 8192 amostras, perfil fixo')
    parser.add_argument('--vad', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if bool(args.unknown_cal) != bool(args.unknown_test):
            raise ValueError('Forneça --unknown-cal e --unknown-test juntos')
        noise = noise_profile(load_audio(args.noise_wav)[:8192]) if args.noise_wav else None
        data = load_dataset(args.audio_dir, args.vad, noise)
        report = {'protocol': 'session-holdout-v1', 'vad': args.vad, 'skip_c0': True,
                  'noise': 'fixed-profile' if noise is not None else 'auto-session-profile',
                  'versions': {p: version(p) for p in ['numpy', 'scipy', 'scikit-learn']},
                  'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.audio_dir.glob('locutor_*.wav'))},
                  'closed_set': closed_set(data, args.seed)}
        if args.unknown_cal:
            report['open_set'] = open_set(data, load_dataset(args.unknown_cal, args.vad, noise),
                                         load_dataset(args.unknown_test, args.vad, noise), args.seed)
        else:
            report['open_set'] = {'status': 'not_evaluated', 'reason': 'Faltam conjuntos de desconhecidos; exige 3 sessões por conhecido.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
