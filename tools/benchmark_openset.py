#!/usr/bin/env python3
"""Benchmark completo Open-Set (5 modelos) e Estudo de Ablação GMM-UBM + CMN.

Usa protocolo estrito sem vazamento de sessão nem de locutor:
- Partição principal: Conhecidos (audio_exemplo) vs Imp. Cal (desconhecidos_cal) vs Imp. Teste (desconhecidos_teste)
- Validação cruzada rotacionada (múltiplas seeds / splits de sessão) para robustez estatística
- Gera docs/results/benchmark-openset.json e figuras 300 DPI em docs/artigo/figs/
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recvoz.evaluation import distances, session_split  # noqa: E402
from recvoz.features import auto_noise_profile, load_audio, pipeline, voiced_windows  # noqa: E402


def load_sessions_custom(directory, vad=300, use_ns=True, alpha=0.8):
    """Carrega sessões com controle explícito de supressão espectral."""
    import re
    pat = re.compile(r"^locutor_(?P<speaker>[a-zA-Z0-9_]+)_(?P<session>s\d+)\.wav$")
    vectors, labels, sessions, frames = [], [], [], []
    for path in sorted(Path(directory).glob("locutor_*.wav")):
        m = pat.fullmatch(path.name)
        if not m:
            continue
        spk = m["speaker"].upper()
        sess = f"{spk}/{m['session']}"
        pcm = load_audio(path)
        ns = auto_noise_profile(pcm, vad) if use_ns else None
        for win in voiced_windows(pcm, vad):
            feat, _ = pipeline(win.astype(np.float64) / 32768.0, noise=ns, alpha=alpha)
            frames.append(feat)  # (39, 126)
            vectors.append(feat.mean(axis=1))
            labels.append(spk)
            sessions.append(sess)
    return {
        "vectors": np.asarray(vectors, dtype=np.float64),
        "labels": np.asarray(labels),
        "sessions": np.asarray(sessions),
        "frames": frames,
    }


def extract_gmm_features(frame_mat, norm="cmn", use_deltas=True, keep_frac=0.70):
    """(39, T) -> (T', D) com Frame-VAD (top keep_frac de c0), c1..c12 (+ delta) e normalização."""
    if keep_frac < 1.0:
        c0 = frame_mat[0]
        k_keep = max(10, int(round(len(c0) * keep_frac)))
        idx_keep = np.sort(np.argsort(c0)[-k_keep:])
        frame_mat = frame_mat[:, idx_keep]
    if use_deltas:
        x = np.vstack([frame_mat[1:13], frame_mat[14:26]]).T.copy()  # (T', 24)
    else:
        x = frame_mat[1:13].T.copy()  # (T', 12)
    if norm == "cmn":
        x = x - x.mean(axis=0, keepdims=True)
    elif norm == "cmn_c1":
        x[:, 0] -= x[:, 0].mean()
    elif norm == "cmvn":
        mu = x.mean(axis=0, keepdims=True)
        sd = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
        x = (x - mu) / sd
    return x


def load_external_ubm_frames(directory, max_windows=350, vad=300):
    """Carrega janelas de um corpus externo (ex.: Google FLEURS PT-BR) para o UBM."""
    frames = []
    for p in sorted(Path(directory).glob("*.wav")):
        pcm = load_audio(p)
        ns = auto_noise_profile(pcm, vad)
        for w in voiced_windows(pcm, vad):
            feat, _ = pipeline(w.astype(np.float64) / 32768.0, noise=ns, alpha=0.8)
            frames.append(feat)
            if len(frames) >= max_windows:
                return frames
    return frames


class GMMUBMClassifier:
    """GMM-UBM diagonal com adaptação MAP de médias (equivalente ao firmware C)."""

    def __init__(self, n_components=8, relevance=8.0, norm="cmn", use_deltas=True, random_state=42):
        self.n_components = n_components
        self.relevance = relevance
        self.norm = norm
        self.use_deltas = use_deltas
        self.random_state = random_state

    def _feat(self, frames_list):
        return [extract_gmm_features(f, self.norm, self.use_deltas) for f in frames_list]

    def fit(self, train_frames, train_labels, ubm_frames=None):
        X_list = self._feat(train_frames)
        ubm_pool = self._feat(ubm_frames) if ubm_frames is not None else X_list
        X_ubm = np.vstack(ubm_pool)

        self.ubm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            max_iter=200,
            random_state=self.random_state,
        ).fit(X_ubm)

        self.classes_ = sorted(set(train_labels))
        self.means_ = {}
        w = self.ubm_.weights_
        mu_u = self.ubm_.means_
        var_u = self.ubm_.covariances_

        for c in self.classes_:
            Xc = np.vstack([x for x, lab in zip(X_list, train_labels) if lab == c])
            gamma = self.ubm_.predict_proba(Xc)  # (N, M)
            nm = gamma.sum(axis=0)  # (M,)
            sm = gamma.T @ Xc       # (M, D)
            alpha = nm / (nm + self.relevance)
            adapted = np.where(
                nm[:, None] > 1e-9,
                alpha[:, None] * (sm / np.maximum(nm[:, None], 1e-9)) + (1.0 - alpha[:, None]) * mu_u,
                mu_u,
            )
            self.means_[c] = adapted
        self._log_w = np.log(np.maximum(w, 1e-30))
        self._inv_var = 1.0 / np.maximum(var_u, 1e-9)
        self._const = -0.5 * np.sum(np.log(2.0 * np.pi * np.maximum(var_u, 1e-9)), axis=1)
        return self

    def _log_prob(self, X, means):
        # X: (T, D), means: (M, D) -> (T,)
        diff = X[:, None, :] - means[None, :, :]
        ll = self._log_w[None, :] + self._const[None, :] - 0.5 * np.sum((diff ** 2) * self._inv_var[None, :, :], axis=2)
        m = ll.max(axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.sum(np.exp(ll - m), axis=1))).mean()

    def score_windows(self, frames_list):
        """Retorna (pred_labels, best_scores, margins) onde maior score = mais conhecido."""
        preds, best_scores, margins = [], [], []
        for f in frames_list:
            X = extract_gmm_features(f, self.norm, self.use_deltas)
            l_ubm = self._log_prob(X, self.ubm_.means_)
            llrs = {c: float(self._log_prob(X, self.means_[c]) - l_ubm) for c in self.classes_}
            ranked = sorted(llrs.items(), key=lambda kv: kv[1], reverse=True)
            preds.append(ranked[0][0])
            best_scores.append(ranked[0][1])
            margins.append(ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1])
        return np.array(preds), np.array(best_scores), np.array(margins)


class IndependentGMMClassifier:
    """GMM diagonal independente por locutor (sem UBM)."""

    def __init__(self, n_components=8, norm="cmn", use_deltas=True, random_state=42):
        self.n_components = n_components
        self.norm = norm
        self.use_deltas = use_deltas
        self.random_state = random_state

    def fit(self, train_frames, train_labels):
        self.classes_ = sorted(set(train_labels))
        self.models_ = {}
        for c in self.classes_:
            Xc = np.vstack([extract_gmm_features(f, self.norm, self.use_deltas)
                            for f, lab in zip(train_frames, train_labels) if lab == c])
            self.models_[c] = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                reg_covar=1e-4,
                max_iter=200,
                random_state=self.random_state,
            ).fit(Xc)
        return self

    def score_windows(self, frames_list):
        preds, best_scores = [], []
        for f in frames_list:
            X = extract_gmm_features(f, self.norm, self.use_deltas)
            scores = {c: float(m.score(X)) for c, m in self.models_.items()}
            best_c, best_s = max(scores.items(), key=lambda kv: kv[1])
            preds.append(best_c)
            best_scores.append(best_s)
        return np.array(preds), np.array(best_scores)


def compute_eer(known_scores, unknown_scores):
    """Scores maiores = conhecido. Retorna (eer, eer_threshold, auc, fpr, tpr)."""
    y_true = np.r_[np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    y_score = np.r_[known_scores, unknown_scores]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx]), auc, fpr, tpr


def calibrate_threshold(cal_known_scores, cal_unknown_scores):
    """Escolhe limiar (score >= tau -> conhecido) minimizando erro balanceado na calibração."""
    cands = np.unique(np.r_[cal_known_scores, cal_unknown_scores])
    # pontos médios entre candidatos adjacentes para melhor generalização
    mids = (cands[:-1] + cands[1:]) / 2.0 if len(cands) > 1 else cands
    all_t = np.r_[cands[0] - 1e-6, mids, cands[-1] + 1e-6]
    losses = [((cal_known_scores < t).mean() + (cal_unknown_scores >= t).mean()) / 2.0 for t in all_t]
    best = int(np.argmin(losses))
    return float(all_t[best]), float(losses[best])


def evaluate_model_scores(cal_k, cal_u, test_k_pred, test_k_true, test_k_scores, test_u_scores):
    tau, cal_err = calibrate_threshold(cal_k, cal_u)
    far = float((test_u_scores >= tau).mean())
    frr = float((test_k_scores < tau).mean())
    closed_acc = float((test_k_pred == test_k_true).mean())
    open_known_acc = float(((test_k_pred == test_k_true) & (test_k_scores >= tau)).mean())
    eer, _, auc, fpr, tpr = compute_eer(test_k_scores, test_u_scores)
    return {
        "tau": tau,
        "cal_balanced_err": cal_err,
        "closed_acc": closed_acc,
        "open_known_acc": open_known_acc,
        "far": far,
        "frr": frr,
        "eer": eer,
        "auc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def run_all_models_on_split(known_ds, cal_u_ds, test_u_ds, train_idx, cal_idx, test_idx, ext_ubms=None):
    X_tr, y_tr = known_ds["vectors"][train_idx], known_ds["labels"][train_idx]
    X_cal = known_ds["vectors"][cal_idx]
    X_te, y_te = known_ds["vectors"][test_idx], known_ds["labels"][test_idx]
    F_tr = [known_ds["frames"][i] for i in train_idx]
    F_cal = [known_ds["frames"][i] for i in cal_idx]
    F_te = [known_ds["frames"][i] for i in test_idx]

    results = {}

    # 1. Centroide Cosseno (sem c0)
    classes = sorted(set(y_tr))
    centroids = np.array([X_tr[y_tr == c, 1:].mean(axis=0) for c in classes])
    def score_centroid(X):
        d = distances(X[:, 1:], centroids)
        idx = d.argmin(axis=1)
        return np.array([classes[i] for i in idx]), -d.min(axis=1)
    _, ck = score_centroid(X_cal)
    _, cu = score_centroid(cal_u_ds["vectors"])
    tp, tk = score_centroid(X_te)
    _, tu = score_centroid(test_u_ds["vectors"])
    results["Centroide Cosseno"] = evaluate_model_scores(ck, cu, tp, y_te, tk, tu)

    # 2. k-NN Cosseno (k=3, sem c0)
    knn = KNeighborsClassifier(n_neighbors=min(3, len(y_tr)), metric="precomputed", weights="distance")
    knn.fit(distances(X_tr[:, 1:], X_tr[:, 1:]), y_tr)
    def score_knn(X):
        d = distances(X[:, 1:], X_tr[:, 1:])
        return knn.predict(d), -d.min(axis=1)
    _, ck = score_knn(X_cal)
    _, cu = score_knn(cal_u_ds["vectors"])
    tp, tk = score_knn(X_te)
    _, tu = score_knn(test_u_ds["vectors"])
    results["k-NN Cosseno (k=3)"] = evaluate_model_scores(ck, cu, tp, y_te, tk, tu)

    # 3. SVM RBF (médias MFCC sem c0; score = margem absoluta da fronteira de decisão)
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=42))
    svm.fit(X_tr[:, 1:], y_tr)
    def score_svm(X):
        pred = svm.predict(X[:, 1:])
        df = svm.decision_function(X[:, 1:])
        score = np.abs(df) if df.ndim == 1 else df.max(axis=1)
        return pred, score
    _, ck = score_svm(X_cal)
    _, cu = score_svm(cal_u_ds["vectors"])
    tp, tk = score_svm(X_te)
    _, tu = score_svm(test_u_ds["vectors"])
    results["SVM (RBF)"] = evaluate_model_scores(ck, cu, tp, y_te, tk, tu)

    # 4. GMM independente + CMN (sem UBM)
    igmm = IndependentGMMClassifier(n_components=8, norm="cmn", use_deltas=True, random_state=42).fit(F_tr, y_tr)
    _, ck = igmm.score_windows(F_cal)
    _, cu = igmm.score_windows(cal_u_ds["frames"])
    tp, tk = igmm.score_windows(F_te)
    _, tu = igmm.score_windows(test_u_ds["frames"])
    results["GMM Indep. + CMN"] = evaluate_model_scores(ck, cu, tp, y_te, tk, tu)

    # 5. GMM-UBM + CMN (UBM interno de 2 locutores)
    gmm_ubm = GMMUBMClassifier(n_components=8, relevance=8.0, norm="cmn", use_deltas=True, random_state=42).fit(
        F_tr, y_tr, ubm_frames=F_tr
    )
    _, ck, _ = gmm_ubm.score_windows(F_cal)
    _, cu, _ = gmm_ubm.score_windows(cal_u_ds["frames"])
    tp, tk, tm = gmm_ubm.score_windows(F_te)
    _, tu, _ = gmm_ubm.score_windows(test_u_ds["frames"])
    res_ubm = evaluate_model_scores(ck, cu, tp, y_te, tk, tu)
    res_ubm["test_known_scores"] = tk.tolist()
    res_ubm["test_unknown_scores"] = tu.tolist()
    res_ubm["min_margin"] = float(tm.min())
    res_ubm["mean_margin"] = float(tm.mean())
    results["GMM-UBM + CMN (2 loc.)"] = res_ubm

    # 6 & 7. GMM-UBM com UBM externo pré-ajustado (FLEURS PT-BR): CMN seletiva em c1 e sem CMN
    if ext_ubms:
        for label, (norm_mode, prefit_ubm) in ext_ubms.items():
            clf = GMMUBMClassifier(n_components=8, relevance=8.0, norm=norm_mode, use_deltas=True, random_state=42)
            clf.ubm_ = prefit_ubm
            clf.classes_ = sorted(set(y_tr))
            clf.means_ = {}
            w, mu_u, var_u = prefit_ubm.weights_, prefit_ubm.means_, prefit_ubm.covariances_
            X_list = clf._feat(F_tr)
            for c in clf.classes_:
                Xc = np.vstack([x for x, lab in zip(X_list, y_tr) if lab == c])
                gamma = prefit_ubm.predict_proba(Xc)
                nm = gamma.sum(axis=0)
                sm = gamma.T @ Xc
                alpha = nm / (nm + clf.relevance)
                clf.means_[c] = np.where(
                    nm[:, None] > 1e-9,
                    alpha[:, None] * (sm / np.maximum(nm[:, None], 1e-9)) + (1.0 - alpha[:, None]) * mu_u,
                    mu_u,
                )
            clf._log_w = np.log(np.maximum(w, 1e-30))
            clf._inv_var = 1.0 / np.maximum(var_u, 1e-9)
            clf._const = -0.5 * np.sum(np.log(2.0 * np.pi * np.maximum(var_u, 1e-9)), axis=1)
            _, ck, _ = clf.score_windows(F_cal)
            _, cu, _ = clf.score_windows(cal_u_ds["frames"])
            tp, tk, tm = clf.score_windows(F_te)
            _, tu, _ = clf.score_windows(test_u_ds["frames"])
            res_e = evaluate_model_scores(ck, cu, tp, y_te, tk, tu)
            res_e["test_known_scores"] = tk.tolist()
            res_e["test_unknown_scores"] = tu.tolist()
            res_e["min_margin"] = float(tm.min())
            res_e["mean_margin"] = float(tm.mean())
            results[label] = res_e

    return results


def run_ablation(known_ns, cal_u_ns, test_u_ns, known_raw, cal_u_raw, test_u_raw, train_idx, cal_idx, test_idx):
    configs = [
        ("GMM-UBM (M=8, CMN, c+Δ, NS α=0,8)", 8, "cmn", True, True),
        ("GMM-UBM (M=4, CMN, c+Δ, NS)", 4, "cmn", True, True),
        ("GMM-UBM (M=12, CMN, c+Δ, NS)", 12, "cmn", True, True),
        ("GMM-UBM (M=16, CMN, c+Δ, NS)", 16, "cmn", True, True),
        ("GMM-UBM (M=32, CMN, c+Δ, NS)", 32, "cmn", True, True),
        ("Sem CMN (M=8, c+Δ, NS)", 8, "none", True, True),
        ("Com CMVN (M=8, c+Δ, NS)", 8, "cmvn", True, True),
        ("Só estáticos (M=8, CMN, c1..c12)", 8, "cmn", False, True),
        ("Sem NS (M=8, CMN, c+Δ, NS off)", 8, "cmn", True, False),
    ]
    out = []
    for name, m_comp, norm, use_d, use_ns in configs:
        k_ds = known_ns if use_ns else known_raw
        c_ds = cal_u_ns if use_ns else cal_u_raw
        t_ds = test_u_ns if use_ns else test_u_raw

        F_tr = [k_ds["frames"][i] for i in train_idx]
        y_tr = k_ds["labels"][train_idx]
        F_cal = [k_ds["frames"][i] for i in cal_idx]
        F_te = [k_ds["frames"][i] for i in test_idx]
        y_te = k_ds["labels"][test_idx]

        clf = GMMUBMClassifier(n_components=m_comp, relevance=8.0, norm=norm, use_deltas=use_d, random_state=42)
        clf.fit(F_tr, y_tr, ubm_frames=F_tr)
        _, ck, _ = clf.score_windows(F_cal)
        _, cu, _ = clf.score_windows(c_ds["frames"])
        tp, tk, tm = clf.score_windows(F_te)
        _, tu, _ = clf.score_windows(t_ds["frames"])
        metrics = evaluate_model_scores(ck, cu, tp, y_te, tk, tu)
        out.append({
            "config": name,
            "closed_acc": metrics["closed_acc"],
            "far": metrics["far"],
            "frr": metrics["frr"],
            "eer": metrics["eer"],
            "mean_margin": float(tm.mean()),
            "min_margin": float(tm.min()),
        })
    return out


def plot_figures(model_results, out_figs_dir):
    out_figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Curva ROC comparativa
    plt.figure(figsize=(5.8, 4.1), dpi=300)
    styles = [
        ("GMM-UBM (FLEURS, CMN-c1)", "#006837", "-", 2.4),
        ("GMM-UBM (FLEURS, sem CMN)", "#238b45", "--", 2.0),
        ("k-NN Cosseno (k=3)", "#d95f02", "-.", 1.7),
        ("Centroide Cosseno", "#7570b3", ":", 1.7),
        ("GMM-UBM + CMN (2 loc.)", "#2b8cbe", "-", 1.6),
        ("SVM (RBF)", "#e7298a", "--", 1.3),
        ("GMM Indep. + CMN", "#762a83", ":", 1.3),
    ]
    for name, color, ls, lw in styles:
        if name not in model_results:
            continue
        r = model_results[name]
        plt.plot(
            np.array(r["fpr"]) * 100,
            np.array(r["tpr"]) * 100,
            label=f"{name} (EER={r['eer']*100:.1f}%)",
            color=color,
            linestyle=ls,
            linewidth=lw,
        )
    plt.plot([0, 100], [0, 100], color="#999999", linestyle=":", linewidth=1.0)
    plt.xlabel("Taxa de Falsa Aceitação — FAR (%)")
    plt.ylabel("Taxa de Verdadeira Aceitação — 1 - FRR (%)")
    plt.title("Curva ROC em Conjunto Aberto (Open-Set)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=7.5, frameon=True)
    plt.tight_layout()
    plt.savefig(out_figs_dir / "fig_openset_roc.png", dpi=300)
    plt.close()

    # 2. Distribuição de LLR (Conhecidos vs Impostores) no melhor GMM-UBM
    key = "GMM-UBM (FLEURS, CMN-c1)" if "GMM-UBM (FLEURS, CMN-c1)" in model_results else "GMM-UBM + CMN (2 loc.)"
    ubm = model_results[key]
    k_scores = np.array(ubm["test_known_scores"])
    u_scores = np.array(ubm["test_unknown_scores"])
    tau = ubm["tau"]

    plt.figure(figsize=(5.6, 3.6), dpi=300)
    bins = np.linspace(min(k_scores.min(), u_scores.min()) - 0.2, max(k_scores.max(), u_scores.max()) + 0.2, 24)
    plt.hist(k_scores, bins=bins, alpha=0.72, color="#1b7837", label=f"Conhecidos teste (n={len(k_scores)})")
    plt.hist(u_scores, bins=bins, alpha=0.72, color="#d73027", label=f"Impostores teste (n={len(u_scores)})")
    plt.axvline(tau, color="#111111", linestyle="--", linewidth=1.8, label=f"Limiar calibrado τ* = {tau:+.2f}")
    plt.xlabel(f"LLR máximo por janela — {key}")
    plt.ylabel("Contagem de janelas (2 s)")
    plt.title("Separação Open-Set com UBM Externo (FLEURS PT-BR)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8.5, frameon=True)
    plt.tight_layout()
    plt.savefig(out_figs_dir / "fig_openset_llr.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "audio_exemplo")
    parser.add_argument("--unknown-cal", type=Path, default=ROOT / "dados/desconhecidos_cal")
    parser.add_argument("--unknown-test", type=Path, default=ROOT / "dados/desconhecidos_teste")
    parser.add_argument("--ubm-dir", type=Path, default=ROOT / "dados/ubm_fleurs_ptbr/dev")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/benchmark-openset.json")
    parser.add_argument("--figs-dir", type=Path, default=ROOT / "docs/artigo/figs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    known_ns = load_sessions_custom(args.audio_dir, use_ns=True)
    cal_u_ns = load_sessions_custom(args.unknown_cal, use_ns=True)
    test_u_ns = load_sessions_custom(args.unknown_test, use_ns=True)

    known_raw = load_sessions_custom(args.audio_dir, use_ns=False)
    cal_u_raw = load_sessions_custom(args.unknown_cal, use_ns=False)
    test_u_raw = load_sessions_custom(args.unknown_test, use_ns=False)

    ext_ubms = {}
    ext_windows = 0
    if args.ubm_dir and args.ubm_dir.exists():
        ext_frames = load_external_ubm_frames(args.ubm_dir, max_windows=350)
        ext_windows = len(ext_frames)
        for label, mode in [("GMM-UBM (FLEURS, CMN-c1)", "cmn_c1"), ("GMM-UBM (FLEURS, sem CMN)", "none")]:
            X_ubm = np.vstack([extract_gmm_features(f, mode, True) for f in ext_frames])
            ubm_gmm = GaussianMixture(8, covariance_type="diag", reg_covar=1e-4, max_iter=200, random_state=42).fit(X_ubm)
            ext_ubms[label] = (mode, ubm_gmm)

    # Split principal (seed 42)
    train_idx, cal_idx, test_idx = session_split(known_ns["labels"], known_ns["sessions"], seed=args.seed, calibration=True)
    main_models = run_all_models_on_split(known_ns, cal_u_ns, test_u_ns, train_idx, cal_idx, test_idx, ext_ubms=ext_ubms)
    ablation = run_ablation(known_ns, cal_u_ns, test_u_ns, known_raw, cal_u_raw, test_u_raw, train_idx, cal_idx, test_idx)

    # Multi-seed sobre todas as combinações de sessões dos conhecidos (média ± DP)
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    seen_splits = set()
    multi = {m: {"closed_acc": [], "far": [], "frr": [], "eer": [], "auc": []} for m in main_models}
    for sd in seeds:
        tr, ca, te = session_split(known_ns["labels"], known_ns["sessions"], seed=sd, calibration=True)
        key = (tuple(sorted(set(known_ns["sessions"][tr]))), tuple(sorted(set(known_ns["sessions"][ca]))), tuple(sorted(set(known_ns["sessions"][te]))))
        if key in seen_splits:
            continue
        seen_splits.add(key)
        res = run_all_models_on_split(known_ns, cal_u_ns, test_u_ns, tr, ca, te, ext_ubms=ext_ubms)
        for m, vals in res.items():
            for k in ("closed_acc", "far", "frr", "eer", "auc"):
                multi[m][k].append(vals[k])

    multi_summary = {}
    for m, d in multi.items():
        multi_summary[m] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in d.items()
        }

    plot_figures(main_models, args.figs_dir)

    report = {
        "dataset": {
            "known_speakers": sorted(set(known_ns["labels"])),
            "known_sessions": sorted(set(known_ns["sessions"])),
            "known_windows": int(len(known_ns["labels"])),
            "unknown_cal_speakers": sorted(set(cal_u_ns["labels"])),
            "unknown_cal_windows": int(len(cal_u_ns["labels"])),
            "unknown_test_speakers": sorted(set(test_u_ns["labels"])),
            "unknown_test_windows": int(len(test_u_ns["labels"])),
            "external_ubm_windows": ext_windows,
        },
        "main_split_seed42": {
            "train_sessions": sorted(set(known_ns["sessions"][train_idx])),
            "cal_sessions": sorted(set(known_ns["sessions"][cal_idx])),
            "test_sessions": sorted(set(known_ns["sessions"][test_idx])),
            "test_known_windows": int(len(test_idx)),
            "test_unknown_windows": int(len(test_u_ns["labels"])),
            "models": {
                m: {k: v for k, v in d.items() if k not in ("fpr", "tpr", "test_known_scores", "test_unknown_scores")}
                for m, d in main_models.items()
            },
        },
        "multi_split_summary": {
            "unique_session_splits": len(seen_splits),
            "models": multi_summary,
        },
        "ablation": ablation,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n=== BENCHMARK OPEN-SET (SEED 42) ===")
    print(f"{'Modelo':22s} | {'Acc Fech.':>9s} | {'FAR':>7s} | {'FRR':>7s} | {'EER':>7s} | {'tau*':>8s}")
    for m, d in main_models.items():
        print(f"{m:22s} | {d['closed_acc']*100:8.1f}% | {d['far']*100:6.1f}% | {d['frr']*100:6.1f}% | {d['eer']*100:6.1f}% | {d['tau']:+8.3f}")

    print(f"\n=== RESUMO MULTI-SPLIT ({len(seen_splits)} splits únicos de sessão) ===")
    for m, d in multi_summary.items():
        print(f"{m:26s} | Acc={d['closed_acc']['mean']*100:5.1f}±{d['closed_acc']['std']*100:4.1f}% | "
              f"AUC={d['auc']['mean']*100:5.1f}±{d['auc']['std']*100:4.1f}% | "
              f"FAR={d['far']['mean']*100:5.1f}±{d['far']['std']*100:4.1f}% | "
              f"FRR={d['frr']['mean']*100:5.1f}±{d['frr']['std']*100:4.1f}% | "
              f"EER={d['eer']['mean']*100:5.1f}±{d['eer']['std']*100:4.1f}%")

    print("\n=== ESTUDO DE ABLAÇÃO (SEED 42) ===")
    for row in ablation:
        print(f"{row['config']:36s} | Acc={row['closed_acc']*100:5.1f}% | FAR={row['far']*100:5.1f}% | FRR={row['frr']*100:5.1f}% | EER={row['eer']*100:5.1f}% | marg={row['mean_margin']:+.3f}")


if __name__ == "__main__":
    main()
