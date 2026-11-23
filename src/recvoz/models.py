"""Estimadores exploratórios usados na comparação do notebook."""
import numpy as np
from sklearn.mixture import GaussianMixture

def frames_of(X, cls, y):
    return np.concatenate([seg for seg, lab in zip(X, y) if lab == cls], axis=1).T


class CentroidModel:
    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        self.W_ = np.array([np.asarray(X)[np.asarray(y) == c].mean(0) for c in self.classes_])
        return self
    def predict(self, X):
        X = np.asarray(X); X = X / np.linalg.norm(X, axis=1)[:, None]
        W = self.W_ / np.linalg.norm(self.W_, axis=1)[:, None]
        return np.array([self.classes_[i] for i in np.argmax(X @ W.T, axis=1)])


class GMMModel:
    def __init__(self, n_components=4): self.n_components = n_components
    def fit(self, X, y):
        self.models_ = {}
        for c in np.unique(np.asarray(y)):
            self.models_[c] = GaussianMixture(self.n_components, covariance_type='diag',
                                              random_state=0).fit(frames_of(X, c, np.asarray(y)))
        return self
    def predict(self, X):
        preds = []
        for seg in X:
            scores = {c: m.score_samples(seg.T).mean() for c, m in self.models_.items()}
            preds.append(max(scores.items(), key=lambda kv: kv[1])[0])
        return np.array(preds)

