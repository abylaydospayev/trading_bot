#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

def _standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return mean, std

def _standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std

class NumpyMLP:
    def __init__(self, input_dim: int, hidden=(32,16), lr=1e-3, max_epochs=50, batch_size=256, seed=42):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        # Xavier init
        self.W1 = rng.normal(0, np.sqrt(2/(input_dim+h1)), size=(input_dim, h1))
        self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, np.sqrt(2/(h1+h2)), size=(h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = rng.normal(0, np.sqrt(2/(h2+1)), size=(h2, 1))
        self.b3 = np.zeros(1)
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # Adam params
        self.m = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2), np.zeros_like(self.W3), np.zeros_like(self.b3)]
        self.v = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2), np.zeros_like(self.W3), np.zeros_like(self.b3)]
        self.t = 0

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3
        p = self._sigmoid(z3)
        cache = (X, z1, a1, z2, a2, z3, p)
        return p, cache

    def _backward(self, cache, y):
        X, z1, a1, z2, a2, z3, p = cache
        # Binary cross-entropy gradient
        m = y.shape[0]
        dz3 = (p - y.reshape(-1,1)) / m
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def _adam_step(self, grads):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        for i, (param, g) in enumerate(zip(params, grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g * g)
            mhat = self.m[i] / (1 - beta1 ** self.t)
            vhat = self.v[i] / (1 - beta2 ** self.t)
            params[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, X, y, pos_weight: float = 1.0):
        n = X.shape[0]
        for epoch in range(self.max_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start+self.batch_size]
                xb, yb = X[batch_idx], y[batch_idx]
                p, cache = self._forward(xb)
                # weighted gradient: scale positive errors
                Xc, z1, a1, z2, a2, z3, pb = cache
                m = yb.shape[0]
                w = np.where(yb > 0.5, pos_weight, 1.0).reshape(-1,1)
                dz3 = w * (pb - yb.reshape(-1,1)) / max(m,1)
                dW3 = a2.T @ dz3
                db3 = dz3.sum(axis=0)
                da2 = dz3 @ self.W3.T
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)
                grads = [dW1, db1, dW2, db2, dW3, db3]
                self._adam_step(grads)

    def predict_proba(self, X):
        p, _ = self._forward(X)
        return np.clip(p, 1e-6, 1-1e-6)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.dropna()


def main():
    ap = argparse.ArgumentParser(description="Train a simple MLP on trading features")
    ap.add_argument("dataset", help="CSV produced by prepare_dataset.py")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "ml" / "models" / "mlp_model.pkl"))
    args = ap.parse_args()

    df = load_dataset(args.dataset)
    feature_names = [c for c in df.columns if c != "y"]
    X = df[feature_names].values
    y = df["y"].values

    # Time-based split: simple last N% as validation
    split_idx = int(len(df) * (1 - args.val_size))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    mean, std = _standardize_fit(X_train)
    X_train_s = _standardize_transform(X_train, mean, std)
    X_val_s = _standardize_transform(X_val, mean, std)

    clf = NumpyMLP(input_dim=X.shape[1], hidden=(32,16), lr=1e-3, max_epochs=80, batch_size=256, seed=args.random_state)
    # Compute positive class weight to address imbalance
    pos_rate = float((y_train==1).mean()) if len(y_train)>0 else 0.0
    pos_weight = 1.0
    if pos_rate > 0 and pos_rate < 0.5:
        pos_weight = max(1.0, (1.0 - pos_rate) / max(pos_rate, 1e-6))
    clf.fit(X_train_s, y_train, pos_weight=pos_weight)

    val_proba = clf.predict_proba(X_val_s).ravel()
    # AUC (manual) if both classes present, implemented without sklearn
    def _roc_auc_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = y_true.astype(int)
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(y_score)
        sorted_scores = y_score[order]
        ranks = np.zeros_like(y_score, dtype=float)
        n = len(y_score)
        i = 0
        rank = 1
        while i < n:
            j = i
            while j < n and sorted_scores[j] == sorted_scores[i]:
                j += 1
            avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
            ranks[order[i:j]] = avg_rank
            rank += (j - i)
            i = j
        sum_ranks_pos = ranks[y_true == 1].sum()
        auc_val = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc_val)

    auc = _roc_auc_manual(y_val, val_proba)

    thr = 0.6
    y_pred = (val_proba >= thr).astype(int)
    tp = int(((y_pred==1)&(y_val==1)).sum())
    fp = int(((y_pred==1)&(y_val==0)).sum())
    tn = int(((y_pred==0)&(y_val==0)).sum())
    fn = int(((y_pred==0)&(y_val==1)).sum())
    prec = tp / max(tp+fp, 1)
    rec = tp / max(tp+fn, 1)
    print(f"Validation AUC: {auc if not np.isnan(auc) else float('nan'):.3f}")
    print(f"Confusion @thr={thr}: TP {tp} FP {fp} TN {tn} FN {fn} | Precision {prec:.3f} Recall {rec:.3f}")

    # Save model bundle
    bundle = {
        "model_type": "numpy_mlp",
        "model": {
            "W1": clf.W1, "b1": clf.b1,
            "W2": clf.W2, "b2": clf.b2,
            "W3": clf.W3, "b3": clf.b3,
            "hidden": (clf.W1.shape[1], clf.W2.shape[1]),
        },
        "scaler": {"mean": mean, "std": std},
        "features": feature_names,
        "metadata": {"val_size": args.val_size, "auc": float(auc)},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
