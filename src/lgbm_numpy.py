"""Run a LightGBM text model (`Booster.save_model`) with numpy only — no lightgbm, no scipy.

    bst = NumpyBooster("models/beta_v0/phase_lgbm_v2.txt")
    y = bst.predict(df[features])          # same output as lightgbm.Booster.predict

Supports gbdt models with numerical splits (decision types with missing = None / Zero / NaN) and the
objectives used here: lambdarank / regression (raw score), binary (sigmoid), multiclass (softmax).
Categorical splits and linear trees are not supported and raise at load time.

A **seed-bagged** model (stage 07) is several text models whose predictions are averaged:

    bag = NumpyBoosterBag(["m_s0.txt", "m_s1.txt", "m_s2.txt"])
    y = bag.predict(df[features])          # == mean of the individual predictions
"""
from __future__ import annotations

import numpy as np

_K_ZERO = 1e-35


class _Tree:
    __slots__ = ("feat", "thr", "default_left", "miss", "left", "right", "leaf")

    def __init__(self, block: dict):
        n_leaves = int(block["num_leaves"])
        if int(block.get("num_cat", 0)) or int(block.get("is_linear", 0)):
            raise NotImplementedError("categorical splits / linear trees are not supported")
        self.leaf = np.array(block["leaf_value"].split(), dtype=np.float64)
        if n_leaves > 1:
            dt = np.array(block["decision_type"].split(), dtype=np.int64)
            if (dt & 1).any():
                raise NotImplementedError("categorical splits are not supported")
            self.default_left = (dt & 2) > 0
            self.miss = (dt >> 2) & 3                     # 0 = none, 1 = zero, 2 = nan
            self.feat = np.array(block["split_feature"].split(), dtype=np.int64)
            self.thr = np.array(block["threshold"].split(), dtype=np.float64)
            self.left = np.array(block["left_child"].split(), dtype=np.int64)
            self.right = np.array(block["right_child"].split(), dtype=np.int64)
        else:
            self.feat = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.feat is None:
            return np.full(n, self.leaf[0])
        node = np.zeros(n, dtype=np.int64)               # >= 0: internal node, < 0: leaf ~node
        active = np.arange(n)
        while active.size:
            nd = node[active]
            v = X[active, self.feat[nd]]
            miss, isnan = self.miss[nd], np.isnan(v)
            missing = ((miss == 2) & isnan) | ((miss == 1) & (isnan | (np.abs(v) <= _K_ZERO)))
            v = np.where(isnan & (miss != 2), 0.0, v)
            go_left = np.where(missing, self.default_left[nd], v <= self.thr[nd])
            nxt = np.where(go_left, self.left[nd], self.right[nd])
            node[active] = nxt
            active = active[nxt >= 0]
        return self.leaf[~node]


class NumpyBooster:
    def __init__(self, model_file):
        header, trees, block, in_tree = {}, [], None, False
        with open(model_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("Tree="):
                    block, in_tree = {}, True
                    trees.append(block)
                elif line == "end of trees":
                    break
                elif "=" in line:
                    k, v = line.split("=", 1)
                    (block if in_tree else header)[k] = v
        self.feature_names = header["feature_names"].split()
        self.n_features = int(header["max_feature_idx"]) + 1
        self.k = int(header.get("num_tree_per_iteration", 1))
        self.objective = header.get("objective", "").split()
        self.average_output = "average_output" in header
        self.trees = [_Tree(b) for b in trees]
        self.sigmoid = 1.0
        for tok in self.objective[1:]:
            if tok.startswith("sigmoid:"):
                self.sigmoid = float(tok.split(":")[1])

    def predict(self, X, raw_score: bool = False) -> np.ndarray:
        if hasattr(X, "to_numpy"):                       # pandas, incl. nullable dtypes with pd.NA
            X = X.to_numpy(dtype=np.float64, na_value=np.nan)
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} feature columns, got {X.shape}")
        raw = np.zeros((X.shape[0], self.k))
        for i, t in enumerate(self.trees):
            raw[:, i % self.k] += t.predict(X)
        if self.average_output:
            raw /= max(1, len(self.trees) // self.k)
        name = self.objective[0] if self.objective else ""
        if not raw_score:
            if name == "binary":
                raw = 1.0 / (1.0 + np.exp(-self.sigmoid * raw))
            elif name in ("multiclass", "softmax"):
                e = np.exp(raw - raw.max(1, keepdims=True))
                raw = e / e.sum(1, keepdims=True)
            elif name in ("multiclassova", "cross_entropy", "xentropy", "poisson", "gamma", "tweedie"):
                raise NotImplementedError(f"objective {name} is not supported")
        return raw[:, 0] if self.k == 1 else raw


class NumpyBoosterBag:
    """Average of K LightGBM text models trained with different seeds (stage 07 bagging).

    The average is taken in the models' own output space -- probabilities for binary /
    multiclass objectives, raw scores for rankers -- which is exactly what the training
    code does when it averages fold predictions."""

    def __init__(self, model_files):
        self.boosters = [f if isinstance(f, NumpyBooster) else NumpyBooster(f)
                         for f in model_files]
        if not self.boosters:
            raise ValueError("NumpyBoosterBag needs at least one model file")
        b0 = self.boosters[0]
        self.feature_names = b0.feature_names
        self.n_features = b0.n_features
        self.k = b0.k
        self.objective = b0.objective
        for b in self.boosters[1:]:
            if b.n_features != self.n_features or b.feature_names != self.feature_names:
                raise ValueError("bagged models must share the same feature list")

    def predict(self, X, raw_score: bool = False) -> np.ndarray:
        out = None
        for b in self.boosters:
            p = b.predict(X, raw_score=raw_score)
            out = p if out is None else out + p
        return out / len(self.boosters)


def predict_average(model_files, X, raw_score: bool = False) -> np.ndarray:
    """One-shot helper: mean prediction of the given LightGBM text models."""
    return NumpyBoosterBag(model_files).predict(X, raw_score=raw_score)
