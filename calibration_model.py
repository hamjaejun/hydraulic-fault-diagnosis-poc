"""Serializable calibration model used by training scripts and Streamlit."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import RANDOM_SEED


class OvRProbabilityCalibrator:
    """One-vs-rest Platt calibration around a frozen multiclass estimator."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrators: list[LogisticRegression] = []
        self.classes_ = np.asarray(base_model.classes_)

    def fit(self, X, y):
        raw = np.clip(self.base_model.predict_proba(X), 1e-6, 1 - 1e-6)
        logits = np.log(raw / (1 - raw))
        self.calibrators = []
        for index, class_id in enumerate(self.classes_):
            calibrator = LogisticRegression(random_state=RANDOM_SEED)
            calibrator.fit(logits[:, [index]], (np.asarray(y) == class_id).astype(int))
            self.calibrators.append(calibrator)
        return self

    def predict_proba(self, X):
        raw = np.clip(self.base_model.predict_proba(X), 1e-6, 1 - 1e-6)
        logits = np.log(raw / (1 - raw))
        calibrated = np.column_stack(
            [cal.predict_proba(logits[:, [idx]])[:, 1] for idx, cal in enumerate(self.calibrators)]
        )
        return calibrated / calibrated.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

