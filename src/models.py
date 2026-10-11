from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


SEED = 0


class KMeansClassifier:
    def __init__(self, k=8):
        self.k = k

    def fit(self, X, y):
        self.models = {}
        for cls in [0, 1]:
            self.models[cls] = KMeans(
                n_clusters=self.k,
                n_init=10,
                random_state=SEED
            ).fit(X[y == cls])
        return self

    def predict(self, X):
        d0 = self.models[0].transform(X).min(axis=1)
        d1 = self.models[1].transform(X).min(axis=1)
        return (d1 < d0).astype(int)

    def predict_proba(self, X):
        d0 = self.models[0].transform(X).min(axis=1)
        d1 = self.models[1].transform(X).min(axis=1)

        score = d0 - d1
        p1 = 1 / (1 + np.exp(-score / (np.std(score) + 1e-8)))

        return np.column_stack([1 - p1, p1])


class KMeansRegressor:
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.km = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=SEED
        ).fit(np.column_stack([X, y]))

        self.centroids = self.km.cluster_centers_
        return self

    def predict(self, X):
        d = (
            (X[:, None, :] - self.centroids[None, :, :-1]) ** 2
        ).sum(axis=2)

        return self.centroids[d.argmin(axis=1), -1]


# ---------------------------------------------------------
# Random Forest tuning
# ---------------------------------------------------------

def tune_rf_classifier(Xtr, ytr, Xdv, ydv):
    best = (-np.inf, 300, 10)

    for n in (100, 300):
        for depth in (5, 10, 15, 20, None):

            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1
            ).fit(Xtr, ytr)

            score = roc_auc_score(
                ydv,
                model.predict_proba(Xdv)[:, 1]
            )

            if score > best[0]:
                best = (score, n, depth)

    return best[1], best[2]


def tune_rf_regressor(Xtr, ytr, Xdv, ydv):
    best = (-np.inf, 300, 10)

    for n in (100, 300):
        for depth in (5, 10, 15, 20, None):

            model = RandomForestRegressor(
                n_estimators=n,
                max_depth=depth,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1
            ).fit(Xtr, ytr)

            score = r2_score(ydv, model.predict(Xdv))

            if score > best[0]:
                best = (score, n, depth)

    return best[1], best[2]


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

def make_classifiers(best_n, best_depth):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "K-Means": KMeansClassifier(k=8),

        "Random Forest": RandomForestClassifier(
            n_estimators=best_n,
            max_depth=best_depth,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ),

        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 256, 256, 256),
            alpha=1e-3,
            batch_size=50,
            early_stopping=True,
            max_iter=200,
            random_state=SEED
        )
    }


def make_regressors(best_n, best_depth):
    return {
        "K-Means": KMeansRegressor(k=10),

        "Random Forest": RandomForestRegressor(
            n_estimators=best_n,
            max_depth=best_depth,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1
        ),

        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(256, 256, 256, 256),
            alpha=1e-3,
            batch_size=50,
            early_stopping=True,
            max_iter=200,
            random_state=SEED
        )
    }


# ---------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------

def tune_threshold(model, X, y):
    """Pick the probability cut-off that maximises Cohen's kappa on the dev set."""
    score = model.predict_proba(X)[:, 1]
    best_t, best_k = 0.5, -np.inf
    for t in np.unique(np.quantile(score, np.linspace(0.05, 0.95, 91))):
        k = cohen_kappa_score(y, (score >= t).astype(int))
        if k > best_k:
            best_t, best_k = float(t), k
    return best_t


def evaluate_classifier(model, X, y, threshold=0.5):

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
        pred = (score >= threshold).astype(int)
    else:
        pred = model.predict(X)
        score = pred

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "kappa": cohen_kappa_score(y, pred),
        "auroc": roc_auc_score(y, score),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ---------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------

def evaluate_regressor(model, X, y, target_scaler):

    pred = model.predict(X)

    rmse_scaled = np.sqrt(
        mean_squared_error(y, pred)
    )

    # Convert back to original days
    y_original = target_scaler.inverse_transform(
        y.reshape(-1, 1)
    ).ravel()

    pred_original = target_scaler.inverse_transform(
        pred.reshape(-1, 1)
    ).ravel()

    rmse_days = np.sqrt(
        mean_squared_error(
            y_original,
            pred_original
        )
    )

    return {
        "rmse_scaled": rmse_scaled,
        "rmse_days": rmse_days,
        "r2": r2_score(y, pred)
    }


# ---------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------

def fit_and_score_classifiers(
    train,
    dev,
    test,
    features,
    split_name="test"
):

    scaler = StandardScaler().fit(train[features])

    Xtr = scaler.transform(train[features])
    Xdv = scaler.transform(dev[features])
    Xte = scaler.transform(test[features])

    ytr = train.erupting.to_numpy()
    ydv = dev.erupting.to_numpy()
    yte = test.erupting.to_numpy()

    best_n, best_depth = tune_rf_classifier(
        Xtr, ytr, Xdv, ydv
    )

    models = make_classifiers(
        best_n,
        best_depth
    )

    rows = []
    fitted = {}

    for name, model in models.items():

        model.fit(Xtr, ytr)
        fitted[name] = model

        # Threshold is chosen on the dev split only (never on test).
        # K-Means keeps its natural nearest-centroid rule.
        threshold = 0.5 if name == "K-Means" else tune_threshold(model, Xdv, ydv)

        for dataset, X, y in [
            ("train", Xtr, ytr),
            ("dev", Xdv, ydv),
            ("test", Xte, yte)
        ]:

            r = evaluate_classifier(model, X, y, threshold)

            rows.append({
                "split": split_name,
                "model": name,
                "dataset": dataset,
                "threshold": threshold,

                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "kappa": r["kappa"],
                "auroc": r["auroc"],

                "tn": r["tn"],
                "fp": r["fp"],
                "fn": r["fn"],
                "tp": r["tp"]
            })

    return (
        pd.DataFrame(rows),
        fitted,
        scaler,
        (best_n, best_depth)
    )


# ---------------------------------------------------------
# Regression pipeline
# ---------------------------------------------------------

def fit_and_score_regressors(
    train,
    dev,
    test,
    features
):

    scaler = StandardScaler().fit(train[features])

    Xtr = scaler.transform(train[features])
    Xdv = scaler.transform(dev[features])
    Xte = scaler.transform(test[features])

    target_scaler = StandardScaler().fit(
        train[["tte"]]
    )

    ytr = target_scaler.transform(
        train[["tte"]]
    ).ravel()

    ydv = target_scaler.transform(
        dev[["tte"]]
    ).ravel()

    yte = target_scaler.transform(
        test[["tte"]]
    ).ravel()

    best_n, best_depth = tune_rf_regressor(
        Xtr, ytr, Xdv, ydv
    )

    models = make_regressors(
        best_n,
        best_depth
    )

    rows = []
    fitted = {}

    for name, model in models.items():

        model.fit(Xtr, ytr)
        fitted[name] = model

        for dataset, X, y in [
            ("train", Xtr, ytr),
            ("dev", Xdv, ydv),
            ("test", Xte, yte)
        ]:

            r = evaluate_regressor(
                model,
                X,
                y,
                target_scaler
            )

            rows.append({
                "model": name,
                "dataset": dataset,
                "rmse_scaled": r["rmse_scaled"],
                "rmse_days": r["rmse_days"],
                "r2": r["r2"]
            })

    return (
        pd.DataFrame(rows),
        fitted,
        scaler,
        target_scaler,
        (best_n, best_depth)
    )