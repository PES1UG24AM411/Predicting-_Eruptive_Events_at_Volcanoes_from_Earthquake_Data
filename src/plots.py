from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_classification_plot(results, path):
    test = results[results["dataset"] == "test"]
    names = test["model"].tolist()
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, test["kappa"], width, label="Cohen's kappa")
    ax.bar(x + width / 2, test["auroc"], width, label="AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Classification — real-data test set")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_importance(model, features, path, title):
    order = np.argsort(model.feature_importances_)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(order)), model.feature_importances_[order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([features[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_regression_plot(model, scaler, target_scaler, test, features, path):
    X = scaler.transform(test[features])
    pred_scaled = model.predict(X)
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual = test["tte"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, pred, s=10)
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xlabel("Observed time to eruption (days)")
    ax.set_ylabel("Predicted time to eruption (days)")
    ax.set_title("Random Forest regression — test set")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rate_plot(df, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["time"], df["rate_7d"], linewidth=1)
    eruption_times = df.loc[df["erupting"].eq(1), "time"]
    if len(eruption_times):
        ax.scatter(eruption_times, df.loc[df["erupting"].eq(1), "rate_7d"], s=5, label="eruption")
    ax.set_xlabel("Time")
    ax.set_ylabel("Earthquakes in previous 7 days")
    ax.set_title("Real earthquake activity over time")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
