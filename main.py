from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data import (
    FEATURES, build_dataset, chronological_split, load_real, load_sample, make_synthetic,
    random_stratified_split
)
from src.models import fit_and_score_classifiers, fit_and_score_regressors
from src.plots import (
    save_classification_plot, save_feature_importance, save_rate_plot,
    save_regression_plot,
)


def run(data_source="real", split_mode="both"):
    if data_source == "real":
        catalog, eruptions = load_real()
        source_name = "REAL — original PEEVED earthquake/eruption files"
    else:
        catalog, eruptions = make_synthetic(seed=42, n_episodes=40)
        source_name = "SYNTHETIC — offline smoke test"

    df = build_dataset(catalog, eruptions)
    print(f"\nData source: {source_name}")
    print(f"Raw earthquakes: {len(catalog)}")
    print(f"Eruption episodes: {len(eruptions)}")
    print(f"Usable labeled earthquakes: {len(df)}")
    print(f"Erupting share: {df.erupting.mean():.2%}")

    Path("results").mkdir(exist_ok=True)
    save_rate_plot(df, "results/earthquake_rate.png")

    modes = ["random", "chronological"] if split_mode == "both" else [split_mode]

    all_cls, all_reg = [], []

    for mode in modes:
        if mode == "random":
            train, dev, test = random_stratified_split(df)
        else:
            train, dev, test = chronological_split(df)

        print(f"\n=== {mode.upper()} SPLIT ===")
        print(f"train/dev/test = {len(train)}/{len(dev)}/{len(test)}")

        cls_results, cls_models, cls_scaler, rf_params = fit_and_score_classifiers(
            train, dev, test, FEATURES, split_name=mode
        )
        reg_results, reg_models, reg_scaler, target_scaler, reg_params = fit_and_score_regressors(
            train, dev, test, FEATURES
        )

        all_cls.append(cls_results)
        all_reg.append(reg_results)

        print("\nClassification test:")
        print(cls_results[cls_results.dataset == "test"].round(3).to_string(index=False))

        print("\nRegression test (target is standardized for training/evaluation):")
        print(reg_results[reg_results.dataset == "test"].round(3).to_string(index=False))

        # Only use the RF models for the headline plots.
        rf_cls = cls_models["Random Forest"]
        rf_reg = reg_models["Random Forest"]

        save_feature_importance(
            rf_cls, FEATURES, f"results/{mode}_classification_feature_importance.png",
            f"Random Forest feature importance — classification ({mode})"
        )
        save_feature_importance(
            rf_reg, FEATURES, f"results/{mode}_regression_feature_importance.png",
            f"Random Forest feature importance — regression ({mode})"
        )
        save_regression_plot(
            rf_reg, reg_scaler, target_scaler, test, FEATURES,
            f"results/{mode}_regression_predictions.png"
        )

    pd.concat(all_cls, ignore_index=True).to_csv("results/classification_results.csv", index=False)
    pd.concat(all_reg, ignore_index=True).to_csv("results/regression_results.csv", index=False)

    print("\nSaved results under results/")
    print("Do not quote metrics from the synthetic smoke test as paper/research results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--split", choices=["random", "chronological", "both"], default="both")
    args = parser.parse_args()
    run(args.data_source, args.split)
