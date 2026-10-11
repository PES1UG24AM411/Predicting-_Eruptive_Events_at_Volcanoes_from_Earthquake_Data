# Volcano Eruption ML

A reproducible implementation of the Stanford CS229 project **“Predicting Eruptive Events at Volcanoes from Earthquake Data”**.

The project has two tasks:

1. **Contemporaneous eruption classification** — was Kīlauea erupting when an earthquake occurred?
2. **Time-to-eruption regression** — how many days until the next eruption begins?

The default pipeline uses the **real earthquake catalog and eruption data published with the original authors' PEEVED repository**. Synthetic data is kept only as an offline smoke-test/demo.

## Project structure

```text
volcano-eruption-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   ├── DATASET.md
│   ├── sample_catalog.csv
│   └── sample_eruptions.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── plots.py
├── notebooks/
│   └── Volcano_Eruption_Prediction.ipynb
└── results/
    └── .gitkeep
```

## Setup

Python 3.11+ is recommended.

```bash
cd volcano-eruption-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the real-data pipeline

The real-data mode downloads the original project data into `data/raw/` if it is not already present.

```bash
python main.py --data-source real
```

Results are written to `results/`.

## Run an offline smoke test

```bash
python main.py --data-source synthetic
```

## Run the notebook

```bash
jupyter notebook notebooks/Volcano_Eruption_Prediction.ipynb
```

Run all cells from top to bottom.

## Important evaluation note

The paper uses a random 70/20/10 split. This project reproduces that setup and also supports a chronological split because earthquakes close in time can be strongly correlated. The chronological result should be treated as the more realistic stress test for future-event generalization.

## Expected outputs

### Classification
- Logistic Regression
- K-Means classifier
- Random Forest classifier
- Neural-network classifier
- AUROC
- Cohen's kappa
- confusion matrix

### Regression
- K-Means regressor
- Random Forest regressor
- Neural-network regressor
- RMSE
- R²
- observed-vs-predicted plot

The project does **not** claim that an eruption can be predicted with certainty. The regression task is intentionally difficult and should be interpreted as an experiment on how much signal is present in the earthquake catalog alone.
