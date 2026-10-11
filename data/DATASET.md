# Dataset

## Primary real data

This project uses the real files published in the original PEEVED repository:

- `puuoo_earthquakes.csv` — earthquake catalog with date/time, latitude, longitude, depth, magnitude, magnitude type, earthquake type, distance and data owner.
- `kilauea_eruptionHistory.xlsx` — eruption history used by the original project.
- `PuuOo.csv` — Puʻu ʻŌʻō eruptive episodes. It is retained as a fallback because the repository also publishes it in CSV form.

Source repository:

https://github.com/bmullet/PEEVED

Raw earthquake file:

https://raw.githubusercontent.com/bmullet/PEEVED/master/puuoo_earthquakes.csv

Raw eruption-history workbook:

https://raw.githubusercontent.com/bmullet/PEEVED/master/kilauea_eruptionHistory.xlsx

## What the model needs

Earthquake table:

```text
time, latitude, longitude, depth, magnitude
```

Eruption table:

```text
start, end
```

The loader normalizes column names and accepts common alternatives such as `Date-time`, `Latitude`, `Longitude`, `Depth`, and `Magnitude`.

## Engineered features

For every earthquake:

```text
longitude
latitude
depth
magnitude
rate_1d
rate_7d
rate_30d
```

`rate_Nd` is the number of earthquakes in the previous N days.

Targets:

```text
erupting = 1 if the earthquake occurs during an eruption, else 0
tte      = days until the next eruption begins; 0 during an eruption
```

## Synthetic files

`sample_catalog.csv` and `sample_eruptions.csv` are intentionally small and are only for an offline smoke test. They are **not** the results dataset and must not be used to claim reproduction of the paper.
