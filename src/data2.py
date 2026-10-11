from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import io

import numpy as np
import pandas as pd

FEATURES = [
    "longitude", "latitude", "depth", "magnitude",
    "rate_1d", "rate_7d", "rate_30d"
]

RAW_DIR = Path("data/raw")
REAL_EARTHQUAKE_URL = (
    "https://raw.githubusercontent.com/bmullet/PEEVED/master/puuoo_earthquakes.csv"
)
REAL_ERUPTION_XLSX_URL = (
    "https://raw.githubusercontent.com/bmullet/PEEVED/master/kilauea_eruptionHistory.xlsx"
)
REAL_PUUOO_URL = (
    "https://raw.githubusercontent.com/bmullet/PEEVED/master/PuuOo.csv"
)


def make_synthetic(seed: int = 42, n_episodes: int = 20):
    """Small offline dataset used only for testing the pipeline."""
    rng = np.random.default_rng(seed)
    starts, ends, t = [], [], 5.0
    for _ in range(n_episodes):
        dur = rng.uniform(2, 5)
        starts.append(t)
        ends.append(t + dur)
        t += dur + rng.uniform(10, 25)

    rows = []
    for day in np.arange(0.5, t, 0.5):
        i = np.searchsorted(starts, day, side="right")
        erupting = i > 0 and day <= ends[i - 1]
        tte = 0 if erupting else (starts[i] - day if i < len(starts) else 999)
        activity = np.exp(-min(tte, 60) / 10) + float(erupting)
        n = rng.poisson(1 + 3 * activity)
        for _ in range(n):
            rows.append((
                pd.Timestamp("1983-01-01") + pd.Timedelta(days=day + rng.random() * 0.2),
                rng.normal(19.35, 0.04),
                rng.normal(-155.20, 0.10),
                rng.gamma(3, 2),
                1.5 + rng.exponential(0.5),
            ))

    cat = pd.DataFrame(
        rows, columns=["time", "latitude", "longitude", "depth", "magnitude"]
    ).sort_values("time").reset_index(drop=True)

    eru = pd.DataFrame({
        "start": pd.Timestamp("1983-01-01") + pd.to_timedelta(starts, unit="D"),
        "end": pd.Timestamp("1983-01-01") + pd.to_timedelta(ends, unit="D"),
    })
    return cat, eru


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url}")
        urlretrieve(url, path)
    return path


def _normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "date-time": "time",
        "datetime": "time",
        "date_time": "time",
        "latitude": "latitude",
        "longitude": "longitude",
        "depth": "depth",
        "magnitude": "magnitude",
    }
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    required = ["time", "latitude", "longitude", "depth", "magnitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Earthquake file is missing columns: {missing}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("time")
    return df[required].reset_index(drop=True)


def _normalize_eruptions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    start_candidates = [c for c in df.columns if "start" in c or c in {"date", "begin"}]
    end_candidates = [c for c in df.columns if "end" in c or "stop" in c]
    if not start_candidates or not end_candidates:
        raise ValueError(
            f"Could not identify start/end columns in eruption table: {list(df.columns)}"
        )
    out = pd.DataFrame({
        "start": pd.to_datetime(df[start_candidates[0]], errors="coerce"),
        "end": pd.to_datetime(df[end_candidates[0]], errors="coerce"),
    }).dropna()
    out = out[out["end"] >= out["start"]].sort_values("start")
    return out.reset_index(drop=True)

def _load_puuoo_fallback(path):
    df = pd.read_csv(path)

    # Convert eruption start date
    df["start"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Extract duration in hours
    hours = pd.to_numeric(
        df["Length"]
        .astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )

    # Remove clearly invalid duration values.
    # The PuuOo.csv file contains a special value:
    # 173566800 hrs, which is not a real eruption duration.
    hours = hours.where(
        (hours >= 0) & (hours <= 10000)
    )

    # Calculate eruption end time
    df["end"] = (
        df["start"]
        + pd.to_timedelta(hours, unit="h")
    )

    return (
        df[["start", "end"]]
        .dropna()
        .sort_values("start")
        .reset_index(drop=True)
    )

def load_real(data_dir: str | Path = "data/raw"):
    """Download/load the original authors' real files.

    The preferred eruption source is the authors' XLSX. If its schema cannot
    be recognized, PuuOo.csv is used as a transparent fallback.
    """
    data_dir = Path(data_dir)
    eq_path = _download(REAL_EARTHQUAKE_URL, data_dir / "puuoo_earthquakes.csv")
    eru_xlsx = _download(
        REAL_ERUPTION_XLSX_URL, data_dir / "kilauea_eruptionHistory.xlsx"
    )

    catalog = _normalize_catalog(pd.read_csv(eq_path))

    try:
        book = pd.ExcelFile(eru_xlsx)
        candidates = []
        for sheet in book.sheet_names:
            raw = pd.read_excel(eru_xlsx, sheet_name=sheet)
            try:
                candidates.append(_normalize_eruptions(raw))
            except ValueError:
                pass
        if not candidates:
            raise ValueError("No recognizable start/end eruption sheet found.")
        eruptions = max(candidates, key=len)
    except Exception as exc:
        print(f"Could not parse eruption workbook ({exc}); using PuuOo.csv fallback.")
        puuoo_path = _download(REAL_PUUOO_URL, data_dir / "PuuOo.csv")
        eruptions = _load_puuoo_fallback(puuoo_path)

    return catalog, eruptions


def load_sample(data_dir: str | Path = "data"):
    data_dir = Path(data_dir)
    catalog = _normalize_catalog(pd.read_csv(data_dir / "sample_catalog.csv"))
    eruptions = _normalize_eruptions(pd.read_csv(data_dir / "sample_eruptions.csv"))
    return catalog, eruptions


def build_dataset(catalog: pd.DataFrame, eruptions: pd.DataFrame) -> pd.DataFrame:
    """Create paper features and both targets without using future earthquakes."""
    cat = catalog.copy().sort_values("time").reset_index(drop=True)
    eru = eruptions.copy().sort_values("start").reset_index(drop=True)

    t0 = cat["time"].min()
    t = ((cat["time"] - t0) / pd.Timedelta(days=1)).to_numpy()
    starts = ((eru["start"] - t0) / pd.Timedelta(days=1)).to_numpy()
    ends = ((eru["end"] - t0) / pd.Timedelta(days=1)).to_numpy()

    # For each event, count strictly prior events inside each time window.
    # searchsorted uses event time and therefore never looks at future events.
    for w in (1, 7, 30):
        cat[f"rate_{w}d"] = np.searchsorted(t, t, side="left") - np.searchsorted(
            t, t - w, side="left"
        )

    i = np.searchsorted(starts, t, side="right")
    previous = np.clip(i - 1, 0, max(len(ends) - 1, 0))
    erupting = (i > 0) & (t <= ends[previous])

    next_start = np.where(
        i < len(starts), starts[np.clip(i, 0, max(len(starts) - 1, 0))], np.inf
    )
    cat["erupting"] = erupting.astype(int)
    cat["tte"] = np.where(erupting, 0.0, next_start - t)

    # Events after the last known eruption cannot have a finite next-eruption target.
    cat = cat[np.isfinite(cat["tte"])].reset_index(drop=True)
    return cat


def chronological_split(df, train_frac=0.70, dev_frac=0.20):
    n = len(df)
    a = int(n * train_frac)
    b = int(n * (train_frac + dev_frac))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def random_stratified_split(df, seed=0):
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["erupting"]
    )
    dev, test = train_test_split(
        temp, test_size=1 / 3, random_state=seed, stratify=temp["erupting"]
    )
    return train.sort_values("time"), dev.sort_values("time"), test.sort_values("time")
