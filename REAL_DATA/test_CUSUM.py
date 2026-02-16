import os
import re
import glob
from datetime import date, datetime
from typing import List, Optional, Tuple, Dict

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ================================
# PATHS
# ================================
ARL0_DIR = "/Users/lunarolland/Desktop/DATASETS/CUSUM_ARL0_VVSTACKS"
ARL1_DIR = "/Users/lunarolland/Desktop/DATASETS/CUSUM_ARL1_VVSTACKS"

N_PLOT_ARL0 = 6
N_PLOT_ARL1 = 6

# ================================
# BASELINE
# ================================
USE_PRE_EVENT_BASELINE = True
EVENT_DATE = date(2024, 10, 29)
BASELINE_M = 6

# ================================
# SIGNALS
# ================================
DARK_LO, DARK_HI = -25.0, -17.0

CLAMP_VV = True
CLAMP_LO, CLAMP_HI = -25.0, 0.0

BASELINE_Q = 99.99
EPS = 1e-3

# Drop bands that are basically empty
MIN_VALID_FRAC_PER_BAND = 0.10  # 10%

# ================================
# PLOTTING
# ================================
SHOW_PLOTS = True
PLOT_IN_PERCENT = True  # so 3.0 == 3%

ROBUST_YLIM = True
YLIM_PCT_LO = 1.0
YLIM_PCT_HI = 99.0
PAD_FRAC = 0.05

LINE_WIDTH = 1.8
SCATTER_SIZE = 18
DEBUG_PRINT = True


# ================================
# BAND DATE PARSING
# ================================
_date_pat = re.compile(r"(20\d{2})(\d{2})(\d{2})$")  # YYYYMMDD at end


def parse_band_date(label: str) -> Optional[date]:
    if not label:
        return None
    s = label.strip().upper()
    m = _date_pat.search(s)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def band_label(src: rasterio.DatasetReader, b: int) -> str:
    if src.descriptions and len(src.descriptions) >= b and src.descriptions[b - 1]:
        return src.descriptions[b - 1].strip()
    tags = src.tags(b)
    for k in ("band_name", "name", "DESCRIPTION", "description"):
        if k in tags and tags[k].strip():
            return tags[k].strip()
    return f"band{b}"


def to_datetimes(ds: List[date]) -> List[datetime]:
    return [datetime(d.year, d.month, d.day) for d in ds]


# ================================
# FIX: choose BEST band per date
# ================================
def read_vv_stack_bestband(path: str) -> Tuple[np.ndarray, List[date]]:
    """
    Reads all bands with parseable dates.
    Filters out near-empty bands (valid_frac < MIN_VALID_FRAC_PER_BAND).
    Groups by date; for each date keeps ONLY the band with the highest valid_frac.
    Returns (T,H,W) stack with UNIQUE dates.
    """
    with rasterio.open(path) as src:
        nodata = src.nodata

        # by_date[d] = (valid_frac, array)
        by_date: Dict[date, Tuple[float, np.ndarray]] = {}

        for b in range(1, src.count + 1):
            lab = band_label(src, b)
            d = parse_band_date(lab)
            if d is None:
                continue

            arr = src.read(b).astype(np.float32)
            if nodata is not None:
                arr[arr == nodata] = np.nan

            finite = np.isfinite(arr)
            valid_frac = float(finite.mean())

            # drop almost-empty bands
            if valid_frac < MIN_VALID_FRAC_PER_BAND:
                continue

            # keep best for that date
            if (d not in by_date) or (valid_frac > by_date[d][0]):
                by_date[d] = (valid_frac, arr)

    if not by_date:
        raise RuntimeError(f"No usable bands after filtering in: {path}")

    dates = sorted(by_date.keys())
    stack = np.stack([by_date[d][1] for d in dates], axis=0).astype(np.float32)
    return stack, dates


def clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def choose_baseline_idx(dates: List[date]) -> np.ndarray:
    if USE_PRE_EVENT_BASELINE:
        idx = np.array([i for i, d in enumerate(dates) if d < EVENT_DATE], dtype=int)
        if idx.size > 0:
            return idx
    return np.arange(min(BASELINE_M, len(dates)), dtype=int)


def baseline_mu_iqr(stack: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base = stack[idx]
    mu = np.nanmedian(base, axis=0)
    q25 = np.nanpercentile(base, 25, axis=0)
    q75 = np.nanpercentile(base, 75, axis=0)
    iqr = (q75 - q25).astype(np.float32)
    return mu.astype(np.float32), iqr.astype(np.float32)


def score_stack(stack: np.ndarray, mu: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return - (stack - mu[None, :, :]) / (iqr[None, :, :] + EPS)


def threshold_from_baseline(scores: np.ndarray, baseline_idx: np.ndarray, q: float) -> float:
    base = scores[baseline_idx]
    finite = base[np.isfinite(base)]
    if finite.size == 0:
        raise RuntimeError("No finite baseline scores (nodata everywhere?)")
    return float(np.percentile(finite, q))


def flooded_fraction_series(scores: np.ndarray, thr: float) -> np.ndarray:
    T = scores.shape[0]
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        s = scores[t]
        finite = np.isfinite(s)
        out[t] = np.nan if not finite.any() else float((s[finite] >= thr).mean())
    return out


def dark_fraction_series(stack: np.ndarray) -> np.ndarray:
    T = stack.shape[0]
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        v = stack[t]
        finite = np.isfinite(v)
        out[t] = np.nan if not finite.any() else float(((v >= DARK_LO) & (v <= DARK_HI) & finite).mean())
    return out


def compute_global_ylims(series_list: List[np.ndarray]) -> Optional[Tuple[float, float]]:
    vals = np.concatenate([s[np.isfinite(s)] for s in series_list if s is not None], axis=0) \
        if series_list else np.array([], dtype=np.float32)
    if vals.size == 0:
        return None
    lo = np.percentile(vals, YLIM_PCT_LO)
    hi = np.percentile(vals, YLIM_PCT_HI)
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1e-6
    pad = (hi - lo) * PAD_FRAC
    return float(lo - pad), float(hi + pad)


def plot_series_grid(chips: List[Dict], title: str, ylab: str):
    n = len(chips)
    if n == 0:
        return

    ylims = None
    if ROBUST_YLIM:
        ylims = compute_global_ylims([c["series"] for c in chips])

    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(13, 4.5 * rows))
    for i, c in enumerate(chips, start=1):
        plt.subplot(rows, cols, i)
        plt.plot(c["x"], c["series"], linewidth=LINE_WIDTH)
        plt.scatter(c["x"], c["series"], s=SCATTER_SIZE)

        if min(c["dates"]) <= EVENT_DATE <= max(c["dates"]):
            plt.axvline(datetime(EVENT_DATE.year, EVENT_DATE.month, EVENT_DATE.day), linestyle="--")

        plt.title(c["name"])
        plt.ylabel(ylab)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        if ylims is not None:
            plt.ylim(*ylims)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def process_chip(path: str, label: str) -> Tuple[Dict, Dict]:
    base = os.path.splitext(os.path.basename(path))[0]
    name = f"{label} | {base}"

    stack, dates = read_vv_stack_bestband(path)  # <<< KEEP BEST BAND PER DATE

    if CLAMP_VV:
        stack = clamp(stack, CLAMP_LO, CLAMP_HI)

    baseline_idx = choose_baseline_idx(dates)
    mu, iqr = baseline_mu_iqr(stack, baseline_idx)
    scores = score_stack(stack, mu, iqr)

    thr = threshold_from_baseline(scores, baseline_idx, BASELINE_Q)
    ft = flooded_fraction_series(scores, thr)
    pt = dark_fraction_series(stack)

    if PLOT_IN_PERCENT:
        ft = ft * 100.0
        pt = pt * 100.0

    if DEBUG_PRINT:
        print(f"\n{name}")
        print(f"  UNIQUE dates(T)={len(dates)} | range={dates[0]}..{dates[-1]}")
        print(f"  baseline_idx_count={len(baseline_idx)}")
        print(f"  THR(Q{BASELINE_Q})={thr:.3f}")
        print(f"  ft finite count={np.isfinite(ft).sum()}/{len(ft)} | range={np.nanmin(ft):.6f}..{np.nanmax(ft):.6f}")
        print(f"  pt finite count={np.isfinite(pt).sum()}/{len(pt)} | range={np.nanmin(pt):.6f}..{np.nanmax(pt):.6f}")

    x = to_datetimes(dates)

    flood_chip = {"name": f"{name}\nTHR(Q{BASELINE_Q})={thr:.3f}", "dates": dates, "x": x, "series": ft}
    dark_chip = {"name": name, "dates": dates, "x": x, "series": pt}
    return flood_chip, dark_chip


def run_folder(folder: str, label: str, n_plot: int) -> Tuple[List[Dict], List[Dict]]:
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    files = files[: min(n_plot, len(files))]

    flood_list, dark_list = [], []
    for path in files:
        f, d = process_chip(path, label)
        flood_list.append(f)
        dark_list.append(d)
    return flood_list, dark_list


def main():
    arl0_flood, arl0_dark = run_folder(ARL0_DIR, "ARL0", N_PLOT_ARL0)
    arl1_flood, arl1_dark = run_folder(ARL1_DIR, "ARL1", N_PLOT_ARL1)

    flood_all = arl0_flood + arl1_flood
    dark_all = arl0_dark + arl1_dark

    unit = "%" if PLOT_IN_PERCENT else "fraction (0.03=3%)"

    plot_series_grid(
        flood_all,
        title=f"Graph 1: Flooded {unit} per date (Score>=THR; THR from baseline Q{BASELINE_Q})\n(best-valid band per date)",
        ylab=f"Flooded pixels ({unit})"
    )

    plot_series_grid(
        dark_all,
        title=f"Graph 2: Dark-pixel {unit} per date (VV in [{DARK_LO},{DARK_HI}] dB)\n(best-valid band per date)",
        ylab=f"Dark pixels ({unit})"
    )

    print("\nFix applied:")
    print("  For each date, we keep ONLY the band with the most valid pixels (likely S1A vs S1B).")
    print("  This removes the alternating all-nodata bands, giving a clean continuous timeline.")


if __name__ == "__main__":
    main()
