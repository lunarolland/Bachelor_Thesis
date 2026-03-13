import csv
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
import matplotlib.pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects.packages import importr


# =========================
# Config (edit these)
# =========================
TIF_PATH = "/Users/lunarolland/Desktop/DATASETS/Spain_MIXED_BASE_OUTSIDE_EXTRA_FOCUS_NODUP.tif"
OUT_CSV  = "/Users/lunarolland/Desktop/THESIS/flood_prop_bcp.csv"

# Prior change probability p0 ~ 1/k, where k is expected segment length
EXPECTED_BLOCK_LEN_K = 20.0       # p0 = 1/20 = 0.05

# BCP sampling
MCMC_SAMPLES = 2000
BURNIN = 500

# Fusion and thresholding
FUSION = "vv"                     # "vv" | "vh" | "avg" | "or"
THRESHOLD = 0.10                  # classify pixel flooded at time t if fused_prob[t] > THRESHOLD

# ===== NEW: fixed window subset (top-right corner) =====
# Set to (60, 60) for your requested subset; set to None for full run
SUBSET_WINDOW_HW: Optional[Tuple[int, int]] = (60, 60)  # (height, width)

# Optional: clip dB ranges (helps if weird outliers)
CLIP = True
VV_RANGE = (-25.0, 0.0)
VH_RANGE = (-35.0, -5.0)


# =========================
# Band parsing + stack build
# =========================
BAND_RE = re.compile(r"^(VV|VH)_(\d{8})$")

@dataclass
class TimeStacks:
    vv: np.ndarray               # (T,H,W)
    vh: np.ndarray               # (T,H,W)
    dates: List[datetime]        # length T
    valid_hw: np.ndarray         # (H,W) bool


def _parse_desc(desc: str) -> Tuple[str, datetime]:
    if desc is None:
        raise ValueError("A band description is None. Need 'VV_YYYYMMDD' / 'VH_YYYYMMDD'.")
    m = BAND_RE.match(desc.strip())
    if not m:
        raise ValueError(f"Band '{desc}' does not match expected pattern VV_YYYYMMDD / VH_YYYYMMDD")
    pol = m.group(1)
    dt = datetime.strptime(m.group(2), "%Y%m%d")
    return pol, dt


def load_stacks_from_tif(path: str) -> TimeStacks:
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        nb = ds.count
        descs = list(ds.descriptions)

        if any(d is None for d in descs):
            raise ValueError("Some band descriptions are missing; cannot map VV/VH to dates.")

        # Map date -> {"VV": band_index, "VH": band_index}
        mapping: Dict[datetime, Dict[str, int]] = {}
        for b in range(1, nb + 1):
            pol, dt = _parse_desc(descs[b - 1])
            mapping.setdefault(dt, {})
            if pol in mapping[dt]:
                raise ValueError(f"Duplicate {pol} band for date {dt.date()} (band {b})")
            mapping[dt][pol] = b

        dates = sorted([dt for dt in mapping.keys() if "VV" in mapping[dt] and "VH" in mapping[dt]])
        if not dates:
            raise ValueError("No timestamps had BOTH VV and VH bands.")

        T = len(dates)
        vv = np.full((T, H, W), np.nan, dtype=np.float32)
        vh = np.full((T, H, W), np.nan, dtype=np.float32)

        for i, dt in enumerate(dates):
            vv[i] = ds.read(mapping[dt]["VV"]).astype(np.float32)
            vh[i] = ds.read(mapping[dt]["VH"]).astype(np.float32)

    if CLIP:
        vv = np.clip(vv, VV_RANGE[0], VV_RANGE[1])
        vh = np.clip(vh, VH_RANGE[0], VH_RANGE[1])

    valid_hw = np.all(np.isfinite(vv) & np.isfinite(vh), axis=0)

    print("Loaded:", path)
    print("T,H,W =", vv.shape)
    print("Valid pixels:", int(valid_hw.sum()), "/", valid_hw.size)
    print("Dates span:", dates[0].date(), "->", dates[-1].date())
    return TimeStacks(vv=vv, vh=vh, dates=dates, valid_hw=valid_hw)


# =========================
# R bcp runner (one series)
# =========================
_bcp = importr("bcp")  # fail fast if missing


def bcp_posterior_change_prob(series: np.ndarray, p0: float, mcmc: int, burnin: int) -> np.ndarray:
    """
    Calls R: bcp::bcp(y, p0=..., mcmc=..., burnin=..., return.mcmc=FALSE)
    Returns posterior change-point probabilities length T.
    """
    y = np.asarray(series, dtype=float)

    # bcp doesn't like NA/NaN
    if not np.all(np.isfinite(y)):
        med = np.nanmedian(y)
        y = np.where(np.isfinite(y), y, med)

    r_y = ro.FloatVector(y.tolist())
    fit = _bcp.bcp(r_y, p0=p0, mcmc=mcmc, burnin=burnin, return_mcmc=False)

    # bcp output field is usually "posterior.prob"
    if "posterior.prob" in fit.names:
        p = np.array(fit.rx2("posterior.prob"), dtype=float)
    else:
        raise RuntimeError(f"Unexpected bcp output fields: {list(fit.names)}")

    # safety: ensure length matches
    if p.shape[0] != y.shape[0]:
        T = y.shape[0]
        if p.shape[0] > T:
            p = p[:T]
        else:
            p = np.pad(p, (0, T - p.shape[0]), mode="edge")
    return p


# =========================
# Fusion + plotting
# =========================
def fuse(vv_p: np.ndarray, vh_p: np.ndarray, mode: str) -> np.ndarray:
    """
    vv_p, vh_p: (N,T)
    returns (N,T)
    """
    mode = mode.lower()
    if mode == "vv":
        return vv_p
    if mode == "vh":
        return vh_p
    if mode == "avg":
        return 0.5 * (vv_p + vh_p)
    if mode == "or":
        # probabilistic OR assuming independence-ish
        return 1.0 - (1.0 - vv_p) * (1.0 - vh_p)
    raise ValueError("FUSION must be one of: vv, vh, avg, or")


def plot_time_series(dates: List[datetime], y: np.ndarray, title: str, ylabel: str):
    plt.figure()
    plt.plot(dates, y)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# =========================
# Main pipeline
# =========================
def run(
    tif_path: str,
    out_csv: str,
    expected_block_len_k: float,
    mcmc: int,
    burnin: int,
    fusion_mode: str,
    threshold: float,
    subset_window_hw: Optional[Tuple[int, int]] = (60, 60),
):
    stacks = load_stacks_from_tif(tif_path)
    vv, vh, dates, valid_hw = stacks.vv, stacks.vh, stacks.dates, stacks.valid_hw
    T, H, W = vv.shape
    p0 = 1.0 / float(expected_block_len_k)

    # ===== NEW: fixed window subset in the top-right corner =====
    if subset_window_hw is not None:
        win_h, win_w = subset_window_hw
        win_h = min(int(win_h), H)
        win_w = min(int(win_w), W)

        rr = slice(0, win_h)       # top rows
        cc = slice(W - win_w, W)   # rightmost cols

        # crop stacks: vv/vh are (T,H,W) -> (T,win_h,win_w)
        vv_win = vv[:, rr, cc]
        vh_win = vh[:, rr, cc]
        valid_win = valid_hw[rr, cc]

        # flatten window to (Nwin,T)
        Nwin = win_h * win_w
        vv_2d = vv_win.reshape(T, Nwin).T
        vh_2d = vh_win.reshape(T, Nwin).T
        valid_1d = valid_win.reshape(Nwin)

        vv_valid = vv_2d[valid_1d, :]
        vh_valid = vh_2d[valid_1d, :]

        N = vv_valid.shape[0]
        print(f"Using top-right window {win_h}x{win_w} => {Nwin} pixels ({N} valid after masking).")
    else:
        # Full image (original behavior, but without random subsetting)
        Npix = H * W
        valid_1d = valid_hw.reshape(Npix)

        vv_2d = vv.reshape(T, Npix).T
        vh_2d = vh.reshape(T, Npix).T

        vv_valid = vv_2d[valid_1d, :]
        vh_valid = vh_2d[valid_1d, :]

        N = vv_valid.shape[0]
        print(f"Using full image: valid pixel time series N={N}, T={T}")

    # Allocate posterior prob arrays
    pvv = np.empty((N, T), dtype=np.float32)
    pvh = np.empty((N, T), dtype=np.float32)

    print(f"Running bcp per pixel (sequential). p0={p0:.4f}, mcmc={mcmc}, burnin={burnin}")
    for i in range(N):
        if (i + 1) % 100 == 0:
            print(f"  pixel {i+1}/{N}")

        pvv[i] = bcp_posterior_change_prob(vv_valid[i], p0=p0, mcmc=mcmc, burnin=burnin).astype(np.float32)
        pvh[i] = bcp_posterior_change_prob(vh_valid[i], p0=p0, mcmc=mcmc, burnin=burnin).astype(np.float32)

    pfused = fuse(pvv, pvh, fusion_mode)

    # Proportion flooded each time t
    flood_prop = np.mean(pfused > threshold, axis=0)

    # Useful extra: max prob each time (often spikes at events)
    max_prob = np.max(pfused, axis=0)

    # Save CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "flooded_pixel_proportion", "max_posterior_cp_prob"])
        for dt, prop, mx in zip(dates, flood_prop, max_prob):
            w.writerow([dt.strftime("%Y-%m-%d"), float(prop), float(mx)])
    print("Saved:", out_csv)

    # Plot
    plot_time_series(
        dates,
        flood_prop,
        title=f"Flood proportion over time (bcp via rpy2, fusion={fusion_mode}, thr={threshold}, p0={p0:.4f})",
        ylabel="Proportion of pixels flagged (p_cp > thr)",
    )
    plot_time_series(
        dates,
        max_prob,
        title="Max posterior change probability across pixels per date",
        ylabel="max(p_cp)",
    )

    return dates, flood_prop, max_prob


if __name__ == "__main__":
    run(
        tif_path=TIF_PATH,
        out_csv=OUT_CSV,
        expected_block_len_k=EXPECTED_BLOCK_LEN_K,
        mcmc=MCMC_SAMPLES,
        burnin=BURNIN,
        fusion_mode=FUSION,
        threshold=THRESHOLD,
        subset_window_hw=SUBSET_WINDOW_HW,
    )