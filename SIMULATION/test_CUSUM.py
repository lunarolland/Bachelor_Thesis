"""
Flood pond simulation harness + ARL0/ARL1 CUSUM benchmarking + parameter sweep
NOW INCLUDING: sweep over (smooth_k, kappa, h) + FAR columns in sweep output
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# =========================================================
# RUN MODES
# =========================================================
# "main_only"     -> run sims, compute CUSUM tables, ARL tables, parameter sweep
# "k_test_only"   -> only run k×k smoothing limit experiment
# "main_then_k"   -> run main + then run k test (can be slow)
RUN_MODE = "main_only"


# =========================================================
# USER SETTINGS
# =========================================================
H, W = 100, 100
T = 240
BASELINE_DAYS = 90
Q_THR = 99.99

# Simulation counts
N_SINGLE = 250
N_DOUBLE = 250
N_PERMANENT_OVERFLOW = 250

# Dedicated ARL0 (no pond) simulations
N_ARL0 = 600
# Extra “harsher-noise” null sims (still no pond)
N_ARL0_HEAVY_NOISE = 600

# Output
OUT_DIR = "./sim_outputs"
SAVE_METRICS_NPZ = True
SAVE_FULL_CUBES = False
PLOT_EXAMPLES_PER_SCENARIO = 0  # set >0 if you want plots
EXAMPLE_FRAMES = [0, 40, 80, 100, 120, 160, 239]

# VV range and dark range
VV_MIN, VV_MAX = -25.0, 0.0
DARK_LO, DARK_HI = -25.0, -17.0

# Background / clutter
BG_MEAN = -10.0
BG_SIGMA = 1.3

SPECKLE_P = 0.015
SPECKLE_RANGE = (-16.5, -13.0)

# Flood pond texture
POND_MEAN = -22.5
POND_SIGMA = 0.6

# Bump shape controls
DURATION_RANGE = (25, 70)
FLOOD_DAY_RANGE = (BASELINE_DAYS + 10, T - 30)

# Radii ranges
R_SINGLE_RANGE = (6.0, 22.0)
R_DOUBLE_RANGE = (5.0, 16.0)
R_PERMANENT_RANGE = (4.0, 10.0)
R_OVERFLOW_EXTRA_RANGE = (6.0, 20.0)

MARGIN = 10
MASTER_SEED = 123


# =========================================================
# OPTIONAL: k×k smoothing limit test
# =========================================================
KTEST_ENABLE = False  # independent from RUN_MODE; RUN_MODE="main_then_k" will run it too
K_LIST = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 91, 99]
R_PEAK_GRID = np.concatenate([
    np.arange(0.25, 2.25, 0.25),
    np.arange(2.0, 8.0, 0.5),
    np.arange(8.0, 26.0, 1.0),
]).astype(np.float32)
N_SIMS_PER_RADIUS = 15
OUT_DIR_KTEST = "./sim_outputs_k_tests"
POST_FLOOD_WINDOW = (-5, 40)


# =========================================================
# CUSUM + PARAMETER SWEEP SETTINGS
# =========================================================
CUSUM_BURN_IN = BASELINE_DAYS

# Parameter sweep grids
KAPPA_GRID = np.array([0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float32)
H_GRID = np.array([6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0], dtype=np.float32)

# SMOOTHING SWEEP (THIS IS NOW REAL)
# These are box filter sizes (odd). 1 means no smoothing.
SMOOTH_K_SWEEP = [1, 3, 5, 9, 15]  # adjust as desired

# Objective constraints / scoring for “best params”
TARGET_ARL0_MIN = 80.0      # want ARL0 >= this (bigger means fewer false alarms)
TARGET_ARL1_MAX = 25.0      # want ARL1 (mean run length after change) <= this
MIN_DETECT_RATE = 0.90      # want detection rate >= this under change

# Outputs
OUT_CUSUM_PER_SIM = os.path.join(OUT_DIR, "cusum_per_sim_table.csv")
OUT_ARL_TABLE = os.path.join(OUT_DIR, "arl_table.csv")
OUT_PARAM_SWEEP = os.path.join(OUT_DIR, "cusum_param_sweep.csv")
OUT_PARAM_BEST = os.path.join(OUT_DIR, "cusum_best_params.csv")


# =========================================================
# UTILITIES
# =========================================================
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def clamp(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip(arr, vmin, vmax)

def disk_mask(H: int, W: int, cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

def smooth_bump(t: np.ndarray, t_peak: float, width: float) -> np.ndarray:
    sigma = max(width / 2.5, 1e-6)
    bump = np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)
    bump = bump / bump.max()
    return bump.astype(np.float32)

def cap_odd_k(k: int, H: int, W: int) -> int:
    kmax = min(H, W)
    if kmax % 2 == 0:
        kmax -= 1
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return int(min(k, kmax))


# =========================================================
# BACKGROUND GENERATORS (null models)
# =========================================================
def make_background(rng: np.random.Generator,
                    bg_mean: float = BG_MEAN,
                    bg_sigma: float = BG_SIGMA,
                    speckle_p: float = SPECKLE_P,
                    speckle_range: Tuple[float, float] = SPECKLE_RANGE) -> np.ndarray:
    bg = rng.normal(bg_mean, bg_sigma, size=(H, W)).astype(np.float32)
    speck = rng.random((H, W)) < speckle_p
    if speck.any():
        bg[speck] = rng.uniform(speckle_range[0], speckle_range[1], size=speck.sum()).astype(np.float32)
    return bg


# =========================================================
# BASELINE + SCORE + METRICS
# =========================================================
def robust_baseline(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    iqr = q75 - q25
    return med.astype(np.float32), iqr.astype(np.float32)

def flood_score_vv(VV_t: np.ndarray, VV_med: np.ndarray, VV_iqr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    z = (VV_t - VV_med) / (VV_iqr + eps)
    return (-z).astype(np.float32)

def baseline_threshold(scores_baseline: np.ndarray, q: float) -> float:
    vals = scores_baseline[np.isfinite(scores_baseline)]
    return float(np.percentile(vals, q))

def percent_dark(VV_t: np.ndarray) -> float:
    m = (VV_t >= DARK_LO) & (VV_t <= DARK_HI)
    return float(m.mean()) * 100.0

def compute_metrics(VV: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    baseline = VV[:BASELINE_DAYS]
    VV_med, VV_iqr = robust_baseline(baseline)
    scores_b = np.stack([flood_score_vv(baseline[i], VV_med, VV_iqr) for i in range(BASELINE_DAYS)], axis=0)
    thr = baseline_threshold(scores_b, Q_THR)

    pct_dark = np.array([percent_dark(VV[i]) for i in range(T)], dtype=np.float32)

    pct_flood = np.zeros((T,), dtype=np.float32)
    for i in range(T):
        sc = flood_score_vv(VV[i], VV_med, VV_iqr)
        pct_flood[i] = float((sc >= thr).mean()) * 100.0

    return thr, pct_dark, pct_flood


# =========================================================
# OPTIONAL SPATIAL SMOOTHING
# =========================================================
def box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    if k % 2 == 0:
        raise ValueError("k must be odd")
    if k == 1:
        return arr.astype(np.float32)

    pad = k // 2
    a = arr.astype(np.float32)

    valid = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    a0p = np.pad(a0, pad, mode="reflect")
    vp = np.pad(valid, pad, mode="reflect")

    S = a0p.cumsum(0).cumsum(1)
    SV = vp.cumsum(0).cumsum(1)

    Hh, Ww = a.shape
    x2 = np.arange(k - 1, k - 1 + Hh)
    y2 = np.arange(k - 1, k - 1 + Ww)

    A = S[np.ix_(x2, y2)]
    B = S[np.ix_(x2 - k, y2)]
    C = S[np.ix_(x2, y2 - k)]
    D = S[np.ix_(x2 - k, y2 - k)]
    win_sum = A - B - C + D

    Av = SV[np.ix_(x2, y2)]
    Bv = SV[np.ix_(x2 - k, y2)]
    Cv = SV[np.ix_(x2, y2 - k)]
    Dv = SV[np.ix_(x2 - k, y2 - k)]
    win_cnt = Av - Bv - Cv + Dv

    out = win_sum / np.maximum(win_cnt, 1.0)
    out[win_cnt == 0] = np.nan
    return out.astype(np.float32)

def apply_spatial_smoothing_stack(VV: np.ndarray, k: int) -> np.ndarray:
    k = cap_odd_k(k, VV.shape[1], VV.shape[2])
    if k == 1:
        return VV.astype(np.float32)
    out = np.empty_like(VV, dtype=np.float32)
    for ti in range(VV.shape[0]):
        out[ti] = box_mean(VV[ti], k)
    return out


# =========================================================
# SCENARIOS + METADATA
# =========================================================
@dataclass
class SimMeta:
    scenario: str
    flood_day: int
    width: float
    ponds: List[Dict]
    onset_day: Optional[int] = None    # change-point for ARL1
    offset_day: Optional[int] = None

def sample_center(rng: np.random.Generator) -> Tuple[int, int]:
    cy = int(rng.integers(MARGIN, H - MARGIN))
    cx = int(rng.integers(MARGIN, W - MARGIN))
    return cy, cx


def gen_no_pond(rng: np.random.Generator,
                bg_mean: float = BG_MEAN,
                bg_sigma: float = BG_SIGMA,
                speckle_p: float = SPECKLE_P,
                speckle_range: Tuple[float, float] = SPECKLE_RANGE) -> Tuple[np.ndarray, SimMeta]:
    """Null: background + speckles only. No change-point."""
    VV = np.zeros((T, H, W), dtype=np.float32)
    for ti in range(T):
        bg = make_background(rng, bg_mean=bg_mean, bg_sigma=bg_sigma,
                             speckle_p=speckle_p, speckle_range=speckle_range)
        VV[ti] = clamp(bg, VV_MIN, VV_MAX)
    meta = SimMeta(
        scenario="no_pond",
        flood_day=-1,
        width=0.0,
        ponds=[],
        onset_day=None,
        offset_day=None
    )
    return VV, meta


def gen_single_transient(rng: np.random.Generator) -> Tuple[np.ndarray, SimMeta]:
    flood_day = int(rng.integers(FLOOD_DAY_RANGE[0], FLOOD_DAY_RANGE[1]))
    width = float(rng.uniform(DURATION_RANGE[0], DURATION_RANGE[1]))
    cy, cx = sample_center(rng)
    r_peak = float(rng.uniform(R_SINGLE_RANGE[0], R_SINGLE_RANGE[1]))

    t = np.arange(T, dtype=np.float32)
    bump = smooth_bump(t, flood_day, width)
    r_t = r_peak * bump

    active = r_t > 0.5
    onset_day = int(np.argmax(active)) if np.any(active) else None
    offset_day = int(len(active) - 1 - np.argmax(active[::-1])) if np.any(active) else None

    VV = np.zeros((T, H, W), dtype=np.float32)
    for ti in range(T):
        bg = make_background(rng)
        if r_t[ti] > 0.5:
            mask = disk_mask(H, W, cy, cx, float(r_t[ti]))
            pond_vals = rng.normal(POND_MEAN, POND_SIGMA, size=(H, W)).astype(np.float32)
            bg[mask] = pond_vals[mask]
        VV[ti] = clamp(bg, VV_MIN, VV_MAX)

    meta = SimMeta(
        scenario="single_transient",
        flood_day=flood_day,
        width=width,
        ponds=[{"cy": cy, "cx": cx, "r_peak": r_peak, "kind": "transient"}],
        onset_day=onset_day,
        offset_day=offset_day
    )
    return VV, meta


def gen_two_transient(rng: np.random.Generator) -> Tuple[np.ndarray, SimMeta]:
    flood_day = int(rng.integers(FLOOD_DAY_RANGE[0], FLOOD_DAY_RANGE[1]))
    width = float(rng.uniform(DURATION_RANGE[0], DURATION_RANGE[1]))

    cy1, cx1 = sample_center(rng)
    cy2, cx2 = sample_center(rng)
    r1 = float(rng.uniform(R_DOUBLE_RANGE[0], R_DOUBLE_RANGE[1]))
    r2 = float(rng.uniform(R_DOUBLE_RANGE[0], R_DOUBLE_RANGE[1]))

    t = np.arange(T, dtype=np.float32)
    bump = smooth_bump(t, flood_day, width)
    r1_t = r1 * bump
    r2_t = r2 * bump

    active = (r1_t > 0.5) | (r2_t > 0.5)
    onset_day = int(np.argmax(active)) if np.any(active) else None
    offset_day = int(len(active) - 1 - np.argmax(active[::-1])) if np.any(active) else None

    VV = np.zeros((T, H, W), dtype=np.float32)
    for ti in range(T):
        bg = make_background(rng)
        if r1_t[ti] > 0.5:
            m1 = disk_mask(H, W, cy1, cx1, float(r1_t[ti]))
            pond_vals = rng.normal(POND_MEAN, POND_SIGMA, size=(H, W)).astype(np.float32)
            bg[m1] = pond_vals[m1]
        if r2_t[ti] > 0.5:
            m2 = disk_mask(H, W, cy2, cx2, float(r2_t[ti]))
            pond_vals = rng.normal(POND_MEAN, POND_SIGMA, size=(H, W)).astype(np.float32)
            bg[m2] = pond_vals[m2]
        VV[ti] = clamp(bg, VV_MIN, VV_MAX)

    meta = SimMeta(
        scenario="two_transient",
        flood_day=flood_day,
        width=width,
        ponds=[
            {"cy": cy1, "cx": cx1, "r_peak": r1, "kind": "transient"},
            {"cy": cy2, "cx": cx2, "r_peak": r2, "kind": "transient"},
        ],
        onset_day=onset_day,
        offset_day=offset_day
    )
    return VV, meta


def gen_permanent_plus_overflow(rng: np.random.Generator) -> Tuple[np.ndarray, SimMeta]:
    flood_day = int(rng.integers(FLOOD_DAY_RANGE[0], FLOOD_DAY_RANGE[1]))
    width = float(rng.uniform(DURATION_RANGE[0], DURATION_RANGE[1]))
    cy, cx = sample_center(rng)

    r_base = float(rng.uniform(R_PERMANENT_RANGE[0], R_PERMANENT_RANGE[1]))
    r_extra = float(rng.uniform(R_OVERFLOW_EXTRA_RANGE[0], R_OVERFLOW_EXTRA_RANGE[1]))

    t = np.arange(T, dtype=np.float32)
    bump = smooth_bump(t, flood_day, width)
    r_over_t = r_base + r_extra * bump

    overflow_active = r_over_t > (r_base + 0.5)
    onset_day = int(np.argmax(overflow_active)) if np.any(overflow_active) else None
    offset_day = int(len(overflow_active) - 1 - np.argmax(overflow_active[::-1])) if np.any(overflow_active) else None

    VV = np.zeros((T, H, W), dtype=np.float32)
    base_mask = disk_mask(H, W, cy, cx, r_base)

    for ti in range(T):
        bg = make_background(rng)

        pond_vals = rng.normal(POND_MEAN, POND_SIGMA, size=(H, W)).astype(np.float32)
        bg[base_mask] = pond_vals[base_mask]

        if r_over_t[ti] > r_base + 0.5:
            m_over = disk_mask(H, W, cy, cx, float(r_over_t[ti]))
            pond_vals2 = rng.normal(POND_MEAN - 0.3, POND_SIGMA, size=(H, W)).astype(np.float32)
            bg[m_over] = pond_vals2[m_over]

        VV[ti] = clamp(bg, VV_MIN, VV_MAX)

    meta = SimMeta(
        scenario="permanent_overflow",
        flood_day=flood_day,
        width=width,
        ponds=[{"cy": cy, "cx": cx, "r_base": r_base, "r_extra": r_extra, "kind": "permanent+overflow"}],
        onset_day=onset_day,
        offset_day=offset_day
    )
    return VV, meta


# =========================================================
# PLOTTING (optional)
# =========================================================
def plot_example(VV: np.ndarray, pct_dark: np.ndarray, pct_flood: np.ndarray, meta: SimMeta, sim_id: int):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"{meta.scenario} | sim {sim_id} | flood_day={meta.flood_day} | width={meta.width:.1f}", y=0.98)
    for j, f in enumerate(EXAMPLE_FRAMES, start=1):
        ax = plt.subplot(2, 4, j)
        ax.imshow(VV[f], vmin=VV_MIN, vmax=VV_MAX, cmap="gray")
        ax.set_title(f"Day {f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    days = np.arange(T)
    plt.figure(figsize=(12, 4))
    plt.plot(days, pct_dark)
    if meta.onset_day is not None:
        plt.axvline(meta.onset_day, linestyle="--")
    plt.title(f"{meta.scenario} | sim {sim_id} — % dark")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(days, pct_flood)
    if meta.onset_day is not None:
        plt.axvline(meta.onset_day, linestyle="--")
    plt.title(f"{meta.scenario} | sim {sim_id} — % flooded")
    plt.tight_layout()
    plt.show()


# =========================================================
# SIM RUNNER
# =========================================================
def run_scenario(gen_fn, name: str, n_sims: int, rng: np.random.Generator, smooth_k: int = 1):
    thr_list = np.zeros((n_sims,), dtype=np.float32)
    pct_dark_all = np.zeros((n_sims, T), dtype=np.float32)
    pct_flood_all = np.zeros((n_sims, T), dtype=np.float32)
    metas: List[Dict] = []

    cubes = None
    if SAVE_FULL_CUBES:
        cubes = np.zeros((n_sims, T, H, W), dtype=np.float32)

    for i in range(n_sims):
        VV, meta = gen_fn(rng)
        if smooth_k != 1:
            VV = apply_spatial_smoothing_stack(VV, smooth_k)

        thr, pct_dark, pct_flood = compute_metrics(VV)

        thr_list[i] = thr
        pct_dark_all[i] = pct_dark
        pct_flood_all[i] = pct_flood
        metas.append(meta.__dict__)

        if SAVE_FULL_CUBES:
            cubes[i] = VV

        if PLOT_EXAMPLES_PER_SCENARIO and i < PLOT_EXAMPLES_PER_SCENARIO:
            plot_example(VV, pct_dark, pct_flood, meta, sim_id=i)

        if (i + 1) % 100 == 0:
            print(f"{name} (smooth_k={smooth_k}): finished {i+1}/{n_sims}")

    return {
        "thr": thr_list,
        "pct_dark": pct_dark_all,
        "pct_flood": pct_flood_all,
        "meta": metas,
        "cubes": cubes,
        "smooth_k": smooth_k
    }


# =========================================================
# CUSUM + ARL COMPUTATION
# =========================================================
def cusum_one_sided_up(x: np.ndarray, burn_in: int, kappa: float, h: float) -> Optional[int]:
    """
    One-sided (up) CUSUM on standardized series.
    Returns alarm day index, or None if no alarm.
    """
    x = np.asarray(x, dtype=np.float32)
    burn_in = int(burn_in)
    if burn_in < 2:
        burn_in = 2
    if burn_in >= len(x):
        return None

    mu = float(np.mean(x[:burn_in]))
    sd = float(np.std(x[:burn_in]) + 1e-6)
    z = (x - mu) / sd

    S = 0.0
    for t in range(burn_in, len(z)):
        S = max(0.0, S + (float(z[t]) - float(kappa)))
        if S >= float(h):
            return t
    return None


def run_cusum_on_results(all_results: Dict[str, dict], burn_in: int, kappa: float, h: float) -> pd.DataFrame:
    rows = []
    for scenario, res in all_results.items():
        pct_dark_all = res["pct_dark"]
        pct_flood_all = res["pct_flood"]
        metas = res["meta"]

        for i in range(pct_dark_all.shape[0]):
            onset_day = metas[i].get("onset_day", None)
            flood_day = metas[i].get("flood_day", None)

            alarm_f = cusum_one_sided_up(pct_flood_all[i], burn_in=burn_in, kappa=kappa, h=h)
            alarm_d = cusum_one_sided_up(pct_dark_all[i],  burn_in=burn_in, kappa=kappa, h=h)

            delay_onset_f = None if (alarm_f is None or onset_day is None) else int(alarm_f - int(onset_day))
            delay_onset_d = None if (alarm_d is None or onset_day is None) else int(alarm_d - int(onset_day))

            rows.append({
                "scenario": scenario,
                "sim_id": i,
                "onset_day": onset_day,
                "flood_day": flood_day,
                "cusum_alarm_day_flood": alarm_f,
                "cusum_alarm_day_dark": alarm_d,
                "delay_vs_onset_flood": delay_onset_f,
                "delay_vs_onset_dark": delay_onset_d,
            })
    return pd.DataFrame(rows)


def arl0_from_alarms(alarm_days: np.ndarray, burn_in: int, T: int) -> Tuple[float, float]:
    """
    ARL0: average run length under null.
    Run length measured from burn_in to alarm (inclusive), or censored at end.
    Returns (ARL0_mean, false_alarm_rate)
    """
    burn_in = int(burn_in)
    rl = []
    fa = 0
    for a in alarm_days:
        if pd.isna(a):
            rl.append(T - burn_in)
        else:
            a = int(a)
            if a < burn_in:
                rl.append(1)
                fa += 1
            else:
                rl.append(a - burn_in + 1)
                fa += 1
    rl = np.array(rl, dtype=np.float32)
    return float(rl.mean()), float(fa / len(alarm_days))


def arl1_from_alarms(alarm_days: np.ndarray, onset_days: np.ndarray, T: int) -> Tuple[float, float]:
    """
    ARL1: average run length after change-point (onset).
    Run length measured from onset to alarm (inclusive), or censored at end.
    Returns (ARL1_mean, detect_rate) where detect_rate is among sims with valid onset.
    """
    rl = []
    det = 0
    n_valid = 0
    for a, o in zip(alarm_days, onset_days):
        if o is None or pd.isna(o):
            continue
        n_valid += 1
        o = int(o)
        if pd.isna(a):
            rl.append(T - o)
        else:
            a = int(a)
            if a < o:
                rl.append(1)
                det += 1
            else:
                rl.append(a - o + 1)
                det += 1

    if n_valid == 0:
        return float("nan"), 0.0
    rl = np.array(rl, dtype=np.float32)
    return float(rl.mean()), float(det / n_valid)


def make_arl_summary_table(df_per_sim: pd.DataFrame, burn_in: int) -> pd.DataFrame:
    rows = []
    for scenario, g in df_per_sim.groupby("scenario"):
        onset = g["onset_day"].values

        arl0_f, far_f = arl0_from_alarms(g["cusum_alarm_day_flood"].values, burn_in=burn_in, T=T)
        arl1_f, dr_f = arl1_from_alarms(g["cusum_alarm_day_flood"].values, onset, T=T)

        arl0_d, far_d = arl0_from_alarms(g["cusum_alarm_day_dark"].values, burn_in=burn_in, T=T)
        arl1_d, dr_d = arl1_from_alarms(g["cusum_alarm_day_dark"].values, onset, T=T)

        rows.append({
            "scenario": scenario,
            "N": int(len(g)),
            "ARL0_flood": arl0_f,
            "FAR_flood": far_f,
            "ARL1_flood": arl1_f,
            "DetectRate_flood": dr_f,
            "ARL0_dark": arl0_d,
            "FAR_dark": far_d,
            "ARL1_dark": arl1_d,
            "DetectRate_dark": dr_d,
        })
    return pd.DataFrame(rows)


# =========================================================
# PARAMETER SWEEP + “BEST” SELECTION
# =========================================================
def score_params(arl0_f: float, arl0_d: float,
                 arl1_f: float, arl1_d: float,
                 dr_f: float, dr_d: float) -> Tuple[float, bool]:
    """
    Objective:
      - Must satisfy:
          avg_ARL0 >= TARGET_ARL0_MIN
          avg_ARL1 <= TARGET_ARL1_MAX
          avg_detect_rate >= MIN_DETECT_RATE
      - Score = avg_ARL0 - 0.5*avg_ARL1  (higher better)
    """
    avg_arl0 = 0.5 * (arl0_f + arl0_d)
    avg_arl1 = 0.5 * (arl1_f + arl1_d) if (np.isfinite(arl1_f) and np.isfinite(arl1_d)) else float("inf")
    avg_dr = 0.5 * (dr_f + dr_d)

    ok = (avg_arl0 >= TARGET_ARL0_MIN) and (avg_arl1 <= TARGET_ARL1_MAX) and (avg_dr >= MIN_DETECT_RATE)
    score = avg_arl0 - 0.5 * avg_arl1
    return float(score), bool(ok)


def run_all_scenarios_for_smooth_k(rng: np.random.Generator, smooth_k: int) -> Dict[str, dict]:
    """Re-run all scenarios at a specific smoothing k, returning all_results dict."""
    smooth_k = cap_odd_k(smooth_k, H, W)

    print(f"\n=== [smooth_k={smooth_k}] Running scenario: single_transient (N={N_SINGLE}) ===")
    res_single = run_scenario(gen_single_transient, "single_transient", N_SINGLE, rng, smooth_k=smooth_k)

    print(f"\n=== [smooth_k={smooth_k}] Running scenario: two_transient (N={N_DOUBLE}) ===")
    res_two = run_scenario(gen_two_transient, "two_transient", N_DOUBLE, rng, smooth_k=smooth_k)

    print(f"\n=== [smooth_k={smooth_k}] Running scenario: permanent_overflow (N={N_PERMANENT_OVERFLOW}) ===")
    res_perm = run_scenario(gen_permanent_plus_overflow, "permanent_overflow", N_PERMANENT_OVERFLOW, rng, smooth_k=smooth_k)

    print(f"\n=== [smooth_k={smooth_k}] Running scenario: no_pond (ARL0) (N={N_ARL0}) ===")
    res_null = run_scenario(lambda rr: gen_no_pond(rr), "no_pond", N_ARL0, rng, smooth_k=smooth_k)

    print(f"\n=== [smooth_k={smooth_k}] Running scenario: no_pond_heavy_noise (ARL0 hard) (N={N_ARL0_HEAVY_NOISE}) ===")
    res_null_hard = run_scenario(
        lambda rr: gen_no_pond(
            rr,
            bg_mean=BG_MEAN - 0.5,
            bg_sigma=BG_SIGMA * 1.25,
            speckle_p=min(0.06, SPECKLE_P * 3.0),
            speckle_range=(SPECKLE_RANGE[0] - 0.5, SPECKLE_RANGE[1] - 0.5),
        ),
        "no_pond_heavy_noise",
        N_ARL0_HEAVY_NOISE,
        rng,
        smooth_k=smooth_k
    )

    all_results = {
        "single_transient": res_single,
        "two_transient": res_two,
        "permanent_overflow": res_perm,
        "no_pond": res_null,
        "no_pond_heavy_noise": res_null_hard,
    }
    return all_results


def run_param_sweep_over_smoothing(rng: np.random.Generator) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Sweeps (smooth_k, kappa, h), computes ARL/FAR tables, and records objective.

    Uses:
      - Nulls for ARL0/FAR: no_pond + no_pond_heavy_noise
      - Changes for ARL1/DR: single_transient, two_transient, permanent_overflow

    Returns:
      df_sweep: all rows
      best_row: dict of best parameters overall (after sorting)
    """
    records: List[Dict] = []

    # Important: to keep comparability across smooth_k, you can either:
    #  (A) reuse the same RNG stream (as below), meaning each smooth_k sees different sims, OR
    #  (B) reset seed per smooth_k so each smooth_k uses same raw sims.
    #
    # If you want (B), uncomment the rng reset line below.
    for smooth_k in SMOOTH_K_SWEEP:
        # rng = np.random.default_rng(MASTER_SEED + 1000 * int(smooth_k))  # option (B)
        all_results = run_all_scenarios_for_smooth_k(rng, smooth_k=smooth_k)

        if SAVE_METRICS_NPZ:
            for name, res in all_results.items():
                npz_path = os.path.join(OUT_DIR, f"{name}_metrics_smoothk{cap_odd_k(smooth_k,H,W)}.npz")
                meta_json = json.dumps(res["meta"])
                if SAVE_FULL_CUBES and res["cubes"] is not None:
                    np.savez_compressed(
                        npz_path,
                        thr=res["thr"],
                        pct_dark=res["pct_dark"],
                        pct_flood=res["pct_flood"],
                        meta_json=meta_json,
                        VV=res["cubes"],
                    )
                else:
                    np.savez_compressed(
                        npz_path,
                        thr=res["thr"],
                        pct_dark=res["pct_dark"],
                        pct_flood=res["pct_flood"],
                        meta_json=meta_json,
                    )
                print(f"Saved: {npz_path}")

        print(f"\n=== [smooth_k={smooth_k}] Running CUSUM parameter sweep (kappa,h) ===")
        for kappa in KAPPA_GRID:
            for h in H_GRID:
                df_sim = run_cusum_on_results(all_results, burn_in=CUSUM_BURN_IN, kappa=float(kappa), h=float(h))
                arl = make_arl_summary_table(df_sim, burn_in=CUSUM_BURN_IN)

                null_rows = arl[arl["scenario"].isin(["no_pond", "no_pond_heavy_noise"])]
                chg_rows = arl[arl["scenario"].isin(["single_transient", "two_transient", "permanent_overflow"])]

                if len(null_rows) == 0 or len(chg_rows) == 0:
                    continue

                # Null aggregation (ARL0 + FAR)
                arl0_f = float(null_rows["ARL0_flood"].mean())
                arl0_d = float(null_rows["ARL0_dark"].mean())
                far_f = float(null_rows["FAR_flood"].mean())
                far_d = float(null_rows["FAR_dark"].mean())

                # Change aggregation (ARL1 + detect rate)
                arl1_f = float(chg_rows["ARL1_flood"].mean())
                arl1_d = float(chg_rows["ARL1_dark"].mean())
                dr_f = float(chg_rows["DetectRate_flood"].mean())
                dr_d = float(chg_rows["DetectRate_dark"].mean())

                score, ok = score_params(arl0_f, arl0_d, arl1_f, arl1_d, dr_f, dr_d)

                records.append({
                    "smooth_k": int(cap_odd_k(smooth_k, H, W)),
                    "kappa": float(kappa),
                    "h": float(h),

                    "ARL0_flood_mean_null": arl0_f,
                    "ARL0_dark_mean_null": arl0_d,
                    "ARL0_avg_null": 0.5 * (arl0_f + arl0_d),

                    "FAR_flood_mean_null": far_f,
                    "FAR_dark_mean_null": far_d,
                    "FAR_avg_null": 0.5 * (far_f + far_d),

                    "ARL1_flood_mean_change": arl1_f,
                    "ARL1_dark_mean_change": arl1_d,
                    "ARL1_avg_change": 0.5 * (arl1_f + arl1_d),

                    "DetectRate_flood_mean_change": dr_f,
                    "DetectRate_dark_mean_change": dr_d,
                    "DetectRate_avg_change": 0.5 * (dr_f + dr_d),

                    "score": score,
                    "meets_constraints": ok
                })

    df = pd.DataFrame(records)
    if len(df):
        df = df.sort_values(["meets_constraints", "score"], ascending=[False, False]).reset_index(drop=True)
        best = df.iloc[0].to_dict()
    else:
        best = {}
    return df, best


# =========================================================
# k×k LIMIT TEST (optional, can be slow)
# =========================================================
def detection_summary(pct_flood: np.ndarray, flood_day: int) -> float:
    lo = max(flood_day + POST_FLOOD_WINDOW[0], 0)
    hi = min(flood_day + POST_FLOOD_WINDOW[1], len(pct_flood))
    if hi <= lo:
        return float(np.max(pct_flood))
    return float(np.max(pct_flood[lo:hi]))

def gen_single_transient_with_fixed_rpeak(rng: np.random.Generator, r_peak: float):
    flood_day = int(rng.integers(FLOOD_DAY_RANGE[0], FLOOD_DAY_RANGE[1]))
    width = float(rng.uniform(DURATION_RANGE[0], DURATION_RANGE[1]))
    cy, cx = sample_center(rng)

    t = np.arange(T, dtype=np.float32)
    bump = smooth_bump(t, flood_day, width)
    r_t = float(r_peak) * bump

    active = r_t > 0.5
    onset_day = int(np.argmax(active)) if np.any(active) else None
    offset_day = int(len(active) - 1 - np.argmax(active[::-1])) if np.any(active) else None

    VV = np.zeros((T, H, W), dtype=np.float32)
    for ti in range(T):
        bg = make_background(rng)
        if r_t[ti] > 0.5:
            mask = disk_mask(H, W, cy, cx, float(r_t[ti]))
            pond_vals = rng.normal(POND_MEAN, POND_SIGMA, size=(H, W)).astype(np.float32)
            bg[mask] = pond_vals[mask]
        VV[ti] = clamp(bg, VV_MIN, VV_MAX)

    meta = SimMeta(
        scenario="single_transient",
        flood_day=flood_day,
        width=width,
        ponds=[{"cy": cy, "cx": cx, "r_peak": float(r_peak), "kind": "transient"}],
        onset_day=onset_day,
        offset_day=offset_day
    )
    return VV, meta

def run_k_smoothing_limits_experiment():
    safe_mkdir(OUT_DIR_KTEST)
    rng = np.random.default_rng(MASTER_SEED + 999)

    k_list_capped = [cap_odd_k(k, H, W) for k in K_LIST]
    k_list_capped = list(dict.fromkeys(k_list_capped))

    mean_detect = np.zeros((len(R_PEAK_GRID), len(k_list_capped)), dtype=np.float32)
    std_detect = np.zeros((len(R_PEAK_GRID), len(k_list_capped)), dtype=np.float32)

    for ri, r_peak in enumerate(R_PEAK_GRID):
        det_by_k = [[] for _ in k_list_capped]
        for _ in range(N_SIMS_PER_RADIUS):
            VV_raw, meta = gen_single_transient_with_fixed_rpeak(rng, r_peak=float(r_peak))
            for ki, k in enumerate(k_list_capped):
                VV_s = apply_spatial_smoothing_stack(VV_raw, k)
                _, _, pct_flood = compute_metrics(VV_s)
                det_by_k[ki].append(detection_summary(pct_flood, meta.flood_day))

        for ki in range(len(k_list_capped)):
            arr = np.array(det_by_k[ki], dtype=np.float32)
            mean_detect[ri, ki] = float(arr.mean())
            std_detect[ri, ki] = float(arr.std())

        if (ri + 1) % 10 == 0:
            print(f"[k-test] progress {ri+1}/{len(R_PEAK_GRID)}")

    np.savez_compressed(
        os.path.join(OUT_DIR_KTEST, "k_limit_test_results.npz"),
        mean_detect=mean_detect,
        std_detect=std_detect,
        r_peak_grid=R_PEAK_GRID.astype(np.float32),
        k_list=np.array(k_list_capped, dtype=np.int32),
    )
    print(f"[k-test] Saved: {os.path.join(OUT_DIR_KTEST, 'k_limit_test_results.npz')}")

    plt.figure(figsize=(11, 6))
    plt.imshow(mean_detect, aspect="auto", origin="lower")
    plt.colorbar(label="Mean peak % flooded")
    plt.xticks(np.arange(len(k_list_capped)), [str(k) for k in k_list_capped], rotation=45, ha="right")
    plt.yticks(np.arange(len(R_PEAK_GRID)), [f"{r:g}" for r in R_PEAK_GRID])
    plt.xlabel("k")
    plt.ylabel("r_peak")
    plt.title("Detectability vs smoothing")
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================
def main():
    safe_mkdir(OUT_DIR)

    if RUN_MODE == "k_test_only":
        run_k_smoothing_limits_experiment()
        return

    rng = np.random.default_rng(MASTER_SEED)

    # =====================================================
    # PARAMETER SWEEP (smooth_k, kappa, h)
    # =====================================================
    print("\n=== Running FULL parameter sweep over smoothing + (kappa,h) ===")
    df_sweep, best = run_param_sweep_over_smoothing(rng)

    if len(df_sweep) == 0:
        raise RuntimeError("Parameter sweep produced no results.")

    df_sweep.to_csv(OUT_PARAM_SWEEP, index=False)
    print(f"\nSaved sweep table: {OUT_PARAM_SWEEP}")
    print(df_sweep.head(12).to_string(index=False))

    pd.DataFrame([best]).to_csv(OUT_PARAM_BEST, index=False)
    print(f"\nBest params saved: {OUT_PARAM_BEST}")
    print(pd.DataFrame([best]).to_string(index=False))

    best_smooth_k = int(best["smooth_k"])
    best_kappa = float(best["kappa"])
    best_h = float(best["h"])

    # =====================================================
    # FINAL TABLES using best params (re-run scenarios at best smooth_k)
    # =====================================================
    print("\n=== Re-running scenarios at BEST smooth_k for final per-sim + ARL tables ===")
    rng_final = np.random.default_rng(MASTER_SEED + 4242)  # separate stream for final reporting
    all_results_best = run_all_scenarios_for_smooth_k(rng_final, smooth_k=best_smooth_k)

    df_per_sim = run_cusum_on_results(all_results_best, burn_in=CUSUM_BURN_IN, kappa=best_kappa, h=best_h)
    df_per_sim["smooth_k"] = best_smooth_k
    df_per_sim["kappa"] = best_kappa
    df_per_sim["h"] = best_h
    df_per_sim.to_csv(OUT_CUSUM_PER_SIM, index=False)
    print(f"\nSaved per-sim CUSUM table: {OUT_CUSUM_PER_SIM}")
    print(df_per_sim.head(12).to_string(index=False))

    df_arl = make_arl_summary_table(df_per_sim.drop(columns=["smooth_k","kappa","h"], errors="ignore"), burn_in=CUSUM_BURN_IN)
    df_arl.insert(0, "smooth_k", best_smooth_k)
    df_arl.insert(1, "kappa", best_kappa)
    df_arl.insert(2, "h", best_h)
    df_arl.to_csv(OUT_ARL_TABLE, index=False)
    print(f"\nSaved ARL table: {OUT_ARL_TABLE}")
    print(df_arl.to_string(index=False))

    if RUN_MODE == "main_then_k" or KTEST_ENABLE:
        run_k_smoothing_limits_experiment()

    print("\nDone.")


if __name__ == "__main__":
    main()