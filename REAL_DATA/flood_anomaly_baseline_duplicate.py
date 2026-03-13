import re
import csv
import os
import json
import numpy as np
import rasterio
from rasterio.features import shapes
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any

#SETTINGS
base_tif_path  = "/Users/lunarolland/Desktop/DATASETS/Spain_7370579_S1_VV_VH_.tif"
extra_tif_path = "/Users/lunarolland/Desktop/DATASETS/Spain_7370579_S1_EXTRA_DESC_anyRelOrbit_20190911_20190929.tif"

# Outside focus window
PLATFORM_BASE = "S1A"  # "S1A" or "S1B"

# Inside focus window
PLATFORMS_EXTRA = ("S1A", "S1B")

# Flood center date 
flood_center_date = date(2019, 9, 17)

# Time window 
start_date = flood_center_date - timedelta(days=365)
end_date   = flood_center_date + timedelta(days=365)

# Baseline window
ref_start = flood_center_date - timedelta(days=365)
ref_end   = flood_center_date - timedelta(days=1)

# Focus window: ONLY use EXTRA here 
focus_start = date(2019, 9, 11)
focus_end   = date(2019, 9, 29)

# Spatial smoothing scale 
k = 3  

# Clamp ranges (dB)
VV_MIN, VV_MAX = -25.0, 0.0
VH_MIN, VH_MAX = -35.0, -5.0

# Flood score mix 
W_VV, W_VH = 0.7, 0.3

BASELINE_Q = 99.9

# Output
out_dir = "/Users/lunarolland/Desktop/flood_anomaly_outputs"
SAVE_SCORE_TIFS = True
SAVE_MASK_TIFS = True
SHOW_PLOTS = True

NODATA_F = -9999.0

# DARK% 
DARK_VV_LO = -25.0
DARK_VV_HI = -17.0

SAVE_DARK_MASK_TIFS = True
SAVE_PNG_QUICKLOOKS = True
PNG_DPI = 150

#CUSUM PARAMETERS
CUSUM_K = 1.25
CUSUM_H = 12.0  

#DUPLICATED WINDOW
DUPLICATE_FOCUS_DAILY = True

FOCUS_DUPLICATION_MAP = {
    date(2019, 9, 11): date(2019, 9, 11),
    date(2019, 9, 12): date(2019, 9, 28),
    date(2019, 9, 13): date(2019, 9, 23),
    date(2019, 9, 14): date(2019, 9, 22),
    date(2019, 9, 15): date(2019, 9, 17),
    date(2019, 9, 16): date(2019, 9, 16),
    date(2019, 9, 17): date(2019, 9, 17),
    date(2019, 9, 22): date(2019, 9, 22),
    date(2019, 9, 23): date(2019, 9, 23),
    date(2019, 9, 28): date(2019, 9, 28),
    date(2019, 9, 29): date(2019, 9, 29),
}

#HELPERS
def parse_band(desc: str) -> Optional[dict]:
    """
    Parse band description to extract platform, timestamp, and polarization.

    Supports BOTH naming styles:
      - Old:  "S1A_20190917T053012_VV"
      - New:  "VV_20190917" or "VH_20190917"
    """
    if not desc:
        return None

    d = desc.upper().strip()

    pol = None
    if d.startswith("VV_") or d.endswith("_VV"):
        pol = "VV"
    elif d.startswith("VH_") or d.endswith("_VH"):
        pol = "VH"

    plat = None
    if d.startswith("S1A_"):
        plat = "S1A"
    elif d.startswith("S1B_"):
        plat = "S1B"

    m1 = re.search(r"(20\d{6}T\d{6})", d)
    if m1:
        dt = datetime.strptime(m1.group(1), "%Y%m%dT%H%M%S")
        return {"platform": plat, "pol": pol, "dt": dt, "desc": desc}

    m2 = re.search(r"(20\d{6})$", d)
    if m2 and (d.startswith("VV_") or d.startswith("VH_")):
        dt = datetime.strptime(m2.group(1), "%Y%m%d")
        return {"platform": plat, "pol": pol, "dt": dt, "desc": desc}

    return {"platform": plat, "pol": pol, "dt": None, "desc": desc}


def clamp(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    a = arr.astype(np.float32)
    a = np.where(np.isfinite(a), a, np.nan)
    return np.clip(a, vmin, vmax)


def box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Fast k×k box mean with reflect padding using integral image."""
    if k % 2 == 0:
        raise ValueError("k must be odd")
    if k == 1:
        return arr.astype(np.float32)

    pad = k // 2
    a = arr.astype(np.float32)
    valid = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)

    a0p = np.pad(a0, pad, mode="reflect")
    vp  = np.pad(valid, pad, mode="reflect")

    S  = a0p.cumsum(0).cumsum(1)
    SV = vp.cumsum(0).cumsum(1)

    H, W = a.shape
    x2 = np.arange(k - 1, k - 1 + H)
    y2 = np.arange(k - 1, k - 1 + W)

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


def find_dates_available(bands: List[dict]) -> List[date]:
    """Return unique available DATES (not datetimes) to avoid timestamp mismatch issues."""
    return sorted({b["dt"].date() for b in bands if b["dt"] is not None})


def get_vh_vv_for_date(src, parsed_bands: List[dict], target_date: date) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads VH and VV for target_date.
    If multiple bands match (e.g., S1A + S1B), it averages them.
    Matching is by DATE (dt.date()).
    """
    def read_band(bidx):
        arr = src.read(bidx).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr

    out = {}
    for pol in ("VH", "VV"):
        matches = [b for b in parsed_bands if b["pol"] == pol and b["dt"] is not None and b["dt"].date() == target_date]
        if not matches:
            raise RuntimeError(f"No {pol} band found for date {target_date}.")

        arrs = [read_band(b["band"]) for b in matches]
        arr = np.nanmean(np.stack(arrs, axis=0), axis=0).astype(np.float32)

        if pol == "VV":
            arr = clamp(arr, VV_MIN, VV_MAX)
        else:
            arr = clamp(arr, VH_MIN, VH_MAX)

        out[pol] = arr

    return out["VH"], out["VV"]


def robust_baseline(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """stack: (T,H,W) returns: median (H,W), IQR (H,W)"""
    med = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    iqr = q75 - q25
    return med.astype(np.float32), iqr.astype(np.float32)


def flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr, eps=1e-3):
    z_vv = (VV_t - VV_med) / (VV_iqr + eps)
    z_vh = (VH_t - VH_med) / (VH_iqr + eps)
    score_vv = -z_vv
    score_vh = -z_vh
    score = W_VV * score_vv + W_VH * score_vh
    return score.astype(np.float32), score_vv.astype(np.float32), score_vh.astype(np.float32)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def write_geotiff(path: str, arr: np.ndarray, profile: dict, dtype, nodata=None):
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def in_focus_window(d: date) -> bool:
    return (focus_start <= d <= focus_end)


def build_focus_virtual_dates() -> List[date]:
    return sorted(FOCUS_DUPLICATION_MAP.keys())


def source_date_for_virtual(d: date) -> date:
    return FOCUS_DUPLICATION_MAP.get(d, d)

def dark_mask_from_vv(VV_t: np.ndarray) -> np.ndarray:
    """Dark-water proxy mask: VV in [DARK_VV_LO, DARK_VV_HI] (finite only)."""
    m = np.isfinite(VV_t) & (VV_t >= DARK_VV_LO) & (VV_t <= DARK_VV_HI)
    return m.astype(np.uint8)


def percent_from_mask(mask_u8: np.ndarray, finite_ref: np.ndarray) -> float:
    """Percent True within finite_ref pixels."""
    if finite_ref.any():
        return float(mask_u8[finite_ref].mean()) * 100.0
    return 0.0


def save_mask_png(path: str, mask_u8: np.ndarray, title: str):
    """Save a quicklook PNG of a uint8 mask (0/1)."""
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_u8, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=PNG_DPI)
    plt.close()


def cusum_upper(z: np.ndarray, K: float) -> np.ndarray:
    """One-sided upper CUSUM on standardized series z."""
    C = np.zeros_like(z, dtype=np.float32)
    c = 0.0
    for i in range(len(z)):
        zi = float(z[i])
        if not np.isfinite(zi):
            C[i] = c
            continue
        c = max(0.0, c + zi - K)
        C[i] = c
    return C


def first_alarm(C: np.ndarray, dates: List[date], h: float) -> Tuple[Optional[date], Optional[float], Optional[int]]:
    for i, c in enumerate(C):
        if np.isfinite(c) and c >= h:
            return dates[i], float(c), i
    return None, None, None


def mask_to_polygons_geojson(mask_u8: np.ndarray, transform, min_pixels: int = 50) -> Dict[str, Any]:
    """
    Vectorize a binary mask into GeoJSON polygons (in map coords).
    - min_pixels filters tiny blobs.
    """
    feats = []
    for geom, val in shapes(mask_u8, mask=(mask_u8 == 1), transform=transform):
        feats.append({"type": "Feature", "properties": {"value": int(val)}, "geometry": geom})

    return {"type": "FeatureCollection", "features": feats}

from skimage import measure

def plot_mask_with_outlines(mask_u8: np.ndarray, vv_backdrop: Optional[np.ndarray], title: str):
    """
    Plot VV image with RED polygon outlines of mask (no fill).
    """

    plt.figure(figsize=(8, 8))

    if vv_backdrop is not None:
        plt.imshow(vv_backdrop, vmin=VV_MIN, vmax=VV_MAX, cmap="gray")
    else:
        plt.imshow(mask_u8, cmap="gray")

    contours = measure.find_contours(mask_u8, 0.5)

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color="red")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


#MAIN
def main():
    safe_mkdir(out_dir)

    with rasterio.open(base_tif_path) as base_src, rasterio.open(extra_tif_path) as extra_src:
        if (base_src.width != extra_src.width or base_src.height != extra_src.height or
            base_src.transform != extra_src.transform or base_src.crs != extra_src.crs):
            raise RuntimeError(
                "BASE and EXTRA GeoTIFFs are not on the same grid/CRS/transform. "
                "Export them with identical region/scale/crs so they align."
            )

        profile = base_src.profile
        transform = base_src.transform

        base_parsed = []
        for band_i, desc in enumerate(base_src.descriptions, start=1):
            info = parse_band(desc)
            if info is None:
                continue
            info["band"] = band_i
            base_parsed.append(info)

        base_parsed = [
            b for b in base_parsed
            if b["pol"] in ("VV", "VH") and b["dt"] is not None and b["platform"] == PLATFORM_BASE
        ]
        if not base_parsed:
            raise RuntimeError(f"No BASE bands found for platform {PLATFORM_BASE} with parsable dates.")

        extra_parsed = []
        for band_i, desc in enumerate(extra_src.descriptions, start=1):
            info = parse_band(desc)
            if info is None:
                continue
            info["band"] = band_i
            extra_parsed.append(info)

        extra_parsed = [
            b for b in extra_parsed
            if b["pol"] in ("VV", "VH") and b["dt"] is not None and
               (b["platform"] in PLATFORMS_EXTRA or b["platform"] is None)
        ]

        base_dates = find_dates_available(base_parsed)
        extra_dates = find_dates_available(extra_parsed)

        # Outside focus
        base_dates_window = [
            d for d in base_dates
            if start_date <= d <= end_date and not in_focus_window(d)
        ]

        # Inside focus
        if DUPLICATE_FOCUS_DAILY:
            focus_virtual_dates = [
                d for d in build_focus_virtual_dates()
                if start_date <= d <= end_date and in_focus_window(d)
            ]
            focus_source_dates = {source_date_for_virtual(d) for d in focus_virtual_dates}

            missing = sorted([sd for sd in focus_source_dates if sd not in extra_dates])
            if missing:
                raise RuntimeError(
                    "Some source dates required by FOCUS_DUPLICATION_MAP are not present in EXTRA GeoTIFF:\n"
                    + "\n".join([str(x) for x in missing])
                )

            focus_dates_window = focus_virtual_dates
        else:
            focus_dates_window = [
                d for d in extra_dates
                if start_date <= d <= end_date and in_focus_window(d)
            ]

        dates_window = sorted(base_dates_window + focus_dates_window)

        # Baseline dates:
        dates_ref = [d for d in base_dates if ref_start <= d <= ref_end]

        print(f"BASE acquisitions total ({PLATFORM_BASE}): {len(base_dates)}")
        print(f"EXTRA acquisitions total (focus stack): {len(extra_dates)}")
        print(f"Window dates total (BASE outside focus + EXTRA inside focus): {len(dates_window)}")
        print(f"  - BASE outside-focus dates: {len(base_dates_window)}")
        print(f"  - Focus dates used:         {len(focus_dates_window)} (DUPLICATE_FOCUS_DAILY={DUPLICATE_FOCUS_DAILY})")
        print(f"Baseline acquisitions (BASE only) ({ref_start}..{ref_end}): {len(dates_ref)}")

        if len(dates_ref) < 5:
            print("WARNING: Very few baseline dates. Baseline may be unstable.")

        #Baseline stacks
        print(f"\nBuilding per-pixel baseline using k={k} (BASE only) ...")
        VV_stack = []
        VH_stack = []
        for d in dates_ref:
            VH, VV = get_vh_vv_for_date(base_src, base_parsed, d)
            if k > 1:
                VH = box_mean(VH, k)
                VV = box_mean(VV, k)
            VV_stack.append(VV)
            VH_stack.append(VH)

        VV_stack = np.stack(VV_stack, axis=0)
        VH_stack = np.stack(VH_stack, axis=0)

        VV_med, VV_iqr = robust_baseline(VV_stack)
        VH_med, VH_iqr = robust_baseline(VH_stack)

        # Baseline threshold
        print(f"\nCalibrating global threshold from baseline at {BASELINE_Q}th percentile ...")
        baseline_scores = []
        for i, d in enumerate(dates_ref):
            VH_t = VH_stack[i]
            VV_t = VV_stack[i]
            score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
            baseline_scores.append(score[np.isfinite(score)])

        baseline_scores = np.concatenate(baseline_scores) if baseline_scores else np.array([], dtype=np.float32)
        if baseline_scores.size == 0:
            raise RuntimeError("No finite baseline scores; check nodata handling / clamps.")

        THR_SCORE = float(np.percentile(baseline_scores, BASELINE_Q))
        print(f"THR_SCORE (from BASE baseline) = {THR_SCORE:.3f}\n")

        # Baseline time series
        print("Computing baseline dark% and flood% series (BASE only)...")
        dark_pct_ref = []
        flood_pct_ref = []

        for i, d in enumerate(dates_ref):
            VH_t = VH_stack[i]
            VV_t = VV_stack[i]
            score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)

            flood_u8 = (score >= THR_SCORE).astype(np.uint8)
            flood_pct_ref.append(percent_from_mask(flood_u8, np.isfinite(score)))

            dark_u8 = dark_mask_from_vv(VV_t)
            dark_pct_ref.append(percent_from_mask(dark_u8, np.isfinite(VV_t)))

        dark_pct_ref = np.array(dark_pct_ref, dtype=np.float32)
        flood_pct_ref = np.array(flood_pct_ref, dtype=np.float32)

        dark_mu0  = float(np.nanmean(dark_pct_ref))
        dark_sd0  = float(np.nanstd(dark_pct_ref) + 1e-6)
        flood_mu0 = float(np.nanmean(flood_pct_ref))
        flood_sd0 = float(np.nanstd(flood_pct_ref) + 1e-6)

        print(f"Baseline dark%:  mean={dark_mu0:.4f}, std={dark_sd0:.4f}")
        print(f"Baseline flood%: mean={flood_mu0:.4f}, std={flood_sd0:.4f}\n")

        #Global plot
        if SHOW_PLOTS:
            print("Computing global plot scaling for score (consistent across dates)...")
            all_scores = []
            for d_virtual in dates_window:
                if in_focus_window(d_virtual):
                    d_source = source_date_for_virtual(d_virtual) if DUPLICATE_FOCUS_DAILY else d_virtual
                    VH_t, VV_t = get_vh_vv_for_date(extra_src, extra_parsed, d_source)
                else:
                    VH_t, VV_t = get_vh_vv_for_date(base_src, base_parsed, d_virtual)

                if k > 1:
                    VH_t = box_mean(VH_t, k)
                    VV_t = box_mean(VV_t, k)

                score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
                all_scores.append(score)

            S = np.stack(all_scores, axis=0)
            PLOT_VMIN = float(np.nanpercentile(S, 2))
            PLOT_VMAX = float(np.nanpercentile(S, 98))
            print(f"PLOT_VMIN={PLOT_VMIN:.3f}, PLOT_VMAX={PLOT_VMAX:.3f}\n")

        print("Scoring dates and writing outputs...")
        for d_virtual in dates_window:
            if in_focus_window(d_virtual):
                src_tag = "EXTRA"
                d_source = source_date_for_virtual(d_virtual) if DUPLICATE_FOCUS_DAILY else d_virtual
                src_use, parsed_use = extra_src, extra_parsed
            else:
                src_tag = "BASE"
                d_source = d_virtual
                src_use, parsed_use = base_src, base_parsed

            VH_t, VV_t = get_vh_vv_for_date(src_use, parsed_use, d_source)

            if k > 1:
                VH_t = box_mean(VH_t, k)
                VV_t = box_mean(VV_t, k)

            score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
            flood_u8 = (score >= THR_SCORE).astype(np.uint8)
            dark_u8 = dark_mask_from_vv(VV_t)

            flood_pct = percent_from_mask(flood_u8, np.isfinite(score))
            dark_pct = percent_from_mask(dark_u8, np.isfinite(VV_t))

            vtag = d_virtual.strftime("%Y%m%d")
            stag = d_source.strftime("%Y%m%d")
            dup_tag = f"_from_{stag}" if (d_virtual != d_source) else ""

            print(f"{d_virtual} [{src_tag}] (pixels={d_source}) flood%={flood_pct:.3f}%  dark%={dark_pct:.3f}%")

            if SAVE_SCORE_TIFS:
                score_out = np.where(np.isfinite(score), score, NODATA_F).astype(np.float32)
                write_geotiff(
                    os.path.join(out_dir, f"score_{src_tag}_{vtag}{dup_tag}_k{k}.tif"),
                    score_out, profile, dtype="float32", nodata=NODATA_F
                )

            if SAVE_MASK_TIFS:
                write_geotiff(
                    os.path.join(out_dir, f"mask_{src_tag}_{vtag}{dup_tag}_k{k}.tif"),
                    flood_u8, profile, dtype="uint8", nodata=0
                )

            if SAVE_DARK_MASK_TIFS:
                write_geotiff(
                    os.path.join(out_dir, f"darkmask_{src_tag}_{vtag}{dup_tag}_k{k}.tif"),
                    dark_u8, profile, dtype="uint8", nodata=0
                )

            if SAVE_PNG_QUICKLOOKS:
                save_mask_png(
                    os.path.join(out_dir, f"mask_{src_tag}_{vtag}{dup_tag}_k{k}.png"),
                    flood_u8,
                    f"Flood mask {src_tag} {d_virtual} (pixels {d_source})\nthreshold={THR_SCORE:.3f}"
                )
                save_mask_png(
                    os.path.join(out_dir, f"darkmask_{src_tag}_{vtag}{dup_tag}_k{k}.png"),
                    dark_u8,
                    f"Dark mask {src_tag} {d_virtual} (pixels {d_source})\nVV in [{DARK_VV_LO},{DARK_VV_HI}] dB"
                )

            if SHOW_PLOTS:
                plt.figure(figsize=(16, 4))

                plt.subplot(1, 4, 1)
                plt.imshow(score, cmap="magma", vmin=PLOT_VMIN, vmax=PLOT_VMAX)
                title = f"Flood score ({src_tag})\n{d_virtual}"
                if d_virtual != d_source:
                    title += f"\n(pixels from {d_source})"
                plt.title(title)
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(flood_u8, cmap="gray", vmin=0, vmax=1)
                plt.title(f"Flood mask\n(flood%={flood_pct:.2f}%)")
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.imshow(dark_u8, cmap="gray", vmin=0, vmax=1)
                plt.title(f"Dark mask VV[{DARK_VV_LO},{DARK_VV_HI}]\n(dark%={dark_pct:.2f}%)")
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.imshow(VV_t, vmin=VV_MIN, vmax=VV_MAX, cmap="gray")
                plt.title("VV (processed)")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

        #TIME SERIES
        print("\nBuilding flood% and dark% time series...")

        dates_ts = []
        flooded_fraction_ts = []
        dark_fraction_ts = []
        def compute_products_for_virtual_date(d_virtual: date):
            if in_focus_window(d_virtual):
                d_source = source_date_for_virtual(d_virtual) if DUPLICATE_FOCUS_DAILY else d_virtual
                src_use, parsed_use, src_tag = extra_src, extra_parsed, "EXTRA"
            else:
                d_source = d_virtual
                src_use, parsed_use, src_tag = base_src, base_parsed, "BASE"

            VH_t, VV_t = get_vh_vv_for_date(src_use, parsed_use, d_source)
            if k > 1:
                VH_t = box_mean(VH_t, k)
                VV_t = box_mean(VV_t, k)

            score, _, _ = flood_score(VH_t, VV_t, VH_med, VH_iqr, VV_med, VV_iqr)
            flood_u8 = (score >= THR_SCORE).astype(np.uint8)
            dark_u8  = dark_mask_from_vv(VV_t)

            flood_pct = percent_from_mask(flood_u8, np.isfinite(score))
            dark_pct  = percent_from_mask(dark_u8, np.isfinite(VV_t))

            return {
                "src_tag": src_tag,
                "d_source": d_source,
                "score": score,
                "flood_u8": flood_u8,
                "dark_u8": dark_u8,
                "VV_t": VV_t,
                "flood_pct": flood_pct,
                "dark_pct": dark_pct,
            }

        for d_virtual in dates_window:
            prod = compute_products_for_virtual_date(d_virtual)
            dates_ts.append(d_virtual)
            flooded_fraction_ts.append(prod["flood_pct"])
            dark_fraction_ts.append(prod["dark_pct"])

        if SHOW_PLOTS and len(dates_ts) > 0:
            # Flood% curve
            plt.figure(figsize=(10, 5))
            plt.plot(dates_ts, flooded_fraction_ts, marker='o')
            plt.axvline(flood_center_date, linestyle='--')
            plt.title("Flooded Area Fraction (%)")
            plt.ylabel("Flooded Pixels (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Dark% curve
            plt.figure(figsize=(10, 5))
            plt.plot(dates_ts, dark_fraction_ts, marker='o')
            plt.axvline(flood_center_date, linestyle='--')
            plt.title(f"Dark Area Fraction (%)  (VV in {DARK_VV_LO}..{DARK_VV_HI} dB)")
            plt.ylabel("Dark Pixels (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        #CUSUM
        print("\nCUSUM detection (no ARL0 sims)...")

        dark_ts  = np.array(dark_fraction_ts, dtype=np.float32)
        flood_ts = np.array(flooded_fraction_ts, dtype=np.float32)

        z_dark  = (dark_ts  - dark_mu0)  / dark_sd0
        z_flood = (flood_ts - flood_mu0) / flood_sd0

        C_dark  = cusum_upper(z_dark,  CUSUM_K)
        C_flood = cusum_upper(z_flood, CUSUM_K)

        alarm_dark_date, alarm_dark_val, alarm_dark_idx = first_alarm(C_dark, dates_ts, CUSUM_H)
        alarm_flood_date, alarm_flood_val, alarm_flood_idx = first_alarm(C_flood, dates_ts, CUSUM_H)

        print(f"\nCUSUM parameters: k={k}, κ={CUSUM_K}, h={CUSUM_H}")
        print(f"Dark% alarm:  {alarm_dark_date} (CUSUM={alarm_dark_val})")
        print(f"Flood% alarm: {alarm_flood_date} (CUSUM={alarm_flood_val})")

        if SHOW_PLOTS and len(dates_ts) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(dates_ts, C_dark, marker='o')
            plt.axhline(CUSUM_H, linestyle='--')
            plt.axvline(flood_center_date, linestyle='--')
            plt.title("CUSUM (upper) on dark% (standardized)")
            plt.ylabel("CUSUM")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(dates_ts, C_flood, marker='o')
            plt.axhline(CUSUM_H, linestyle='--')
            plt.axvline(flood_center_date, linestyle='--')
            plt.title("CUSUM (upper) on flood% (standardized)")
            plt.ylabel("CUSUM")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # VISUALISATION AT ALARM DATE
        def dump_alarm_products(kind: str, alarm_date: Optional[date]):
            """
            kind in {"dark", "flood"}
            Shows the mask map at the alarm date (and overlay), and writes a GeoJSON polygon outline.
            """
            if alarm_date is None:
                print(f"\nNo {kind} alarm (CUSUM never crossed h).")
                return

            prod = compute_products_for_virtual_date(alarm_date)
            src_tag = prod["src_tag"]
            d_source = prod["d_source"]

            vtag = alarm_date.strftime("%Y%m%d")
            stag = d_source.strftime("%Y%m%d")
            dup_tag = f"_from_{stag}" if (alarm_date != d_source) else ""

            if kind == "dark":
                mask_u8 = prod["dark_u8"]
                title = f"DARK ALARM {alarm_date} [{src_tag}] (pixels {d_source})"
                base_name = f"ALARM_dark_{src_tag}_{vtag}{dup_tag}_k{k}"
            else:
                mask_u8 = prod["flood_u8"]
                title = f"FLOOD ALARM {alarm_date} [{src_tag}] (pixels {d_source})"
                base_name = f"ALARM_flood_{src_tag}_{vtag}{dup_tag}_k{k}"

            if SHOW_PLOTS:
                plot_mask_with_outlines(mask_u8, prod["VV_t"], title)

            if SAVE_PNG_QUICKLOOKS:
                save_mask_png(
                    os.path.join(out_dir, f"{base_name}.png"),
                    mask_u8,
                    title
                )

            gj = mask_to_polygons_geojson(mask_u8, transform=transform)
            geojson_path = os.path.join(out_dir, f"{base_name}.geojson")
            with open(geojson_path, "w") as f:
                json.dump(gj, f)

            print(f"\n{kind.upper()} alarm products:")
            print(f"  date (virtual): {alarm_date}")
            print(f"  pixels from:    {d_source} ({src_tag})")
            print(f"  mask pixels=1:  {int(mask_u8.sum())}")
            print(f"  wrote polygons: {geojson_path}")
            if SAVE_PNG_QUICKLOOKS:
                print(f"  wrote PNG:      {os.path.join(out_dir, f'{base_name}.png')}")

        dump_alarm_products("dark", alarm_dark_date)
        dump_alarm_products("flood", alarm_flood_date)

    print("\nDone.")
    print("Regular outputs: mask_*.tif + mask_*.png, darkmask_*.tif + darkmask_*.png")
    print("Alarm outputs:   ALARM_*_*.png and ALARM_*_*.geojson")


if __name__ == "__main__":
    main()
