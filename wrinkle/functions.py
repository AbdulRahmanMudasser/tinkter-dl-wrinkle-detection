# import subprocess as sp
from docx import Document
import os
import tkinter as tk
from tkinter import messagebox
import globals as vars
import LJXAwrap
import queue
import ctypes
import sys
from multiprocessing import Process, Value, Array, Queue, Pipe, Manager, current_process
import numpy as np
import threading
import pandas as pd
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pytz
from influxdb_client.client.flux_table import FluxTable
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import savgol_filter
import random
import time
import customtkinter as ctk
import wrinkle_aux_funcs as wrinkle_aux

from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from configparser import ConfigParser
# from wrinkle_detection_new import frangi_core
# at the top with the other imports
import globals as vars
import numpy as np
import pandas as pd


# ---- Defaults so PyCharm knows these names exist (override later from config.ini) ----
WRINKLE_PIPELINE: str = "GABOR45"   # or "FRANGI"
TH_PERCENTILE: float = 96.0
GABOR_SIGMA: float = 2.0
GABOR_FREQ: float = 0.08

WR_DIAG_CENTER_DEG: float = 45.0
WR_DIAG_TOL_DEG: float = 12.0
WR_EDGE_MARGIN_PX: int = 10
WR_MIN_LEN_PX: int = 16
WR_AR_MIN: float = 2.0


import configparser

# --- load config.ini once ---
config = configparser.ConfigParser()
config.read("config.ini")



def _load_heightmap_table(path):
    import pandas as pd, numpy as np

    # try German CSV
    try:
        df = pd.read_csv(path, decimal=',', header=None, sep=';')
    except Exception:
        df = pd.read_csv(path, decimal='.', header=None, sep=',')

    # retry with standard CSV if first cell looks non-numeric
    if isinstance(df.iloc[0, 0], str):
        try:
            df = pd.read_csv(path, decimal='.', header=None, sep=',')
        except Exception:
            pass

    df = df.apply(pd.to_numeric, errors="coerce")

    # ---- recommended guards ----
    if df.isna().all().all():
        raise ValueError("Loaded table contains only NaN after numeric coercion. Check separator/decimal.")

    # (optional) Fill NaNs so filters/imshow never fail; comment out if you prefer raw NaNs.
    arr = df.to_numpy()
    finite = np.isfinite(arr)
    med = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    arr = np.where(finite, arr, med)
    return pd.DataFrame(arr)




# --- Back-compat shims (keep callers working) ---
def detect_wrinkles_tophat_thesis(file_path, y_lower=None, y_upper=None):
    """Back-compat shim that always returns (df, df_props, {'mask':..., 'skeleton':...})."""
    import numpy as np
    try:
        from skimage.morphology import skeletonize
    except Exception:
        skeletonize = None

    res = detect_wrinkles_pipeline(file_path, y_lower, y_upper)

    if isinstance(res, tuple) and len(res) == 4:
        df, df_props, mask, skel = res
    elif isinstance(res, tuple) and len(res) == 3:
        df, df_props, mask = res
        # create a skeleton if the pipeline didn’t return one
        if skeletonize is not None:
            try:
                skel = skeletonize(np.asarray(mask).astype(bool))
            except Exception:
                skel = np.asarray(mask).astype(bool)
        else:
            skel = np.asarray(mask).astype(bool)
    else:
        raise ValueError(f"Unexpected return from detect_wrinkles_pipeline: {type(res)}")

    return df, df_props, {"mask": mask, "skeleton": skel}


def _read_csv_robust_thesis(file_path):
    return _read_heightmap_table(file_path)


# === WRINKLE PIPELINE CONSTANTS (module-level) ===
# Read from [thresholds] in config.ini, with robust fallbacks.
# ---- compact config override for [thresholds] ----
def _cfg_or_existing(name, getter, keys, default):
    """
    If global NAME already exists, keep it.
    Else, read the first existing key from `keys` (list of alternatives) in [thresholds],
    cast via the provided `getter` (e.g., config.getfloat), else use default.
    """
    if name in globals():
        return globals()[name]
    for key in keys:
        if config.has_option('thresholds', key):
            return getter('thresholds', key, fallback=default)
    return default

WRINKLE_PIPELINE = _cfg_or_existing(
    'WRINKLE_PIPELINE',
    lambda s,k,**kw: config.get(s, k, **kw).upper(),
    ['WRINKLE_PIPELINE', 'wrinkle_pipeline'],
    'GABOR45'
)

TH_PERCENTILE = _cfg_or_existing(
    'TH_PERCENTILE', config.getfloat,
    ['TH_PERCENTILE', 'th_percentile'],
    96.0
)

GABOR_SIGMA = _cfg_or_existing(
    'GABOR_SIGMA', config.getfloat,
    ['GABOR_SIGMA', 'gabor_sigma'],
    2.0
)

GABOR_FREQ = _cfg_or_existing(
    'GABOR_FREQ', config.getfloat,
    ['GABOR_FREQ', 'gabor_freq'],
    0.08
)

WR_DIAG_CENTER_DEG = _cfg_or_existing(
    'WR_DIAG_CENTER_DEG', config.getfloat,
    ['WR_DIAG_CENTER_DEG', 'wr_diag_center_deg'],
    45.0
)

WR_DIAG_TOL_DEG = _cfg_or_existing(
    'WR_DIAG_TOL_DEG', config.getfloat,
    ['WR_DIAG_TOL_DEG', 'wr_diag_tol_deg'],
    12.0
)

WR_EDGE_MARGIN_PX = _cfg_or_existing(
    'WR_EDGE_MARGIN_PX', config.getint,
    ['WR_EDGE_MARGIN_PX', 'wr_edge_margin_px'],
    10
)

WR_MIN_LEN_PX = _cfg_or_existing(
    'WR_MIN_LEN_PX', config.getint,
    ['WR_MIN_LEN_PX', 'wr_min_len_px'],
    16
)

WR_AR_MIN = _cfg_or_existing(
    'WR_AR_MIN', config.getfloat,
    ['WR_AR_MIN', 'wr_ar_min'],
    2.0
)

# ===== Canonical thresholds (single source of truth) =====

DEFAULTS = dict(
    WRINKLE_PIPELINE='GABOR45',
    TH_PERCENTILE=96.0,
    GABOR_SIGMA=2.0,
    GABOR_FREQ=0.08,

    WR_DIAG_CENTER_DEG=45.0,
    WR_DIAG_TOL_DEG=12.0,
    WR_EDGE_MARGIN_PX=10,
    WR_MIN_LEN_PX=16,
    WR_AR_MIN=2.0,
    WR_MAX_ANGLE_DEG=45.0,

    LF_MIN_LEN_PX=60,
    LF_AR_MIN=2.5,
    LF_MAX_ANGLE_DEG=20.0,
    LF_EDGE_MARGIN=0,

    ECC_MIN=0.95,
    VESSEL_MIN=0.0,
)

# Read thresholds from [thresholds] (case-insensitive), with sensible fallbacks.
WRINKLE_PIPELINE = config.get('thresholds', 'WRINKLE_PIPELINE',
                              fallback=DEFAULTS['WRINKLE_PIPELINE']).upper()
TH_PERCENTILE    = config.getfloat('thresholds', 'TH_PERCENTILE',
                                   fallback=DEFAULTS['TH_PERCENTILE'])
GABOR_SIGMA      = config.getfloat('thresholds', 'GABOR_SIGMA',
                                   fallback=DEFAULTS['GABOR_SIGMA'])
GABOR_FREQ       = config.getfloat('thresholds', 'GABOR_FREQ',
                                   fallback=DEFAULTS['GABOR_FREQ'])

# keep backward-compat with older lowercase keys if present
WR_DIAG_CENTER_DEG = config.getfloat('thresholds', 'WR_DIAG_CENTER_DEG',
    fallback=config.getfloat('thresholds', 'wr_diag_center_deg',
                             fallback=DEFAULTS['WR_DIAG_CENTER_DEG']))
WR_DIAG_TOL_DEG    = config.getfloat('thresholds', 'WR_DIAG_TOL_DEG',
    fallback=config.getfloat('thresholds', 'wr_diag_tol_deg',
                             fallback=DEFAULTS['WR_DIAG_TOL_DEG']))
WR_EDGE_MARGIN_PX  = config.getint('thresholds', 'WR_EDGE_MARGIN_PX',
    fallback=config.getint('thresholds', 'wr_edge_margin_px',
                           fallback=DEFAULTS['WR_EDGE_MARGIN_PX']))
WR_MIN_LEN_PX      = config.getint('thresholds', 'WR_MIN_LEN_PX',
    fallback=config.getint('thresholds', 'wr_min_len_px',
                           fallback=DEFAULTS['WR_MIN_LEN_PX']))
WR_AR_MIN          = config.getfloat('thresholds', 'WR_AR_MIN',
    fallback=config.getfloat('thresholds', 'wr_ar_min',
                             fallback=DEFAULTS['WR_AR_MIN']))
WR_MAX_ANGLE_DEG   = config.getfloat('thresholds', 'WR_MAX_ANGLE_DEG',
                                     fallback=DEFAULTS['WR_MAX_ANGLE_DEG'])

LF_MIN_LEN_PX    = config.getint  ('thresholds', 'LF_MIN_LEN_PX',   fallback=DEFAULTS['LF_MIN_LEN_PX'])
LF_AR_MIN        = config.getfloat('thresholds', 'LF_AR_MIN',       fallback=DEFAULTS['LF_AR_MIN'])
LF_MAX_ANGLE_DEG = config.getfloat('thresholds', 'LF_MAX_ANGLE_DEG',fallback=DEFAULTS['LF_MAX_ANGLE_DEG'])
LF_EDGE_MARGIN   = config.getint  ('thresholds', 'LF_EDGE_MARGIN',  fallback=DEFAULTS['LF_EDGE_MARGIN'])

ECC_MIN   = config.getfloat('thresholds', 'ECC_MIN',   fallback=DEFAULTS['ECC_MIN'])
VESSEL_MIN= config.getfloat('thresholds', 'VESSEL_MIN',fallback=DEFAULTS['VESSEL_MIN'])

# Back-compat aliases expected elsewhere
AR_MIN = WR_AR_MIN
EDGE_MARGIN_PX = WR_EDGE_MARGIN_PX

# =============================================================================


# ---------- PATCH B: Wrinkle overlay helpers ----------
def _plot_centroids(ax, df):
    import numpy as np, pandas as pd
    if df is None or getattr(df, "empty", True):
        return
    if not {"centroid_x", "centroid_y"}.issubset(df.columns):
        return
    cx = pd.to_numeric(df["centroid_x"], errors="coerce").to_numpy()
    cy = pd.to_numeric(df["centroid_y"], errors="coerce").to_numpy()
    m = np.isfinite(cx) & np.isfinite(cy)
    if m.any():
        ax.scatter(cx[m], cy[m], s=28, facecolors="none", edgecolors="red", linewidths=1.5)



def _RUN_NEW_DIAGONAL_PIPELINE(file_path, canvas, figure, y_lower, y_upper):
    """
    Force the manual button to use the new unified selector (FRANGI/GABOR45),
    diagonal-only band-pass, and edge margin. Draws only our filtered result.
    """
    df, df_props, lines = detect_wrinkles_edgecast(file_path, y_lower, y_upper)

    # Persist for RHS overlay consumers
    if not hasattr(vars, "last_df_by_sensor"):
        vars.last_df_by_sensor = {}
    sensor_key = "0"
    try:
        if hasattr(figure, "axes") and figure.axes:
            t = str(figure.axes[0].get_title())
            if t.endswith("(2)"):
                sensor_key = "2"
            elif t.endswith("(1)"):
                sensor_key = "1"
    except Exception:
        pass
    vars.last_df_by_sensor[sensor_key] = {"wr": df_props, "lf": None}

    # Draw (NO legacy HUD text)
    ax = figure.add_subplot(111)
    ax.clear()
    ax.imshow(_viz_contrast(df.values), cmap="gray", aspect="auto")
    _draw_final_defects(ax, df.values, df_wr=df_props, df_lf=None,
                        title="", show_labels=True, skeleton_bool=None)
    canvas.draw()
    return df_props



def _ensure_measure_columns(_df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np, pandas as pd
    if _df is None or getattr(_df, "empty", True):
        return _df
    df = _df.copy()

    # angle to X-axis in degrees (abs)
    if "angle_deg" not in df.columns and "orientation" in df.columns:
        try:
            df["angle_deg"] = np.abs(np.degrees(pd.to_numeric(df["orientation"], errors="coerce")))
        except Exception:
            df["angle_deg"] = 0.0
    elif "angle_deg" not in df.columns:
        df["angle_deg"] = 0.0

    # length in pixels
    if "length_px" not in df.columns:
        if "major_axis_length" in df.columns:
            df["length_px"] = pd.to_numeric(df["major_axis_length"], errors="coerce")
        elif {"bbox-3", "bbox-1"}.issubset(df.columns):
            df["length_px"] = pd.to_numeric(df["bbox-3"], errors="coerce") - pd.to_numeric(df["bbox-1"], errors="coerce")
        elif {"bbox-2", "bbox-0"}.issubset(df.columns):
            df["length_px"] = pd.to_numeric(df["bbox-2"], errors="coerce") - pd.to_numeric(df["bbox-0"], errors="coerce")
        else:
            df["length_px"] = 0.0

    # AR = major/minor
    if "AR" not in df.columns:
        if {"major_axis_length", "minor_axis_length"}.issubset(df.columns):
            num = pd.to_numeric(df["major_axis_length"], errors="coerce")
            den = pd.to_numeric(df["minor_axis_length"], errors="coerce").replace(0, np.nan)
            df["AR"] = num / den
        else:
            df["AR"] = 0.0

    # sanitize
    for col in ("angle_deg", "length_px", "AR"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["length_px"] = df["length_px"].clip(lower=0)
    df["AR"] = df["AR"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # convenience: angle to Y-axis (thesis uses near-vertical)
    df["angle_to_y_deg"] = np.abs(90.0 - df["angle_deg"])

    return df


def apply_wrinkle_strict_filter(df: pd.DataFrame,
                                *,
                                min_len_px: int = WR_MIN_LEN_PX,
                                min_ar: float = WR_AR_MIN,
                                # keep a near-vertical band: |90° − angle| <= tol
                                angle_center_deg: float = 90.0,
                                angle_tol_deg: float = 15.0,   # 90±15 ⇒ 75..105; with abs() this becomes 75..90
                                x_edge: int | None = None,
                                edge_margin_px: int = WR_EDGE_MARGIN_PX,
                                y_lo: int | None = None,
                                y_hi: int | None = None) -> pd.DataFrame:
    """
    Strict filter for *wrinkles* per thesis:
      - long & thin (length/AR thresholds)
      - near-VERTICAL orientation (≈ 80° to X; i.e., small angle to Y)
      - optional: inside Y-ROI and to the right of coating edge + margin
    Returns a new filtered DataFrame.
    """
    if df is None or getattr(df, "empty", True):
        return df

    df = _ensure_measure_columns(df)  # creates angle_deg, length_px, AR, angle_to_y_deg

    # geometry: near-vertical (small angle to Y-axis)
    if "angle_to_y_deg" not in df.columns:
        df["angle_to_y_deg"] = np.abs(90.0 - pd.to_numeric(df["angle_deg"], errors="coerce"))

    keep = (
        (pd.to_numeric(df["length_px"], errors="coerce") >= float(min_len_px)) &
        (pd.to_numeric(df["AR"],        errors="coerce") >= float(min_ar)) &
        (df["angle_to_y_deg"] <= float(angle_tol_deg))    # e.g., <=15° from vertical
    )

    # optional Y-ROI
    if y_lo is not None and y_hi is not None and "centroid_y" in df.columns:
        cy = pd.to_numeric(df["centroid_y"], errors="coerce")
        keep &= (cy >= int(y_lo)) & (cy <= int(y_hi))

    # optional coating-edge margin (avoid edge artefacts)
    if x_edge is not None and "centroid_x" in df.columns:
        cx = pd.to_numeric(df["centroid_x"], errors="coerce")
        keep &= (cx >= (int(x_edge) + int(edge_margin_px)))

    out = df.loc[keep].reset_index(drop=True)
    return out


def apply_fold_strict_filter_old(df: pd.DataFrame,
                             *,
                             min_len_px: int = LF_MIN_LEN_PX,
                             min_ar: float = LF_AR_MIN,
                             max_angle_deg: float = LF_MAX_ANGLE_DEG,
                             x_edge: int | None = None,
                             edge_margin_px: int = 0,
                             y_lo: int | None = None,
                             y_hi: int | None = None) -> pd.DataFrame:
    """
    Strict filter for folds (very long, very elongated, almost horizontal).
    """
    if df is None or getattr(df, "empty", True):
        return df

    df = _ensure_measure_columns(df)

    keep = (
            (df["length_px"] >= float(min_len_px)) &
            (df["AR"] >= float(min_ar)) &
            (df["angle_deg"] <= float(max_angle_deg))
    )

    if y_lo is not None and y_hi is not None and {"centroid_y"}.issubset([*df.columns]):
        cy = pd.to_numeric(df["centroid_y"], errors="coerce")
        keep &= (cy >= y_lo) & (cy <= y_hi)

    if x_edge is not None and {"centroid_x"}.issubset([*df.columns]) and edge_margin_px > 0:
        cx = pd.to_numeric(df["centroid_x"], errors="coerce")
        keep &= (cx >= (int(x_edge) + int(edge_margin_px)))

    out = df.loc[keep].reset_index(drop=True)
    return out


# ---------- END STRICT FILTER HELPERS ----------


def _prep_overlay(base_img, mask, alpha=0.4):
    """
    Blend a binary mask into a grayscale image for visualization.
    """
    import numpy as np
    import matplotlib.cm as cm

    base_norm = (base_img - np.nanmin(base_img)) / (np.nanmax(base_img) - np.nanmin(base_img) + 1e-8)
    base_rgb = cm.gray(base_norm)[..., :3]

    if mask is not None and mask.any():
        mask_rgb = cm.autumn(mask.astype(float))[..., :3]  # orange-red overlay
        blended = (1 - alpha) * base_rgb + alpha * mask_rgb
    else:
        blended = base_rgb

    return blended


# --- Band masking helpers (coating edge, vertical & horizontal jumps) ---

def _mask_vertical_bands(df, x_edge=None, edge_margin_px=8, z_col=2.0):
    """
    Returns a boolean mask (True = keep, False = mask) that removes:
      1) a band around the coating edge (if x_edge is given), and
      2) columns with abnormally high vertical energy (big vertical ridges).
    """
    import numpy as np, pandas as pd
    a = df.values.astype(float)
    keep = np.ones_like(a, dtype=bool)

    # 1) mask band around coating edge
    if x_edge is not None:
        x0 = max(0, int(x_edge) - edge_margin_px)
        x1 = min(a.shape[1], int(x_edge) + edge_margin_px)
        keep[:, x0:x1] = False

    # 2) mask “strong vertical” columns using column standard deviation as a proxy
    col_std = np.nanstd(a, axis=0)
    thr = np.nanmedian(col_std) + z_col * np.nanstd(col_std)
    bad_cols = col_std > thr
    keep[:, bad_cols] = False

    return pd.DataFrame(keep, index=df.index, columns=df.columns)


def _mask_horizontal_bands(df, z_row=2.0):
    """
    Boolean mask (True = keep) that removes rows with strong horizontal jumps/edges.
    Uses row-wise max(abs(diff along x)) to detect step rows.
    """
    import numpy as np, pandas as pd
    a = df.values.astype(float)
    keep = np.ones_like(a, dtype=bool)

    row_step = np.nanmax(np.abs(np.diff(a, axis=1)), axis=1)  # per-row jump strength
    thr = np.nanmedian(row_step) + z_row * np.nanstd(row_step)
    bad_rows = row_step > thr
    keep[bad_rows, :] = False

    return pd.DataFrame(keep, index=df.index, columns=df.columns)


def _apply_keep_mask(df, keep_mask):
    """
    Apply boolean keep mask to a DataFrame (False -> NaN).
    """
    import numpy as np, pandas as pd
    a = df.values.astype(float)
    a[~keep_mask.values] = np.nan
    return pd.DataFrame(a, index=df.index, columns=df.columns)



def _draw_wrinkle_overlay(ax, data, skeleton_bool, title=""):
    """
    Draw grayscale base and overlay a boolean skeleton (True pixels) transparently.
    """
    import numpy as np
    ax.clear()
    ax.imshow(data, cmap='gray', aspect='auto')
    try:
        if skeleton_bool is not None:
            sk = np.asarray(skeleton_bool, dtype=bool)
            if sk.any():
                sk_mask = np.ma.masked_where(~sk, sk)
                ax.imshow(sk_mask, cmap='winter', alpha=0.8, aspect='auto', interpolation='nearest')
    except Exception:
        pass
    if title:
        ax.set_title(title)
    ax.axis('off')



def _estimate_coating_edge_polyline(img_2d):
    """
    Rough coating-edge estimator:
    - compute |dI/dx| per row and take argmax column index
    - smooth the x-index along y to reduce jitter
    Returns (y_index, x_index) for plotting.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    a = np.asarray(img_2d, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    gx = np.abs(np.diff(a, axis=1))
    if gx.shape[1] == 0:
        return np.arange(a.shape[0]), np.zeros(a.shape[0])
    gx = gaussian_filter1d(gx, sigma=1.0, axis=1)
    x_idx = np.argmax(gx, axis=1).astype(float)
    x_idx = gaussian_filter1d(x_idx, sigma=5.0)  # smooth across rows
    y_idx = np.arange(a.shape[0])
    return y_idx, x_idx


def detect_wrinkles_fan(file_path, y_lower, y_upper):
    """
    Thesis-like diagonal fan with adaptive fallback.
    Returns (df, df_props, dbg) where dbg['lines'] are the short diagonals,
    dbg['x_ref'] the straight reference line (dark blue), dbg['x_edge'] the
    cyan coating-edge polyline.
    """
    import numpy as np, pandas as pd
    from skimage.filters import gaussian
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    # ---------- load + ROI + normalize ----------
    df = _read_heightmap_table(file_path)
    df = crop_to_roi(df, int(y_lower), int(y_upper))
    img = df.values.astype(float)

    a = np.where(np.isfinite(img), img, np.nan)
    p1, p99 = np.nanpercentile(a, (1, 99))
    imn = np.clip((a - p1) / (p99 - p1 + 1e-9), 0, 1)
    # imn was created a few lines above
    H, W = imn.shape

    # coating edge + straight reference
    x_edge = _estimate_coating_edge_polyline(imn)  # <-- use imn here too

    X_REF_PCTL = 25.0
    x_ref = float(np.nanpercentile(x_edge, X_REF_PCTL))
    x_ref = np.clip(x_ref, W * 0.10, W - 5.0)

    # ---------- detector (adaptive) ----------
    def _run(canny_lo, canny_hi, h_thresh, line_len, line_gap,
             ang_lo=35.0, ang_hi=80.0, anchor_px=12, Lmin=20, Lmax=200):
        ed = canny(gaussian(imn, sigma=1.2, preserve_range=True),
                   sigma=1.2, low_threshold=canny_lo, high_threshold=canny_hi)
        raw = probabilistic_hough_line(ed,
                                       threshold=h_thresh,
                                       line_length=line_len,
                                       line_gap=line_gap)
        kept = []
        for (r0, c0), (r1, c1) in raw:   # (row,col)
            x0, y0 = float(c0), float(r0)
            x1, y1 = float(c1), float(r1)

            # angle w.r.t horizontal; accept both diagonals by |angle|
            ang = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            if not (ang_lo <= ang <= ang_hi):
                continue

            # anchor: one end near the straight ref line, the other to the RIGHT of it
            d0, d1 = abs(x0 - x_ref), abs(x1 - x_ref)
            xn, yn = (x0, y0) if d0 < d1 else (x1, y1)
            xf, yf = (x1, y1) if d0 < d1 else (x0, y0)
            if d0 > anchor_px and d1 > anchor_px:
                continue
            if xf <= x_ref + 2.0:
                continue

            L = float(np.hypot(x1 - x0, y1 - y0))
            if not (Lmin <= L <= Lmax):
                continue

            kept.append(((xn, yn), (xf, yf), ang, L))

        # vertical de-duplication for tidy spacing
        kept.sort(key=lambda s: min(s[0][1], s[1][1]))
        pruned, last_y = [], -1e9
        for s in kept:
            y = min(s[0][1], s[1][1])
            if y - last_y < 22:          # spacing in pixels between slivers
                continue
            pruned.append(s)
            last_y = y

        return raw, pruned

    # pass 1 (normal), pass 2 (relaxed), pass 3 (very relaxed)
    tries = [
        # Pass 1 — thesis-y (clean fan)
        dict(
            canny_lo=0.03, canny_hi=0.10,
            h_thresh=6, line_len=12, line_gap=4,
            ang_lo=40.0, ang_hi=70.0,  # narrower diagonal band
            anchor_px=int(min(x_ref, 50)),  # start near straight ref line
            Lmin=12, Lmax=220
        ),
        # Pass 2 — a bit looser (if pass 1 finds <3)
        dict(
            canny_lo=0.02, canny_hi=0.09,
            h_thresh=4, line_len=10, line_gap=6,
            ang_lo=35.0, ang_hi=75.0,
            anchor_px=int(min(x_ref, 60)),
            Lmin=10, Lmax=250
        ),
        # Pass 3 — safety net (if still <3)
        dict(
            canny_lo=0.015, canny_hi=0.08,
            h_thresh=3, line_len=8, line_gap=8,
            ang_lo=30.0, ang_hi=80.0,
            anchor_px=int(min(x_ref, 70)),
            Lmin=8, Lmax=280
        ),
    ]

    pruned = []
    raw_last = []

    for t in tries:
        raw_last, pruned = _run(**t)
        print(f"[fan] raw={len(raw_last)} kept={len(pruned)} with {t}")
        if len(pruned) >= 3:  # “fan” detected
            break

    # SUPER fallback
    if len(pruned) < 2:
        raw_last, pruned = _run(
            canny_lo=0.02, canny_hi=0.09,
            h_thresh=3, line_len=6, line_gap=6,
            ang_lo=25.0, ang_hi=85.0,
            anchor_px=25,  # <-- was int(x_ref)
            Lmin=8, Lmax=240
        )
        print(f"[fan] SUPER fallback kept={len(pruned)}")

    # build props + dbg
    rows = []
    for i, s in enumerate(pruned, 1):
        (x0, y0), (x1, y1), ang, L = s
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rows.append({
            "label": i,
            "centroid_x": cx, "centroid_y": cy,
            "major_axis_length": L, "minor_axis_length": 1.0,
            "orientation": np.radians(ang), "angle_deg": ang
        })
    df_props = pd.DataFrame(rows)

    dbg = {
        "lines": [((int(s[0][0]), int(s[0][1])), (int(s[1][0]), int(s[1][1]))) for s in pruned],
        "x_ref": x_ref,
        "x_edge": x_edge,
    }
    return df, df_props, dbg




def _draw_final_defects(ax, img, df_wr=None, df_lf=None, *, title: str = "", show_labels: bool = True,
                        skeleton_bool=None):
    """
    Render final (filtered) defects only:
    - grayscale base
    - oriented ellipses (major/minor axis, orientation)
    - centroid markers (yellow=wrinkle, cyan=fold)
    - small text labels (length px and angle°)
    - legend
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    import pandas as pd
    ax.clear()
    ax.imshow(img, cmap="gray", aspect="auto")

    def _safe_series(df, key, default=0.0):
        try:
            return pd.to_numeric(df.get(key, default), errors="coerce").fillna(default)
        except Exception:
            return pd.Series([], dtype=float)

    def _draw_df(df, color, marker, label):
        if df is None or getattr(df, "empty", True):
            return 0
        L = _safe_series(df, "major_axis_length", 0.0)
        W = _safe_series(df, "minor_axis_length", 0.0)
        if (L.max() == 0) or (W.max() == 0):
            L = _safe_series(df, "bbox_h", 0.0)
            W = _safe_series(df, "bbox_w", 0.0)
        cx = _safe_series(df, "centroid_x", 0.0).to_numpy()
        cy = _safe_series(df, "centroid_y", 0.0).to_numpy()
        if "angle_deg" in getattr(df, "columns", []):
            ang = _safe_series(df, "angle_deg", 0.0).to_numpy()
        else:
            ang = _safe_series(df, "orientation", 0.0).to_numpy() * (180.0 / 3.141592653589793)

        n = 0
        for i in range(min(len(cx), len(cy))):
            x, y = float(cx[i]), float(cy[i])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            try:
                w = float(W.iloc[i]) if hasattr(W, "iloc") else float(W[i])
                h = float(L.iloc[i]) if hasattr(L, "iloc") else float(L[i])
            except Exception:
                w, h = 4.0, 12.0
            theta = float(ang[i])

            try:
                e = Ellipse((x, y), width=max(1.0, w), height=max(1.0, h), angle=theta, fill=False, lw=1.0, color=color,
                            alpha=0.95)
                ax.add_patch(e)
            except Exception:
                pass

            ax.scatter([x], [y], s=22, c=color, marker=marker, linewidths=0.7, edgecolors="black", alpha=0.9)

            if show_labels:
                try:
                    ax.text(x + 3, y - 3, f"L={int(h)} px, θ={int(round(theta))}°", color=color, fontsize=7, ha="left",
                            va="top", bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=1.5))
                except Exception:
                    pass
            n += 1
        return n

    n_wr = _draw_df(df_wr, color="yellow", marker="o", label="Wrinkle")
    n_lf = _draw_df(df_lf, color="cyan", marker="o", label="Fold")

    handles = []
    from matplotlib.lines import Line2D
    if n_wr > 0:
        handles.append(Line2D([0], [0], color="yellow", lw=2, label="Wrinkle"))
    if n_lf > 0:
        handles.append(Line2D([0], [0], color="cyan", lw=2, label="Fold"))
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)

    if title:
        ax.set_title(title)
    ax.axis("off")

    if skeleton_bool is not None:
        try:
            skel_mask = np.ma.masked_where(~skeleton_bool, skeleton_bool)
            ax.imshow(skel_mask, cmap='winter', alpha=0.8, aspect='auto', interpolation='nearest')
        except Exception:
            pass

    if title:
        ax.set_title(title)
    ax.axis("off")


def _read_heightmap_table(file_path):
    """
    Robust reader for FAZU exports. Tries multiple (sep, decimal) combos
    and returns a clean numeric DataFrame. Raises ValueError if nothing works.
    """
    import pandas as pd
    from pathlib import Path

    p = Path(str(file_path))

    # Excel → simple
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, header=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if df.empty:
            raise ValueError("Excel file contained no numeric data after cleaning.")
        return df

    # Common (separator, decimal) pairs in your data
    candidates = [
        (";", ","),  # semicolon + comma-decimal
        (";", "."),  # semicolon + dot-decimal
        (",", "."),  # comma + dot-decimal
        (",", ","),  # comma + comma-decimal
        ("\t", "."),  # tab + dot
        ("\t", ","),  # tab + comma
        (r"\s+", "."),  # whitespace + dot
        (r"\s+", ","),  # whitespace + comma
    ]

    last_err = None
    for sep, dec in candidates:
        try:
            df = pd.read_csv(file_path, sep=sep, decimal=dec, header=None, engine="python")
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if df.shape[1] >= 2 and df.notna().to_numpy().any():
                return df
        except Exception as e:
            last_err = e

    # Last-chance: let pandas auto-detect sep
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if df.shape[1] >= 2 and df.notna().to_numpy().any():
            return df
    except Exception as e:
        last_err = e

    raise ValueError(f"Could not parse file with any sep/decimal combination. Last error: {last_err}")


def crop_to_roi(df: pd.DataFrame, y_lower: int, y_upper: int) -> pd.DataFrame:
    y0 = max(0, int(y_lower))
    y1 = min(df.shape[0], int(y_upper))
    return df.iloc[y0:y1, :]


def suppress_horizontal_jumps(df: pd.DataFrame, z=0.8) -> pd.DataFrame:
    """Mask rows with big step changes across X (sensor jump lines)."""
    arr = df.values.astype(float)
    diffs = np.nanmax(np.abs(np.diff(arr, axis=1)), axis=1)
    thr = np.nanmedian(diffs) + z * np.nanstd(diffs)
    bad_rows = diffs > thr
    arr[bad_rows, :] = np.nan
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def mask_near_edge(df: pd.DataFrame, x_edge: int, margin_px: int = 8) -> pd.DataFrame:
    """Zero out a vertical strip around the coating edge to avoid false blobs."""
    arr = df.values.copy()
    x0 = max(0, x_edge - margin_px)
    x1 = min(arr.shape[1], x_edge + margin_px)
    arr[:, x0:x1] = np.nan
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def frangi_detect_on_axis(df_2d: pd.DataFrame, ax, title: str = ""):
    """Run Frangi and draw a transparent skeleton overlay (zeros invisible)."""
    import numpy as np
    # Call the imported frangi_core; pass stricter params if needed
    #    df_res, ridge, skeleton = frangi_core(df_2d.values, min_size_frac=2e-4, angle_exclude=30)

    ax.clear()
    ax.imshow(df_2d.values, cmap='gray', aspect='auto')

    # Draw only True pixels from skeleton; 0 stays transparent
    # skel_bool = skeleton.astype(bool)
    # skel_mask = np.ma.masked_where(~skel_bool, skel_bool)
    # ax.imshow(skel_mask, cmap='winter', alpha=0.8, aspect='auto', interpolation='nearest')

    if title:
        ax.set_title(title)
    ax.set_axis_off()


# return df_res


def getTSfromInfluxDB(measurement, start_time, stop_time, bucket, TargetDir):
    print(f"[INFO] Starte Datenabruf für {measurement} ...")
    global query_api

    query = f"""from(bucket: "{bucket}") 
    |> range(start: {start_time}, stop: {stop_time}) 
    |> filter(fn: (r) => r["_measurement"] == "{measurement}") 
    |> aggregateWindow(every: 1ms, fn: last, createEmpty: false)
    """

    df = pd.DataFrame()
    try:
        tables = vars.query_api.query(query)
        frames = [pd.DataFrame({f'{table.records[0].get_field()}': [record.get_value() for record in table.records]})
                  for table in tables]
        df = pd.concat(frames, axis=1)
        startName = start_time.replace(":", "_")
        startName = startName.replace("+", "_")
        df.to_csv(f'{TargetDir}/{measurement}_{startName}.csv')

        print(f"[INFO] Data of measurement {measurement} saved to {TargetDir}")
    except ValueError:
        print(f"[FEHLER] Die Datenbank hat eine leere Antwort für {measurement} gesendet")
        print("Überprüfen sie den Zeitstemepl")


def access_influxDB(report_date, report_start_time, report_stop_time, mode, terminal):
    if mode == "manual":
        day, month, year = report_date.split('.')
        date_int_list = [int(year), int(month), int(day)]

        start_hours, start_minutes, start_seconds = report_start_time.split(',')
        start_time_list = [int(start_hours), int(start_minutes), int(start_seconds)]

        stop_hours, stop_minutes, stop_seconds = report_stop_time.split(',')
        stop_time_list = [int(stop_hours), int(stop_minutes), int(stop_seconds)]

        start = datetime(date_int_list[0], date_int_list[1], date_int_list[2], start_time_list[0], start_time_list[1],
                         start_time_list[2])  # JJJJ,MM,DD,HH,MM,SS
        stop = datetime(date_int_list[0], date_int_list[1], date_int_list[2], stop_time_list[0], stop_time_list[1],
                        stop_time_list[2])

        # Converting to InfluXDB UTC-Timestamp
        start = start.astimezone(pytz.UTC)
        stop = stop.astimezone(pytz.UTC)

        start = start.isoformat().replace("+00.00", "Z")
        stop = stop.isoformat().replace("+00.00", "Z")

    if mode == "batch":
        start = report_start_time
        stop = report_stop_time

    # Acessing influxDB
    token = "REDACTED"
    global org
    org = "Institute"
    url = "http://127.0.0.1:8086"
    # url = "http://localhost:8086"

    # Setting up InfluxDB
    client = InfluxDBClient(url=url, token=token, org=org)
    vars.query_api = client.query_api()
    bucket = "kalander"

    # Sensor_0 data
    add_terminal_line(terminal, "Lädt Daten von Sensor 0")
    getTSfromInfluxDB("Sensor_0", start, stop, bucket, vars.output_folder)
    add_terminal_line(terminal, "Sensor 0 abgeschlossen")

    # Sensor_1 data
    add_terminal_line(terminal, "Lädt Daten von Sensor 1")
    getTSfromInfluxDB("Sensor_1", start, stop, bucket, vars.output_folder)
    add_terminal_line(terminal, "Sensor 1 abgeschlossen")

    # Sensor_2 data
    add_terminal_line(terminal, "Lädt Daten von Sensor 2")
    getTSfromInfluxDB("Sensor_2", start, stop, bucket, vars.output_folder)
    add_terminal_line(terminal, "Sensor 2 abgeschlossen")

    # OPCUA data
    add_terminal_line(terminal, "Lädt Daten von OPC-UA Server")
    getTSfromInfluxDB("opcua", start, stop, bucket, vars.output_folder)
    add_terminal_line(terminal, "OPC-UA Server abgeschlossen")


def initialize_sensors(process_list, trigger_list, status):
    ##############################################################################
    # SETUP Multiprocessing
    # Jeder Sensor wird auf einem eigenen Prozessorkern ausgeführt
    ##############################################################################
    # Erstellen der Prozesse für die Sensoren, Funktion handle_sensor übernimmt den Sensorbetrieb
    S1Process, S2Process, S3Process, = process_list[0], process_list[1], process_list[2]
    Trigger_S1, Trigger_S2, Trigger_S3 = trigger_list[0], trigger_list[1], trigger_list[2]
    S1Process.start()
    S2Process.start()
    S3Process.start()

    # Hier wird gewartet, bis alle Sensoren initialisiert sind und den Status True zurückmelden

    while True:
        # Status empfangen
        try:
            resp = status.get(block=False)

        except queue.Empty:
            resp = None
        # resp=Status.get()
        # Status in Liste speichern
        if resp == True:
            vars.status_list.append(resp)

        # Wenn alle drei Sensoren bereit sind wird Schleife beendet
        if len(vars.status_list) == 3:
            print("Sensors initialized")
            break

        # Falls die Sensoren vorher einen Fehler werfen, wird das Programm beendet
        if S1Process.is_alive() == False or S2Process.is_alive() == False or S3Process.is_alive() == False:
            Trigger_S1.put(False)
            Trigger_S2.put(False)
            Trigger_S3.put(False)
            print("[main] =>Error occured: Emergency Stop .....")
            # sys.exit()


def start_measurement(trigger_list):
    Trigger_S1, Trigger_S2, Trigger_S3 = trigger_list[0], trigger_list[1], trigger_list[2]
    print("Starting Measurements\n[main] Sending Trigger to threads....")
    Trigger_S1.put(True)
    Trigger_S2.put(True)
    Trigger_S3.put(True)


def stop_measurement(trigger_list):
    Trigger_S1, Trigger_S2, Trigger_S3 = trigger_list[0], trigger_list[1], trigger_list[2]
    print("Ending Measurements....")
    Trigger_S1.put(False)
    Trigger_S2.put(False)
    Trigger_S3.put(False)


def end_program(process_list):
    ###############################################################################
    # Sensor Handling Function
    # This function connects to the sensor and extracts the data
    ###############################################################################

    S1Process, S2Process, S3Process, = process_list[0], process_list[1], process_list[2]
    print("shutting down...")
    try:
        S1Process.kill()
        S2Process.kill()
        S3Process.kill()
    except:
        print("No Processes to kill")


def handle_sensor(IPArray, SensorNumber, FreqSetting, ScaleSetting, Status, Trigger):
    # Preparing InfluxDB
    token = "REDACTED"
    org = "Institute"
    url = "http://127.0.0.1:8086"
    # url = "http://localhost:8086"

    write_client = InfluxDBClient(url=url, token=token, org=org)

    write_api = write_client.write_api(write_options=SYNCHRONOUS)

    image_available = False  # Flag to confirm the completion of image acquisition.
    luminance_enable = 1
    deviceId = SensorNumber
    ysize = 500
    ysize_acquired = 0  # Number of Y lines of acquired image.
    z_val = []  # The buffer for height image.
    # lumi_val = []            # The buffer for luminance image.
    time_array = []

    ###############################################################################
    # Inner Callback function
    # It is called when the specified number of profiles are received.
    ###############################################################################
    def callback_s_a(p_header,
                     p_height,
                     p_lumi,
                     luminance_enable,
                     xpointnum,
                     profnum,
                     notify, user):

        nonlocal ysize_acquired
        nonlocal image_available
        nonlocal z_val
        nonlocal ysize
        # nonlocal lumi_val
        # nonlocal luminance_enable

        if (notify == 0) or (notify == 0x10000):
            # if profnum != 0:
            if image_available is False:
                # image_available = True
                for i in range(xpointnum * profnum):
                    z_val[i] = p_height[i]
                    # if luminance_enable == 1:
                    #  lumi_val[i] = p_lumi[i]

                ysize_acquired = profnum
                if ysize_acquired == ysize:
                    image_available = True

        return

    # IP Adress Configuration
    ethernetConfig = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
    ethernetConfig.abyIpAddress[0] = IPArray[0]  # IP address
    ethernetConfig.abyIpAddress[1] = IPArray[1]
    ethernetConfig.abyIpAddress[2] = IPArray[2]
    ethernetConfig.abyIpAddress[3] = IPArray[3]
    ethernetConfig.wPortNo = 24691  # Port No.
    HighSpeedPortNo = 24692  # Port No. for high-speed

    # Ethernet open
    res = LJXAwrap.LJX8IF_EthernetOpen(deviceId, ethernetConfig)
    print(f"[Sensor {SensorNumber}] LJXAwrap.LJX8IF_EthernetOpen:", hex(res))
    if res != 0:
        print("Failed to connect contoller.")
        print("Exit the program.")

        sys.exit()

    # Setting Sampling cycle to 100Hz
    SetLJXSetting(deviceId=SensorNumber, setType=0x10, setCategory=0x02, setItem=0x02, setValue=FreqSetting)

    # Setting X-Axis Sampling rate to 1/2 800px->400px
    SetLJXSetting(deviceId=SensorNumber, setType=0x10, setCategory=0x02, setItem=0x02, setValue=ScaleSetting)

    # Setting Batch Mode to off
    SetLJXSetting(deviceId=SensorNumber, setType=0x10, setCategory=0x00, setItem=0x03, setValue=0)

    # Initialize Hi-Speed Communication
    my_callback_s_a = LJXAwrap.LJX8IF_CALLBACK_SIMPLE_ARRAY(callback_s_a)

    res = LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(
        deviceId,
        ethernetConfig,
        HighSpeedPortNo,
        my_callback_s_a,
        ysize,
        0)
    print(f"[Sensor {SensorNumber}] LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray:",
          hex(res))
    if res != 0:
        print("\nExit the program.")
        sys.exit()

    # PreStart Hi-Speed Communication
    req = LJXAwrap.LJX8IF_HIGH_SPEED_PRE_START_REQ()
    req.bySendPosition = 2
    profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()

    res = LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication(
        deviceId,
        req,
        profinfo)
    print(f"[Sensor {SensorNumber}] LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication:", hex(res))
    if res != 0:
        print("\nExit the program.")
        sys.exit()

    # allocate the memory
    xsize = profinfo.wProfileDataCount
    z_val = [0] * xsize * ysize
    lumi_val = [0] * xsize * ysize

    # Start Hi-Speed Communication
    image_available = False
    res = LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(deviceId)
    print(f"[Sensor {SensorNumber}]LJXAwrap.LJX8IF_StartHighSpeedDataCommunication:", hex(res))
    if res != 0:
        print("\nExit the program.")
        sys.exit()

    # Return to Main Function that Sensor is ready
    # Status[deviceId]=True
    Status.put(True)
    print(f"Sensor {deviceId} is ready for measurement!")
    # Start Measure (Start Batch)
    while True:
        while True:
            if Trigger.get() == True:
                LJXAwrap.LJX8IF_StartMeasure(deviceId)

                time_array.append(datetime.now(timezone.utc))
                break

        while True:

            if image_available:

                if ysize_acquired == ysize:
                    image_available = False
                time_array.append(datetime.now(timezone.utc))

                print(f"[Sensor {deviceId}] Received Data Batch with xsize: {xsize}")
                tmpData = extract_data(z_val=z_val, BatchSize=ysize, deviceId=deviceId, time_array=time_array)
                # print(len(tmpData["timestamp"]))
                # would send data of tmpCopy to influxDB here
                PushDatatoInfluxDB(tmpData, f"Sensor_{deviceId}", write_api)
                # Überprüfen ob die Messung beendet werden soll
                try:
                    data = Trigger.get(block=False)
                except queue.Empty:
                    data = None
                if data == False:
                    LJXAwrap.LJX8IF_StopMeasure(deviceId)
                    break

        if image_available is not True:
            print(f"[Sensor {SensorNumber}]==>Terminated normally.")

    # Stop
    res = LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(deviceId)
    print(f"[Sensor {SensorNumber}]LJXAwrap.LJX8IF_StopHighSpeedDataCommunication:", hex(res))

    # Finalize
    res = LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(deviceId)
    print(f"[Sensor {SensorNumber}]LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication:", hex(res))

    # Close
    res = LJXAwrap.LJX8IF_CommunicationClose(deviceId)
    print(f"[Sensor {SensorNumber}]LJXAwrap.LJX8IF_CommunicationClose:", hex(res))

    sys.exit()


def extract_data(z_val, BatchSize, deviceId, time_array):
    ###############################################################################
    # Data extraction function
    # It is called when a batch is received in order to extract the profile data.
    ###############################################################################
    # Leeres Data-Frame zum zwischenspeichern
    DataFrame = pd.DataFrame()
    ZUnit = ctypes.c_ushort()
    LJXAwrap.LJX8IF_GetZUnitSimpleArray(deviceId, ZUnit)
    dataMatrix = np.split(np.array(z_val), BatchSize)

    xsize = len(dataMatrix[0])
    # Generierung von zeitstempeln für InfluxDB
    timestamps = pd.date_range(start=time_array[-2], end=time_array[-1], periods=int(BatchSize + 1))
    # Löschen alter Messpunkte
    time_array = time_array[-2:]
    timestamps = timestamps[1:]
    # Extraktion der Z-Werte
    z_array = []
    for el in dataMatrix:
        z_val_mm = [0.0] * int(xsize)

        for i in range(int(xsize)):

            # Convert Z data to the actual length in millimeters
            if el[i] == 0:  # invalid value
                z_val_mm[i] = np.nan
                # z_val_mm[i] = 0.0
            else:
                # 'Simple array data' is offset to be unsigned 16-bit data.
                # Decode by subtracting 32768 to get a signed value.
                z_val_mm[i] = int(el[i]) - 32768  # decode
                z_val_mm[i] *= ZUnit.value / 100.0  # um
                z_val_mm[i] /= 1000.0  # mm
                z_val_mm[i] = np.round(z_val_mm[i], 6)  # round

        z_array.append(z_val_mm)
    DataFrame['timestamp'] = timestamps
    DataFrame['height profile'] = z_array
    return DataFrame


def PushDatatoInfluxDB(DataFrame, name, write_api):
    ###############################################################################
    # Sending Data to InfluxDB
    # It is called when a batch is received in order to send profile data.
    ###############################################################################
    bucket = "kalander"
    org = "Institute"
    pointList = []
    # i=int(len(DataFrame["height profile"])/2)
    for i in range(0, len(DataFrame["height profile"])):
        MyPoint = Point(name).time(DataFrame["timestamp"][i])
        for j in range(0, len(DataFrame["height profile"][i])):
            MyPoint.field(f"px_" + str(j).zfill(3), DataFrame["height profile"][i][j])

        pointList.append(MyPoint)
    write_api.write(bucket=bucket, org=org, record=pointList, write_precision='ms', write_options=SYNCHRONOUS)


def SetLJXSetting(deviceId, setType, setCategory, setItem, setValue):
    ###############################################################################
    # Setting Sensor Settings
    # It is called to change a Sensor Setting before measurement
    ###############################################################################
    depth = 1  # 0: Write, 1: Running, 2: Save

    targetSetting = LJXAwrap.LJX8IF_TARGET_SETTING()
    targetSetting.byType = setType  # Program No.0
    targetSetting.byCategory = setCategory  # Trigger Category
    targetSetting.byItem = setItem  # Sampling Cycle
    targetSetting.byTarget1 = 0x00  # reserved
    targetSetting.byTarget2 = 0x00  # reserved
    targetSetting.byTarget3 = 0x00  # reserved
    targetSetting.byTarget4 = 0x00  # reserved

    dataSize = 4

    # setting the value
    err = ctypes.c_uint()
    pyArr = [setValue, 0, 0, 0]
    settingData_set = (ctypes.c_ubyte * dataSize)(*pyArr)

    res = LJXAwrap.LJX8IF_SetSetting(deviceId, depth,
                                     targetSetting,
                                     settingData_set, dataSize, err)
    print(f"[Sensor {deviceId}] LJXAwrap.LJX8IF_SetSetting:", hex(res),
          "<Set value>=", settingData_set[0],
          "<SettingError>=", hex(err.value))


def sample_random_values_average(dataframe, sample_size):
    sum_of_values = 0.0
    number_of_usable_values = 0
    nr_of_rows, nr_of_columns = dataframe.shape
    nr_of_rows -= 1
    nr_of_columns -= 1
    for x in range(sample_size):
        rand_row = random.randint(1, nr_of_rows)
        rand_col = random.randint(1, nr_of_columns)
        rand_val = dataframe.iat[rand_row, rand_col]
        if type(rand_val) is not np.nan:
            sum_of_values += rand_val
            number_of_usable_values += 1
        else:
            continue

    average = sum_of_values / float(number_of_usable_values)
    return average


def _viz_contrast(a, lo=2, hi=98):
    import numpy as np
    a = a.astype("float64")
    a = np.where(np.isfinite(a), a, np.nan)
    p_lo, p_hi = np.nanpercentile(a, (lo, hi))
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        return np.zeros_like(a, dtype=float)
    out = (a - p_lo) / (p_hi - p_lo + 1e-9)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def create_output_plot(file_path, canvas, fig, sensor_nr):  # OPTIMIZED (David)
    # Load CSV once and determine usable columns
    df_header = pd.read_csv(file_path, nrows=1)
    num_cols = len(df_header.columns)
    df = pd.read_csv(file_path, decimal='.', header=1, sep=",", usecols=range(5, num_cols - 5))

    jump_peaks = []

    if sensor_nr == 0 or sensor_nr == 2:
        # Equalize row height
        equalize_row_height(df, 2)
        jump_peaks = find_substrate_jumps(df, 2)

    # Contrast enhancement thresholds
    img_data = df.to_numpy()
    min_thres = np.nanpercentile(img_data, 0.1)
    max_thres = np.nanpercentile(img_data, 99.1)

    # Mask outliers
    mask = (img_data < min_thres) | (img_data > max_thres)
    filtered_data = np.where(mask, np.nan, img_data)

    # Apply Gaussian smoothing
    smoothed_data = gaussian_filter(filtered_data, sigma=2)

    # Start plotting
    fig.clear()
    mode = vars.image_mode

    if mode == "3D_reduced":
        ax = fig.add_subplot(111, projection='3d')
        ax.set_position([0.05, 0.05, 0.9, 0.9])
        ax.set_box_aspect([1, 5, 1])

        y, x = np.indices(smoothed_data.shape)
        z = filtered_data

        res_row = res_col = 5
        ax.plot_surface(
            x[::res_row, ::res_col],
            y[::res_row, ::res_col],
            z[::res_row, ::res_col],
            cmap='RdYlGn_r'
        )
        ax.set_xlabel('Spalten')
        ax.set_ylabel('Zeilen')

    elif mode in ("2D_reduced", "2D_full"):
        ax = fig.add_subplot(111)
        plot_data = smoothed_data if mode == "2D_reduced" else filtered_data
        jump_x_values = [10] * len(jump_peaks)
        ax.imshow(plot_data, cmap='gray', aspect='auto')
        ax.scatter(jump_x_values, jump_peaks, color="red", label="jumps")
        ax.set_title('2D Grauwerte')
        ax.set_xlabel('Spalten')
        ax.set_ylabel('Zeilen')

    canvas.draw()
    return df


def add_terminal_line(terminal, text):
    named_time_tuple = time.localtime()
    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", named_time_tuple)
    terminal.insert(ctk.INSERT, f"{time_string} | {text} \n")
    terminal.see(tk.END)


def batch_measurement(terminal, batch_len, triggers, master):
    def sensor_runner():
        time.sleep(2)
        start_measurement(trigger_list=triggers)
        time.sleep(batch_len + 2)
        stop_measurement(trigger_list=triggers)
        time.sleep(2)

    def terminal_runner():
        add_terminal_line(terminal, "Batchmessung wird vorbereitet...")
        time.sleep(2)
        add_terminal_line(terminal, "Messung gestartet. BITTE WARTEN")
        time.sleep(1)

        batch_start_time = datetime.now()
        batch_start_time = batch_start_time.replace(microsecond=0)
        batch_start_time = batch_start_time.astimezone(pytz.UTC)
        batch_start_time = batch_start_time.isoformat().replace("+00.00", "Z")

        time_remaining = batch_len
        while time_remaining > 0:
            time.sleep(1)
            add_terminal_line(terminal, f"Messung läuft | {time_remaining}")
            time_remaining -= 1

        batch_end_time = datetime.now()
        batch_end_time = batch_end_time.replace(microsecond=0)
        batch_end_time = batch_end_time.astimezone(pytz.UTC)
        batch_end_time = batch_end_time.isoformat().replace("+00.00", "Z")

        time.sleep(1)
        add_terminal_line(terminal, "Messung beendet\n")
        add_terminal_line(terminal, "\n")
        time.sleep(2)
        access_influxDB(report_date=None, report_start_time=batch_start_time, report_stop_time=batch_end_time,
                        mode="batch", terminal=terminal)
        call_output_plots(master, directory=vars.output_folder)

    sensor_thread = threading.Thread(target=sensor_runner)
    terminal_thread = threading.Thread(target=terminal_runner)
    sensor_thread.start()
    terminal_thread.start()


def call_output_plots(self, directory):
    import os

    def latest_csv(prefix: str):
        # find the latest CSV for a given sensor prefix
        candidates = [f for f in os.listdir(directory)
                      if f.startswith(prefix) and f.lower().endswith(".csv")]
        if not candidates:
            return None
        full = [os.path.join(directory, f) for f in candidates]
        full.sort(key=lambda p: os.path.getmtime(p))  # by modification time
        return full[-1]  # newest

    # Sensor 0
    p0 = latest_csv("Sensor_0")
    if p0:
        vars.current_sensor_0_file_path = p0
        create_output_plot(
            p0,
            self.frame_output_plots.canvas_sensor_0,
            self.frame_output_plots.sensor_0_fig,
            0
        )
        # propagate ROI fields if present
        if hasattr(self, "frame_wrinkle_detection"):
            if hasattr(vars, "lower_y_value"):
                self.frame_wrinkle_detection.entry_y_lower_value.set(vars.lower_y_value)
            if hasattr(vars, "upper_y_value"):
                self.frame_wrinkle_detection.entry_y_upper_value.set(vars.upper_y_value)

    # Sensor 1
    p1 = latest_csv("Sensor_1")
    if p1:
        vars.current_sensor_1_file_path = p1
        create_output_plot(
            p1,
            self.frame_output_plots.canvas_sensor_1,
            self.frame_output_plots.sensor_1_fig,
            1
        )

    # Sensor 2
    p2 = latest_csv("Sensor_2")
    if p2:
        vars.current_sensor_2_file_path = p2
        create_output_plot(
            p2,
            self.frame_output_plots.canvas_sensor_2,
            self.frame_output_plots.sensor_2_fig,
            2
        )
        # propagate ROI fields if present
        if hasattr(self, "frame_wrinkle_detection"):
            if hasattr(vars, "lower_y_value"):
                self.frame_wrinkle_detection.entry_y_lower_value.set(vars.lower_y_value)
            if hasattr(vars, "upper_y_value"):
                self.frame_wrinkle_detection.entry_y_upper_value.set(vars.upper_y_value)


def load_result_directory(self):
    # Open a file dialog to choose the output folder
    selected_directory = tk.filedialog.askdirectory(title="Ordner für CSV-Export auswählen")
    vars.current_selected_test_directory = selected_directory
    call_output_plots(self.master, directory=vars.current_selected_test_directory)
    update_parameter_field(self.master, dir=selected_directory)


def set_experiment_parameters(self):
    # Create a new Toplevel window
    input_window = ctk.CTkToplevel(self.master)
    input_window.geometry("400x400")
    input_window.title("Enter Parameters")

    # Dictionary to store entries
    entries = {}

    # Parameter names
    params = [
        "line_load",
        "temperature",
        "line_speed",
        "web_tension",
        "unwinder_tension",
        "rewinder_tension"
    ]

    # Create labels and entry fields
    for i, param in enumerate(params):
        label = ctk.CTkLabel(input_window, text=param)
        label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
        entry = ctk.CTkEntry(input_window, width=200)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[param] = entry

    def save_to_file():
        config = ConfigParser()
        config.read('config.ini')
        date = str(config.get('Zeiteinstellungen', 'current_date'))
        formatted_date = date.replace(".", "_")
        exp_nr = int(config.get('Arbeitsverzeichnisse', 'result_number_in_directory')) - 1

        file_directory = vars.output_folder

        file_name = f"{file_directory}/parameters_kalander_{formatted_date}_V{exp_nr}.txt"
        with open(file_name, "w") as file:
            for param, entry in entries.items():
                value = entry.get()
                file.write(f"{param}={value}\n")
        input_window.destroy()  # Close the window after saving

    # Save button
    save_button = ctk.CTkButton(input_window, text="Save Parameters", command=lambda: [save_to_file(),
                                                                                       update_parameter_field(
                                                                                           self.master,
                                                                                           dir=vars.output_folder)])
    save_button.grid(row=len(params), column=0, columnspan=2, pady=20)


def update_parameter_field(self, dir):
    import os
    expected_parameters = {
        "line_load",
        "temperature",
        "line_speed",
        "web_tension",
        "rewinder_tension",
        "unwinder_tension",
    }

    # find the newest parameters file (if many)
    candidates = [f for f in os.listdir(dir) if f.lower().startswith("parameters")]
    if not candidates:
        print("Keine Parameterdatei gefunden")
        self.break_value = 0
        return
    full = [os.path.join(dir, f) for f in candidates]
    full.sort(key=lambda p: os.path.getmtime(p))
    file_path = full[-1]

    parameters = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as loaded_file:
        for line in loaded_file:
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key in expected_parameters:
                    # parse as float; if it happens to be int, it still works
                    try:
                        parameters[key] = float(value)
                    except Exception:
                        # leave missing if unparsable
                        pass

    print("Parameterdatei geladen")
    # only set if present to avoid KeyError
    if "line_load" in parameters:
        vars.exp_line_load = parameters["line_load"]
        self.frame_output_plots.exp_line_load_var.set(vars.exp_line_load)
    if "temperature" in parameters:
        vars.exp_temperature = parameters["temperature"]
        self.frame_output_plots.exp_temperature_var.set(vars.exp_temperature)
    if "line_speed" in parameters:
        vars.exp_web_speed = parameters["line_speed"]
        self.frame_output_plots.exp_line_speed_var.set(vars.exp_web_speed)
    if "web_tension" in parameters:
        vars.exp_web_tension = parameters["web_tension"]
        self.frame_output_plots.exp_web_tension_var.set(vars.exp_web_tension)
    if "unwinder_tension" in parameters:
        vars.exp_unwinder_tension = parameters["unwinder_tension"]
        self.frame_output_plots.exp_unwinder_tension_var.set(vars.exp_unwinder_tension)
    if "rewinder_tension" in parameters:
        vars.exp_rewinder_tension = parameters["rewinder_tension"]
        self.frame_output_plots.exp_rewinder_tension_var.set(vars.exp_rewinder_tension)


def update_plots(self, image_mode):
    vars.image_mode = image_mode
    # make sure a directory is selected
    d = getattr(vars, "current_selected_test_directory", None)
    if d:
        # if call_output_plots is a method on the same class:
        # self.call_output_plots(directory=d)
        # if it's a free function (as in your code previously):
        call_output_plots(self.master, directory=d)



def equalize_row_height(data, exponent):  # takes a pandas Dataframe as "data" and a low number as "exponent"
    # Step 1: Square all values
    squared_df = data ** exponent

    # Step 2: Find the maximum value in the entire DataFrame
    max_value = squared_df.to_numpy().max()

    # Step 3: Normalize each row so that the highest value in that row equals max_value
    def normalize_row(row):
        row_max = row.max()
        if row_max == 0:
            return row  # avoid division by zero
        return row / row_max * max_value

    normalized_df = squared_df.apply(normalize_row, axis=1)

    # Step 4: Prepare for grayscale mapping
    normalized_num = normalized_df.to_numpy()

    # Step 5: Find the smallest non-zero value
    # Find the smallest non-zero value
    nonzero_values = normalized_num[normalized_num > 0]
    min_nonzero = nonzero_values.min() if nonzero_values.size > 0 else 0

    # Step 6: Map to [0, 1] intensity range
    # - max_value maps to 1.0 (white)
    # - min_nonzero maps to ~0.0 (black)
    # - <= 0 maps to exactly 0.0 (black)
    def custom_normalize(val):
        if val <= 0:
            return 0.0
        return (val - min_nonzero) / (max_value - min_nonzero)

    normalized_image = np.vectorize(custom_normalize)(normalized_num)

    # Step 7: Display as grayscale image
    # plt.imshow(normalized_image, cmap='gray', interpolation='nearest')
    # plt.colorbar(label='Grayscale Intensity (custom)')
    # plt.title("Custom Normalized Height Map")
    # plt.show()
    return normalized_image


def find_substrate_jumps(data, threshold):
    # Step 1: Define sampled columns from 10 to 390 (inclusive), every 10 columns
    columns_to_sample = list(range(10, 391, 20))

    # Step 2: Filter out columns that don't exist in the DataFrame
    valid_columns = [col for col in columns_to_sample if col < data.shape[1]]

    # Step 3: Compute derivatives along rows for each valid column
    derivatives = []
    for col in valid_columns:
        column_slice = data.iloc[:, col].to_numpy()
        dy = np.gradient(column_slice)
        abs_dy = np.abs(dy)  # take absolute value
        derivatives.append(abs_dy)

    # Step 4: Average the derivatives across sampled columns
    avg_derivative = np.mean(np.vstack(derivatives), axis=0)
    sum_derivative = np.sum(np.vstack(derivatives), axis=0)

    processed_derivative = sum_derivative
    median_derivative = np.median(processed_derivative)
    threshold = 3 * median_derivative
    # Step 5: Smooth the average derivative using Savitzky–Golay filter
    # window_length = min(len(processed_derivative) // 2 * 2 + 1, 15)  # odd and reasonable size
    # processed_derivative = savgol_filter(processed_derivative, window_length=window_length, polyorder=2)

    # Step 5: Find indices where smoothed sum > 4
    peak_indices = np.where(processed_derivative > threshold)[0]

    # Step 6: Find quiet ranges (length > 100, all values <= 4)
    quiet_ranges = []
    start = None
    for i, val in enumerate(processed_derivative):
        if val <= threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start > 100:
                quiet_ranges.append((start, i - 1))
            start = None
    # Check if final range at end qualifies
    if start is not None and len(processed_derivative) - start > 100:
        quiet_ranges.append((start, len(processed_derivative) - 1))

    # --- Output Results ---
    print(f"Indices where smoothed sum > {threshold}:")
    print(peak_indices.tolist())

    print(f"\nQuiet ranges (length > 100, no peak > {threshold}):")
    for r in quiet_ranges:
        print(f"Rows {r[0]} to {r[1]} (length: {r[1] - r[0] + 1})")

    # Find the longest quiet range
    if quiet_ranges:
        longest_range = max(quiet_ranges, key=lambda r: r[1] - r[0])
        start_row, end_row = longest_range
        if vars.lower_y_value <= start_row and start_row <= vars.upper_y_value:
            vars.lower_y_value = start_row + 5

        if vars.upper_y_value >= end_row and end_row >= vars.lower_y_value:
            vars.upper_y_value = end_row - 5
        print(f"\nLongest quiet range: Rows {start_row} to {end_row} (length: {end_row - start_row + 1})")
    else:
        start_row = end_row = None
        print("\nNo quiet ranges longer than 100 rows found.")

    # Optional: Visualize with peak markers
    # plt.figure(figsize=(8, 4))
    # plt.plot(processed_derivative, label='Smoothed Sum of |Derivatives|', linewidth=2)
    # plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    # plt.scatter(peak_indices, processed_derivative[peak_indices], color='red', s=10, label=f'Peaks > {threshold}')
    # plt.title("Summed Absolute Derivatives with Peaks and Quiet Zones")
    # plt.xlabel("Row Index")
    # plt.ylabel("Sum of |Derivatives|")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return peak_indices.tolist()


# ==== Strict wrinkle/fold filter (add once) ====
import numpy as np
import pandas as pd

"""# Tunables
WR_MIN_LEN_PX    = 20     # min length (px)
WR_MAX_ANGLE_DEG = 35     # |angle| must be <= this (wrinkles ~ horizontal)
EDGE_MARGIN_PX   = 4     # px away from coating edge (x_edge)
AR_MIN           = 2.0    # aspect ratio = length/width
ECC_MIN          = 0.95   # if no AR, use eccentricity fallback
VESSEL_MIN       = 0.00   # min vesselness_mean (if present)"""


def _derive_lengths_from_row(row):
    if "major_axis_length" in row and "minor_axis_length" in row:
        L = float(row["major_axis_length"]);
        W = float(row["minor_axis_length"])
    else:
        bw = float(row.get("bbox_w", 0) or 0)
        bh = float(row.get("bbox_h", 0) or 0)
        L = max(bw, bh)
        W = max(1.0, min(bw, bh))
    return L, W


def apply_wrinkle_strict_filter_old(df_wr, *, y_lo=None, y_hi=None, x_edge=None):
    """Return only long, thin, in-ROI, away-from-edge candidates."""
    if df_wr is None or getattr(df_wr, "empty", True):
        return df_wr

    df = df_wr.copy()

    # Length/width & aspect ratio
    Ls, Ws = [], []
    for _, r in df.iterrows():
        L, W = _derive_lengths_from_row(r)
        Ls.append(L);
        Ws.append(W)
    df["length_px"] = Ls
    df["width_px"] = Ws
    df["ar"] = df["length_px"] / np.maximum(1.0, df["width_px"])

    # Fallbacks when columns are missing
    if "eccentricity" not in df.columns:
        df["eccentricity"] = np.clip((df["ar"] - 1.0) / (df["ar"] + 1.0), 0, 0.999)
    if "vesselness_mean" not in df.columns:
        df["vesselness_mean"] = 1.0

    # Required checks
    keep_len = df["length_px"] >= WR_MIN_LEN_PX
    keep_ang = np.ones(len(df), dtype=bool)
    if "angle_deg" in df.columns and WR_MAX_ANGLE_DEG is not None:
        a = df["angle_deg"].abs()
        keep_ang = (a <= WR_MAX_ANGLE_DEG) | (np.abs(a - 90.0) <= WR_MAX_ANGLE_DEG)

    # ROI (optional)
    keep_roi = np.ones(len(df), dtype=bool)
    if "centroid_y" in df.columns and y_lo is not None and y_hi is not None:
        cy = df["centroid_y"].astype(float).values
        keep_roi = (cy >= float(y_lo)) & (cy <= float(y_hi))

    # Edge margin (optional)
    keep_edge = np.ones(len(df), dtype=bool)
    if "centroid_x" in df.columns and (x_edge is not None) and (WR_EDGE_MARGIN_PX and WR_EDGE_MARGIN_PX > 0):
        cx = df["centroid_x"].astype(float).values
        keep_edge = cx >= (float(x_edge) + float(WR_EDGE_MARGIN_PX))

    # Shape quality: AR or eccentricity
    if {"major_axis_length", "minor_axis_length"}.issubset(df.columns):
        ar = df["major_axis_length"] / np.maximum(1.0, df["minor_axis_length"])
        keep_shape = ar >= WR_AR_MIN
    elif {"bbox_w", "bbox_h"}.issubset(df.columns):
        bw = df["bbox_w"];
        bh = df["bbox_h"]
        ar = np.maximum(bw, bh) / np.maximum(1.0, np.minimum(bw, bh))
        keep_shape = ar >= WR_AR_MIN
    else:
        keep_shape = df["eccentricity"] >= ECC_MIN

    # Vesselness quality (optional)
    keep_vq = df["vesselness_mean"] >= VESSEL_MIN

    # --- DEBUG: see which rule is killing candidates ---
    dbg_total = len(df)
    dbg_len = int((df["length_px"] >= WR_MIN_LEN_PX).sum())
    dbg_ang = int(keep_ang.sum())
    dbg_roi = int(keep_roi.sum())
    dbg_edge = int(keep_edge.sum())

    # shape pass count (AR/ecc)
    if {"major_axis_length", "minor_axis_length"}.issubset(df.columns):
        dbg_shape = int((df["major_axis_length"] / np.maximum(1.0, df["minor_axis_length"]) >= WR_AR_MIN).sum())
    elif {"bbox_w", "bbox_h"}.issubset(df.columns):
        bw = df["bbox_w"];
        bh = df["bbox_h"]
        dbg_shape = int((np.maximum(bw, bh) / np.maximum(1.0, np.minimum(bw, bh)) >= WR_AR_MIN).sum())
    else:
        dbg_shape = int((df["eccentricity"] >= ECC_MIN).sum())

    dbg_vq = int((df.get("vesselness_mean", 1.0) >= VESSEL_MIN).sum())

    print(f"[filter] total={dbg_total} "
          f"len>={WR_MIN_LEN_PX}:{dbg_len} "
          f"angle:{dbg_ang} roi:{dbg_roi} edge:{dbg_edge} "
          f"shape:{dbg_shape} vq:{dbg_vq}")
    # --- END DEBUG ---

    # --- TEMP: stage the gates so we can tighten step by step ---
    # Start with only the strongest two:
    keep = keep_len & keep_shape

    # If counts look ok, uncomment the next line:
    # keep = keep & keep_ang

    # If still ok, uncomment the next line:
    # keep = keep & keep_roi

    # If still ok, uncomment the next line:
    # keep = keep & keep_edge

    # Finally, if you want the vesselness gate:
    # keep = keep & keep_vq

    return df.loc[keep].reset_index(drop=True)


# ===============================================

# ====== FOLD (Längsfalten) STRICT FILTER ======
"""# Tunables for folds (horizontal wide bands)
LF_MIN_LEN_PX    = 100     # folds should be long; raise to 180–220 if still noisy
LF_AR_MIN        = 4.0     # very elongated
LF_MAX_ANGLE_DEG = 12      # near 0° (horizontal)
LF_EDGE_MARGIN   = 0       # px from coating edge (0 = off)"""


def apply_fold_strict_filter(df, *, y_lo=None, y_hi=None, x_edge=None):
    """
    Keep only long, very elongated, ~horizontal bands (Längsfalten).
    Works with either major/minor axis or bbox fallback.
    """
    import numpy as np
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df

    d = df.copy()

    # Length/width from major/minor or bbox fallback
    if {"major_axis_length", "minor_axis_length"}.issubset(d.columns):
        d["length_px"] = d["major_axis_length"].astype(float)
        d["width_px"] = d["minor_axis_length"].astype(float).clip(lower=1.0)
    else:
        bw = d.get("bbox_w", 0).astype(float)
        bh = d.get("bbox_h", 0).astype(float)
        d["length_px"] = np.maximum(bw, bh)
        d["width_px"] = np.minimum(bw, bh).clip(lower=1.0)

    d["ar"] = d["length_px"] / d["width_px"]

    # Angle: prefer horizontal (0°). If missing, pass.
    keep_ang = np.ones(len(d), dtype=bool)
    if "angle_deg" in d.columns:
        a = d["angle_deg"].abs()
        keep_ang = (a <= float(LF_MAX_ANGLE_DEG))

    # ROI & edge gates (optional)
    keep_roi = np.ones(len(d), dtype=bool)
    if "centroid_y" in d.columns and (y_lo is not None) and (y_hi is not None):
        cy = d["centroid_y"].astype(float).values
        keep_roi = (cy >= float(y_lo)) & (cy <= float(y_hi))

    keep_edge = np.ones(len(d), dtype=bool)
    if "centroid_x" in d.columns and (x_edge is not None) and (LF_EDGE_MARGIN > 0):
        cx = d["centroid_x"].astype(float).values
        keep_edge = cx >= (float(x_edge) + float(LF_EDGE_MARGIN))

    # --- TEMP staging for folds ---
    keep = (d["length_px"] >= LF_MIN_LEN_PX) & (d["ar"] >= LF_AR_MIN)

    # then, if ok:
    # keep = keep & keep_ang

    # then, if ok:
    # keep = keep & keep_roi

    # then:
    # keep = keep & keep_edge

    return d.loc[keep].reset_index(drop=True)


def detect_wrinkles_hough(file_path, y_lower, y_upper):
    """
    Multi-pass diagonal detector (manual view):
      1) percentile normalize
      2) background suppress (high-pass)
      3) run several Canny/Hough settings
      4) deduplicate + keep diagonals only

    Returns (df_img, df_props, diag_lines)
    """
    import numpy as np, pandas as pd
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line
    from skimage.filters import gaussian

    # -------- tuning knobs ----------
    ANGLE_MIN, ANGLE_MAX = 20, 85     # diagonals only (deg vs horizontal)
    MIN_LEN_PX = 8                    # accept short segments
    ANG_TOL_DEG = 8                   # dedup angle similarity
    POS_TOL_PX = 15                   # dedup midpoint proximity
    # Canny/Hough passes (sigma, low, high, threshold, line_length, line_gap)
    PASSES = [
        (0.8, 0.02, 0.08, 2,  6, 4),
        (1.0, 0.03, 0.12, 3,  8, 4),
        (1.3, 0.04, 0.14, 4, 10, 5),
    ]
    # --------------------------------

    def _percentile_norm(a):
        a = a.astype("float64")
        a = np.where(np.isfinite(a), a, np.nan)
        p1, p99 = np.nanpercentile(a, (1, 99))
        return np.clip((a - p1) / (p99 - p1 + 1e-9), 0, 1)

    def _highpass(a_norm):
        # remove slow background (mostly horizontal) then re-scale to 0..1
        bg = gaussian(a_norm, sigma=(2.0, 8.0), preserve_range=True)
        hp = a_norm - bg
        p1, p99 = np.nanpercentile(hp[np.isfinite(hp)], (1, 99))
        return np.clip((hp - p1) / (p99 - p1 + 1e-9), 0, 1)

    def _angle_deg(p0, p1):
        dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
        return abs(np.degrees(np.arctan2(dy, dx)))

    def _dedup(lines, ang_tol=ANG_TOL_DEG, pos_tol=POS_TOL_PX):
        kept = []
        for (x0, y0), (x1, y1) in lines:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ang = _angle_deg((x0, y0), (x1, y1))
            L   = float(np.hypot(x1 - x0, y1 - y0))
            if L < MIN_LEN_PX:
                continue
            if not (ANGLE_MIN <= ang <= ANGLE_MAX):
                continue
            dup = False
            for item in kept:
                (kx0, ky0), (kx1, ky1), kcx, kcy, kang = item
                if abs(kang - ang) <= ang_tol and np.hypot(kcx - cx, kcy - cy) <= pos_tol:
                    dup = True
                    break
            if not dup:
                kept.append(((x0, y0), (x1, y1), cx, cy, ang))
        return [ (p0, p1) for (p0, p1, _, _, _) in kept ]

    # ---- load & ROI ----
    df = _read_heightmap_table(file_path)
    if (y_lower is not None) and (y_upper is not None):
        df = crop_to_roi(df, int(y_lower), int(y_upper))
    img = df.to_numpy(dtype=float)

    # ---- normalize & high-pass ----
    img_n  = _percentile_norm(img)
    img_hp = _highpass(img_n)

    # ---- multi-pass edges + Hough ----
    all_lines = []
    for sig, lo, hi, thr, Lmin, Lgap in PASSES:
        edges = canny(img_hp, sigma=sig, low_threshold=lo, high_threshold=hi)
        segs  = probabilistic_hough_line(edges, threshold=thr,
                                         line_length=Lmin, line_gap=Lgap)
        all_lines.extend(segs)

    # ---- deduplicate + diagonal filter ----
    diag_lines = _dedup(all_lines)

    # ---- pack props like regionprops table ----
    rows = []
    for (x0, y0), (x1, y1) in diag_lines:
        L   = float(np.hypot(x1 - x0, y1 - y0))
        ang = _angle_deg((x0, y0), (x1, y1))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rows.append({
            "label": len(rows) + 1,
            "centroid_x": cx, "centroid_y": cy,
            "major_axis_length": L, "minor_axis_length": 1.0,
            "eccentricity": 0.99, "orientation": np.radians(ang),
            "angle_deg": ang,
        })
    df_props = pd.DataFrame(rows)

    print(f"[Hough-multi] passes={len(PASSES)} raw={len(all_lines)} diag_dedup={len(diag_lines)}")
    return df, df_props, diag_lines




# ==============================================


# ---------- STRICT DEBUG HELPERS ----------
def _len_df(df):
    return 0 if (df is None or getattr(df, "empty", True)) else int(len(df))


def _plot_points(ax, df, color, label, s=10, z=50):
    """Scatter centroid points; return count plotted."""
    if df is None or getattr(df, "empty", True):
        return 0
    if "centroid_x" not in df.columns or "centroid_y" not in df.columns:
        return 0
    ax.scatter(
        df["centroid_x"], df["centroid_y"],
        s=s, c=color, edgecolors="black", linewidths=0.3,
        alpha=0.8, zorder=z, label=label
    )
    return len(df)


def _strict_filter(df, *, min_len, min_ar, max_ang, x_edge=None, edge_margin=0):
    """Very light filter: length, aspect ratio, angle, and ‘keep away from coating edge’."""
    import numpy as np
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df

    out = df.copy()

    # derive length/width if not present
    if "length_px" not in out.columns:
        if "major_axis_length" in out.columns:
            out["length_px"] = out["major_axis_length"]
        elif "bbox-3" in out.columns and "bbox-1" in out.columns:
            out["length_px"] = out["bbox-3"] - out["bbox-1"]

    if "width_px" not in out.columns:
        if "minor_axis_length" in out.columns:
            out["width_px"] = out["minor_axis_length"]
        elif "bbox-2" in out.columns and "bbox-0" in out.columns:
            out["width_px"] = out["bbox-2"] - out["bbox-0"]

    # aspect ratio
    if "width_px" in out.columns:
        out["ar"] = out["length_px"] / np.clip(out["width_px"], 1e-6, None)
    else:
        out["ar"] = 0.0

    # angle
    ang = out["angle_deg"].abs() if "angle_deg" in out.columns else 0.0

    keep = (out.get("length_px", 0) >= float(min_len)) & (out["ar"] >= float(min_ar))
    if isinstance(ang, pd.Series):
        keep &= (ang <= float(max_ang))

    # keep away from coating edge if given
    if x_edge is not None and "centroid_x" in out.columns and edge_margin > 0:
        keep &= (out["centroid_x"] <= (float(x_edge) - float(edge_margin)))

    survivors = out.loc[keep].reset_index(drop=True)
    print(f"[strict-filter] in={len(out)}  out={len(survivors)} "
          f"(len≥{min_len}, AR≥{min_ar}, ang≤{max_ang}, edge_margin={edge_margin})")
    return survivors


def _diag_footprint(k=5, flip=False):
    import numpy as np
    k = int(max(3, k))
    m = np.zeros((k, k), dtype=bool)
    if flip:
        m[np.arange(k), np.arange(k)[::-1]] = True
    else:
        m[np.arange(k), np.arange(k)] = True
    return m

def _bridge_diagonal_gaps(mask, iters=1):
    from skimage.morphology import binary_dilation
    se1 = _diag_footprint(5, flip=False)
    se2 = _diag_footprint(5, flip=True)
    out = mask.copy()
    for _ in range(max(1, iters)):
        out = binary_dilation(out, se1) | binary_dilation(out, se2)
    return out

def _gabor_diag_response_multi(img2d):
    import numpy as np
    from skimage.filters import gabor
    angles_deg = (30, 45, 60, 120, 135, 150)
    lambdas = (3, 4, 5, 6)
    resp = np.zeros_like(img2d, dtype=float)
    for a in angles_deg:
        th = np.deg2rad(a)
        for lam in lambdas:
            real, imag = gabor(img2d, frequency=1.0/lam, theta=th)
            resp += np.hypot(real, imag)
    rmin, rmax = np.nanmin(resp), np.nanmax(resp)
    if np.isfinite(rmin) and np.isfinite(rmax) and rmax > rmin:
        resp = (resp - rmin) / (rmax - rmin)
    return resp

# ====== BEGIN DIAGONAL-DETECTOR HELPERS (safe to paste once) ======
# (Adds the exact helpers your editor marks as 'unresolved reference')

# --- imports needed by the helpers (no-op if already imported) ---
try:
    from scipy.ndimage import gaussian_filter, binary_fill_holes
except Exception:
    from scipy.ndimage import gaussian_filter
    # binary_fill_holes may not exist in older SciPy; skimage can do it
    try:
        from skimage.morphology import binary_fill_holes  # type: ignore
    except Exception:
        pass

try:
    from skimage.filters import gabor
except Exception:
    pass

try:
    from skimage.morphology import (
        binary_opening, binary_closing, binary_dilation,
        remove_small_objects, disk, erosion, skeletonize
    )
except Exception:
    pass

try:
    from skimage.measure import label, regionprops_table
except Exception:
    pass

import numpy as np
import pandas as pd

# --- helpers (define only if missing) ---
if "_to_numeric_frame" not in globals():
    def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce all cells to numeric, bad values -> NaN."""
        return df.apply(pd.to_numeric, errors="coerce")

if "_fill_nans_columnwise" not in globals():
    def _fill_nans_columnwise(img2d: np.ndarray) -> np.ndarray:
        """
        Fill NaNs per column with the column median, fallback to global median.
        Keeps overall column offsets and avoids creating bands.
        """
        a = np.array(img2d, dtype=float, copy=True)
        gmed = np.nanmedian(a)
        if not np.isfinite(gmed):
            gmed = 0.0
        for j in range(a.shape[1]):
            col = a[:, j]
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = gmed
            m = ~np.isfinite(col)
            if np.any(m):
                col[m] = med
                a[:, j] = col
        return a

if "_compute_coating_core_mask" not in globals():
    def _compute_coating_core_mask(img2d: np.ndarray, y_lo, y_hi,
                                   *, edge_trim_px=8, dark_q=0.12, erode_px=12):
        """
        Build a 'coating core' mask:
        - removes very dark side bands and the first/last edge_trim_px columns
        - erodes a little so we avoid counting near hard edges
        Returns a boolean mask with shape img2d.shape (True = keep).
        """
        h, w = img2d.shape
        y0 = max(0, int(y_lo) if y_lo is not None else 0)
        y1 = min(h, int(y_hi) if y_hi is not None else h)
        roi = img2d[y0:y1, :]

        col_med = np.nanmedian(roi, axis=0)
        cmin, cmax = np.nanmin(col_med), np.nanmax(col_med)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            col_med_norm = np.zeros_like(col_med, dtype=float)
        else:
            col_med_norm = (col_med - cmin) / (cmax - cmin)

        # columns that are not too dark are considered 'core'
        dark_thresh = np.quantile(col_med_norm, dark_q)
        core_cols = col_med_norm > dark_thresh

        # remove very left/right edges (mechanical shadowing etc.)
        edge = int(max(0, edge_trim_px))
        if edge > 0:
            core_cols[:edge] = False
            core_cols[-edge:] = False

        mask = np.repeat(core_cols[None, :], h, axis=0)

        # close tiny holes, smooth, then erode to step away from edges
        try:
            mask = binary_fill_holes(mask)
        except Exception:
            # if binary_fill_holes is missing, skip – not critical
            pass
        try:
            mask = binary_opening(mask, disk(3))
        except Exception:
            pass
        if erode_px > 0:
            try:
                mask = erosion(mask, disk(erode_px))
            except Exception:
                pass

        return mask
# ====== END DIAGONAL-DETECTOR HELPERS ======

# delete this line
print("[diag] call_wrinkle_detection_manual running from:", __file__)
# or
if __name__ == "__main__":
    print("[diag] call_wrinkle_detection_manual running from:", __file__)








# === Diagonal-only helpers & unified pipeline (FRANGI / GABOR45) ===
def _angle_band_mask(angle_deg_series, center_deg=45.0, tol_deg=12.0):
    """
    Keep only features whose |angle| lies within [center - tol, center + tol],
    AND also accept the mirrored diagonal around 90° (for the other diagonal).
    Angles are expected in degrees within [-90, +90] relative to the horizontal.
    """
    import numpy as np, pandas as pd
    a = np.abs(pd.to_numeric(angle_deg_series, errors="coerce").fillna(0.0).to_numpy())
    lo = max(0.0, float(center_deg) - float(tol_deg))
    hi = min(90.0, float(center_deg) + float(tol_deg))
    keep_main = (a >= lo) & (a <= hi)
    a_m = np.abs(90.0 - a)
    keep_mirr = (a_m >= lo) & (a_m <= hi)
    return keep_main | keep_mirr


def _wrinkle_mask_frangi(a):
    """Ridge enhancement using Frangi; returns binary mask."""
    import numpy as np
    from skimage.filters import frangi
    a = np.where(np.isfinite(a), a, np.nan)
    a = a - np.nanmin(a)
    a = a / (np.nanmax(a) - np.nanmin(a) + 1e-8)
    ridges = frangi(np.nan_to_num(a, nan=0.0), sigmas=(1, 2, 3), black_ridges=False)
    thr = float(np.nanpercentile(ridges, TH_PERCENTILE))
    return (ridges >= thr)


def _wrinkle_mask_gabor45(a):
    """Oriented Gabor at ±45°, combine magnitude; returns binary mask."""
    import numpy as np
    from skimage.filters import gabor
    a = np.where(np.isfinite(a), a, np.nan)
    a = a - np.nanmin(a)
    a = a / (np.nanmax(a) - np.nanmin(a) + 1e-8)
    real1, imag1 = gabor(np.nan_to_num(a, nan=0.0),
                         frequency=float(GABOR_FREQ),
                         theta=np.deg2rad(45.0),
                         sigma_x=GABOR_SIGMA, sigma_y=GABOR_SIGMA)
    real2, imag2 = gabor(np.nan_to_num(a, nan=0.0),
                         frequency=float(GABOR_FREQ),
                         theta=np.deg2rad(135.0),
                         sigma_x=GABOR_SIGMA, sigma_y=GABOR_SIGMA)
    mag = np.hypot(real1, imag1) + np.hypot(real2, imag2)
    thr = float(np.nanpercentile(mag, TH_PERCENTILE))
    return (mag >= thr)


def detect_wrinkles_pipeline(file_path, y_lower, y_upper):
    """
    Unified pipeline selector: FRANGI or GABOR45.
    Returns (df_image_after_roi_mirror, df_props_filtered, mask).
    """
    import numpy as np, pandas as pd
    from skimage.measure import label, regionprops_table

    # 1) Load + ROI
    df = _read_heightmap_table__hotfix(file_path)
    df = crop_to_roi(df, y_lower, y_upper)

    # 2) Heuristic mirror if edge on right (consistency with existing flow)
    try:
        if df.iloc[:, 20].sum() < df.iloc[:, -20].sum():
            df = df.iloc[:, ::-1]
    except Exception:
        pass

    # 3) Mask coating edge to reduce artefacts
    try:
        x_coating_edge = 25
        df_masked = mask_near_edge(df, x_coating_edge, margin_px=int(WR_EDGE_MARGIN_PX))
    except Exception:
        df_masked = df.copy()

    a = df_masked.to_numpy(dtype=float)

    # 4) Choose detector
    pipe = str(WRINKLE_PIPELINE).upper()
    if pipe == "FRANGI":
        mask = _wrinkle_mask_frangi(a)
    else:
        mask = _wrinkle_mask_gabor45(a)

    # 5) Label + measure
    lab = label(mask.astype('uint8'), connectivity=2)
    props = regionprops_table(
        lab, intensity_image=None,
        properties=('label','area','orientation','centroid',
                    'bbox','major_axis_length','minor_axis_length','eccentricity')
    )
    df_props = pd.DataFrame(props).rename(columns={
        'centroid-0':'centroid_y','centroid-1':'centroid_x',
        'bbox-0':'bbox-0','bbox-1':'bbox-1','bbox-2':'bbox-2','bbox-3':'bbox-3'
    })

    # 6) Strict diagonal band-pass + shape filters
    if 'angle_deg' not in df_props.columns:
        import numpy as np
        if 'orientation' in df_props.columns:
            df_props['angle_deg'] = np.degrees(pd.to_numeric(df_props['orientation'], errors='coerce')).abs()
        else:
            df_props['angle_deg'] = 0.0

    ang_keep = _angle_band_mask(df_props['angle_deg'],
                                center_deg=float(WR_DIAG_CENTER_DEG),
                                tol_deg=float(WR_DIAG_TOL_DEG))
    import pandas as pd, numpy as np
    major = pd.to_numeric(df_props.get('major_axis_length', 0), errors='coerce').fillna(0)
    minor = pd.to_numeric(df_props.get('minor_axis_length', 0), errors='coerce').replace(0, 1)
    ar = major / minor
    keep = (
        (major >= float(WR_MIN_LEN_PX)) &
        (ar >= float(WR_AR_MIN)) &
        ang_keep
    )

    # edge margin
    if 'centroid_x' in df_props.columns and WR_EDGE_MARGIN_PX and WR_EDGE_MARGIN_PX > 0:
        cx = pd.to_numeric(df_props['centroid_x'], errors='coerce')
        keep &= (cx >= (25 + int(WR_EDGE_MARGIN_PX)))

    df_props = df_props.loc[keep].reset_index(drop=True)
    return df, df_props, mask




# ================== FINAL HOTFIX: guaranteed diagonal overlay in manual view ==================

# --- robust CSV reader (matches your create_output_plot path) ---
def _read_heightmap_table__hotfix(file_path):
    import pandas as pd
    df_header = pd.read_csv(file_path, nrows=1)
    ncols = len(df_header.columns)
    df = pd.read_csv(file_path, decimal='.', header=1, sep=",", usecols=range(5, max(5, ncols - 5)))
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if df.empty:
        raise ValueError("hotfix reader: file contains no numeric data after cleaning")
    return df

# --- display helper (prevents white or black screens) ---
def _viz_contrast(a, lo=2, hi=98):
    import numpy as np
    a = a.astype("float64")
    a = np.where(np.isfinite(a), a, np.nan)
    p_lo, p_hi = np.nanpercentile(a, (lo, hi))
    out = (a - p_lo) / (p_hi - p_lo + 1e-9)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

# --- diagonal band helper (45°±tol and its 90° mirror) ---
def _angle_band_mask(angle_deg_series, center_deg=None, tol_deg=None):
    import numpy as np, pandas as pd
    c = float(WR_DIAG_CENTER_DEG if center_deg is None else center_deg)   # 45 by default
    t = float(WR_DIAG_TOL_DEG   if tol_deg   is None else tol_deg)        # 22 by default
    a = pd.to_numeric(angle_deg_series, errors="coerce").abs().to_numpy()
    lo = max(0.0, c - t); hi = min(90.0, c + t)
    keep_main = (a >= lo) & (a <= hi)
    keep_mirr = (np.abs(90.0 - a) >= lo) & (np.abs(90.0 - a) <= hi)
    return keep_main | keep_mirr

# --- override strict filter: DIAGONALS ONLY (not near-horizontal) ---
def apply_wrinkle_strict_filter(
    df, *,
    min_len_px=WR_MIN_LEN_PX,
    min_ar=WR_AR_MIN,
    x_edge=25,
    edge_margin_px=WR_EDGE_MARGIN_PX,
    y_lo=None, y_hi=None
):
    import numpy as np, pandas as pd
    if df is None or getattr(df, "empty", True):
        return df

    df = df.copy()

    # angle in degrees (|…|), fallback to 0
    if "angle_deg" not in df.columns:
        if "orientation" in df.columns:
            df["angle_deg"] = np.abs(np.degrees(pd.to_numeric(df["orientation"], errors="coerce")))
        else:
            df["angle_deg"] = 0.0

    # length/AR from major/minor if present; else try bbox; else pass-through
    if {"major_axis_length", "minor_axis_length"}.issubset(df.columns):
        num = pd.to_numeric(df["major_axis_length"], errors="coerce")
        den = pd.to_numeric(df["minor_axis_length"], errors="coerce").replace(0, np.nan)
        df["length_px"] = num
        df["AR"] = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif {"bbox-3","bbox-1","bbox-2","bbox-0"}.issubset(df.columns):
        bw = pd.to_numeric(df["bbox-3"], errors="coerce") - pd.to_numeric(df["bbox-1"], errors="coerce")
        bh = pd.to_numeric(df["bbox-2"], errors="coerce") - pd.to_numeric(df["bbox-0"], errors="coerce")
        L = np.maximum(bw, bh)
        W = np.maximum(1.0, np.minimum(bw, bh))
        df["length_px"] = L
        df["AR"] = (L / W).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        df["length_px"] = pd.to_numeric(df.get("length_px", 0), errors="coerce").fillna(0.0)
        df["AR"] = pd.to_numeric(df.get("ar", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # diagonal band (uses WR_DIAG_CENTER_DEG / WR_DIAG_TOL_DEG via helper)
    keep = _angle_band_mask(df["angle_deg"])
    keep &= (df["length_px"] >= float(min_len_px))
    keep &= (df["AR"]        >= float(min_ar))

    # ROI y filter (optional)
    if y_lo is not None and y_hi is not None and "centroid_y" in df.columns:
        cy = pd.to_numeric(df["centroid_y"], errors="coerce")
        keep &= (cy >= int(y_lo)) & (cy <= int(y_hi))

    # keep away from coating edge (optional)
    if x_edge is not None and "centroid_x" in df.columns and int(edge_margin_px) > 0:
        cx = pd.to_numeric(df["centroid_x"], errors="coerce")
        keep &= (cx >= (int(x_edge) + int(edge_margin_px)))

    return df.loc[keep].reset_index(drop=True)

# --- quick props helper for Hough segments ---
def _props_from_lines(lines):
    import numpy as np, pandas as pd

    def _as_xy(pt):
        # Accept either (x,y) or (row,col); pick ordering where x is horizontal
        a, b = map(float, pt)
        # If values are clearly "row,col" (y tends to be larger on tall images), we don't know H/W here,
        # so just return both possibilities consistently: prefer (x,y) ordering as given.
        return a, b

    rows = []
    for (p0, p1) in lines:
        x0, y0 = _as_xy(p0)
        x1, y1 = _as_xy(p1)
        dx, dy = (x1 - x0), (y1 - y0)

        L = float(np.hypot(dx, dy))
        if not np.isfinite(L) or L <= 0:
            continue

        ang = float(np.degrees(np.arctan2(dy, dx)))
        ang = abs(ang)
        if ang > 90.0:  # fold into [0,90]
            ang = 180.0 - ang

        rows.append({
            "label": len(rows) + 1,
            "centroid_x": (x0 + x1) / 2.0,
            "centroid_y": (y0 + y1) / 2.0,
            "major_axis_length": L,
            "minor_axis_length": 1.0,
            "eccentricity": 0.99,
            "orientation": float(np.radians(ang)),
            "angle_deg": ang,
        })

    df = pd.DataFrame(rows)
    # Ensure numeric dtypes
    for c in ("centroid_x","centroid_y","major_axis_length","minor_axis_length","eccentricity","orientation","angle_deg"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


# --- detector: Hough (schedule) → Sobel orientation fallback ---
def _detect_diagonals_auto(img01, y_lower, y_upper):
    import numpy as np, pandas as pd
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line
    from skimage.filters import gaussian, sobel_h, sobel_v
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.measure import label, regionprops_table

    H, W = img01.shape[:2]

    # 1) Hough schedules (guarantee some lines if edges exist)
    for (low, high, thr, L, gap) in [
        (0.05, 0.15,  8, 12, 6),
        (0.08, 0.22, 12, 18, 8),
        (0.12, 0.28, 18, 24, 8),
    ]:
        edges = canny(gaussian(img01, 1.0, preserve_range=True),
                      sigma=1.2, low_threshold=low, high_threshold=high)
        lines = probabilistic_hough_line(edges, threshold=thr,
                                         line_length=L, line_gap=gap)
        if lines:
            dfp = _props_from_lines(lines)
            if not dfp.empty:
                dfp = apply_wrinkle_strict_filter(dfp, y_lo=y_lower, y_hi=y_upper)
                if not dfp.empty:
                    return dfp, {"mode": "hough", "lines": lines, "edges": edges}

    # 2) Sobel orientation fallback (diagonal gradient + skeleton)
    gx, gy = sobel_h(img01), sobel_v(img01)
    mag = np.hypot(gx, gy)
    # absolute angle w.r.t. horizontal in [0..90]
    ang = np.abs(np.degrees(np.arctan2(gy, gx)))
    ang = np.where(ang > 90.0, 180.0 - ang, ang)  # fold to [0,90]

    for p in (80, 75, 70, 65, 60, 55, 50, 45):
        hi = np.nanpercentile(mag, p)
        strong = mag >= hi  # 2D boolean (H, W)

        # --- FIX: build diagonal-band mask with matching 2D shape ---
        # _angle_band_mask expects a 1D series; give it flattened angles,
        # then reshape back to (H, W)
        band_1d = _angle_band_mask(pd.Series(ang.ravel()))
        band = np.asarray(band_1d, dtype=bool).reshape(H, W)

        mask = strong & band
        skel = skeletonize(mask)
        skel = remove_small_objects(skel, min_size=max(int(WR_MIN_LEN_PX // 2), 8))

        lab = label(skel, connectivity=2)
        props = regionprops_table(
            lab,
            properties=('label', 'area', 'orientation', 'centroid',
                        'major_axis_length', 'minor_axis_length', 'eccentricity')
        )
        dfp = pd.DataFrame(props).rename(columns={'centroid-0':'centroid_y','centroid-1':'centroid_x'})
        if not dfp.empty:
            # make angle_deg explicitly [0,90]
            if 'orientation' in dfp.columns:
                a = np.abs(np.degrees(pd.to_numeric(dfp['orientation'], errors='coerce')))
                a = np.where(a > 90.0, 180.0 - a, a)
                dfp['angle_deg'] = a
            else:
                dfp['angle_deg'] = 0.0

            dfp = apply_wrinkle_strict_filter(dfp, y_lo=y_lower, y_hi=y_upper)
            if not dfp.empty:
                return dfp, {"mode": "sobel", "skel": skel}

    return pd.DataFrame(), {"mode": "none"}

# --- manual entrypoint: read → crop → detect → draw (always shows something if present) ---

def add_terminal_ok(terminal, text):
    add_terminal_line(terminal, f"OK: {text}")

def add_terminal_warn(terminal, text):
    add_terminal_line(terminal, f"WARN: {text}")

def add_terminal_error(terminal, text):
    add_terminal_line(terminal, f"ERROR: {text}")
# ================= END TERMINAL HELPER SHIMS =================


# --- Backwards compatibility aliases ---
def detect_wrinkles_tophat_thesis(file_path, y_lower=None, y_upper=None):
    """
    Backwards-compat alias expected by some callers.
    Calls the thesis Top-Hat pipeline and returns its outputs.
    """
    return detect_wrinkles_pipeline(file_path, y_lower, y_upper)


import numpy as np

def _as_xy(pt, W, H):
    """Return (x,y) even if pt was (row,col)."""
    a, b = map(float, pt)
    return (a, b) if (0 <= a < W and 0 <= b < H) else (b, a)

def _filter_diagonal_slivers(lines, x_ref, W, H,
                             angle_c=-62.0, ang_tol=7.0,
                             start_tol=6.0, min_len=25, max_len=220,
                             min_sep_y=22):
    """Keep only nice diagonal slivers that start near x_ref and go down-right."""
    candidates = []
    for seg in lines:
        (x0, y0) = _as_xy(seg[0], W, H)
        (x1, y1) = _as_xy(seg[1], W, H)

        # pick start = endpoint closest to the straight ref line
        d0 = abs(x0 - x_ref); d1 = abs(x1 - x_ref)
        if d0 <= d1: xs, ys, xe, ye = x0, y0, x1, y1
        else:        xs, ys, xe, ye = x1, y1, x0, y0

        # start must be near the ref line; end must be to the right of it
        if abs(xs - x_ref) > start_tol:
            continue
        # basic length & direction
        L = float(np.hypot(xe - xs, ye - ys))
        if L < min_len or L > max_len:
            continue
        if (xe - x_ref) < 0:   # shouldn’t go left
            continue

        ang = float(np.degrees(np.arctan2(ye - ys, xe - xs)))
        if abs(ang - angle_c) > ang_tol:
            continue

        candidates.append(((xs, ys), (xe, ye), L, ang))

    # non-max suppression in y: keep longer sliver within a vertical band
    candidates.sort(key=lambda k: k[0][1])  # by start y
    kept = []
    for s in candidates:
        if kept and (s[0][1] - kept[-1][0][1]) < min_sep_y:
            if s[2] > kept[-1][2]:
                kept[-1] = s
        else:
            kept.append(s)
    return kept






def detect_wrinkles_edgecast(file_path, y_lower, y_upper):
    """
    Thesis-like detector:
      • normalize + high-pass
      • ensure bright coating on the RIGHT
      • estimate coating edge polyline (x_edge[y])
      • for y in [y_lower..y_upper] step STEP_Y: cast a short ray from the edge,
        find a strong peak along that direction -> short diagonal segment
    Returns (df_img, df_props, lines)
    """
    import numpy as np, pandas as pd
    from skimage.filters import gaussian


    # ---------- knobs (tune to taste) ----------
    ANGLE_DEG = 60.0  # y grows downward -> down-right is +60°
    RAY_LEN = 100  # longer ray so it can reach the wrinkle
    STEP_Y = 24  # slightly denser sampling along Y
    START_SKIP = 10  # skip the halo right at the edge
    THRESH_PCTL = 90.0  # lower = more sensitive (was 97)
    MIN_SEP_Y = 20  # non-max suppression window
    # -------------------------------------------

    # 1) load + ROI
    df = _read_heightmap_table(file_path)
    if (y_lower is not None) and (y_upper is not None):
        df = crop_to_roi(df, int(y_lower), int(y_upper))
    img = df.to_numpy(dtype=float)
    H, W = img.shape[:2]

    # 2) normalize + high-pass (remove broad horizontal banding)
    finite = np.isfinite(img)
    p1, p99 = np.nanpercentile(img[finite], (1, 99))
    im_n = np.clip((img - p1) / (p99 - p1 + 1e-9), 0, 1)
    bg = gaussian(im_n, sigma=(2.0, 12.0), preserve_range=True)
    hp = im_n - bg
    # rescale HP to 0..1 for stable threshold
    pf1, pf99 = np.nanpercentile(hp[finite], (1, 99))
    hp_n = np.clip((hp - pf1) / (pf99 - pf1 + 1e-9), 0, 1)

    # 3) ensure bright coating on the RIGHT
    try:
        if np.nanmean(im_n[:, :50]) > np.nanmean(im_n[:, -50:]):
            im_n = im_n[:, ::-1]
            hp_n = hp_n[:, ::-1]
            img  = img[:, ::-1]
            df   = pd.DataFrame(img)  # keep df consistent
    except Exception:
        pass


    # 4) estimate coating edge (smooth blue polyline)
    x_edge = _estimate_coating_edge_polyline(im_n)

    # --- straight reference like thesis (left boundary near the edge) ---
    X_REF_PCTL = 10.0  # try 10..25; smaller -> more to the left
    x_ref = float(np.nanpercentile(x_edge, X_REF_PCTL))
    x_ref = np.clip(x_ref, W * 0.10, W - 5.0)

    # 5) ray sampler
    def _bilinear(a, x, y):
        x0 = int(np.floor(x)); y0 = int(np.floor(y))
        x1 = x0 + 1;          y1 = y0 + 1
        if x0 < 0 or y0 < 0 or x1 >= a.shape[1] or y1 >= a.shape[0]:
            return np.nan
        fx = x - x0; fy = y - y0
        v00 = a[y0, x0]; v10 = a[y0, x1]
        v01 = a[y1, x0]; v11 = a[y1, x1]
        return (v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy)

    theta = np.deg2rad(ANGLE_DEG)
    dx, dy = float(np.cos(theta)), float(np.sin(theta))

    # dynamic threshold from whole HP image
    thr = np.nanpercentile(hp_n[finite], THRESH_PCTL)

    cand = []   # ( (x0,y0),(x1,y1), peak, yref )
    for y in range(0, H, STEP_Y):
        # use straight reference for all rays so the cyan slivers start on a vertical line
        xe = x_ref
        xs, ys = xe, float(y)
        # cast RAY_LEN px along the ray
        vals = []
        coords = []
        for t in range(START_SKIP, RAY_LEN):
            xt = xs + t*dx
            yt = ys + t*dy
            if xt < 0 or xt >= W or yt < 0 or yt >= H:
                break
            v = _bilinear(hp_n, xt, yt)
            vals.append(v); coords.append((xt, yt))
        if not vals:
            continue
        vals = np.array(vals, dtype=float)
        if not np.isfinite(vals).any():
            continue
        t_peak = int(np.nanargmax(vals))
        v_peak = float(vals[t_peak]) if np.isfinite(vals[t_peak]) else 0.0
        if v_peak < thr:
            continue  # not strong enough

        # build short segment from edge to the detected peak (consistent, tidy)
        x1, y1 = coords[t_peak]
        cand.append(((xs, ys), (x1, y1), v_peak, float(y)))

    # 6) non-maximum suppression along Y
    cand.sort(key=lambda c: c[3])
    kept = []
    for seg in cand:
        yref = seg[3]
        if all(abs(yref - r[3]) >= MIN_SEP_Y for r in kept):
            kept.append(seg)
        else:
            # replace if stronger
            idx = None; best_gap = 1e9
            for i, r in enumerate(kept):
                gap = abs(yref - r[3])
                if gap < MIN_SEP_Y and gap < best_gap:
                    best_gap, idx = gap, i
            if idx is not None and seg[2] > kept[idx][2]:
                kept[idx] = seg

    # 7) pack results
    lines = [ (p0, p1) for (p0, p1, *_ ) in kept ]
    rows = []
    for i, ((x0,y0),(x1,y1)) in enumerate(lines, start=1):
        L = float(np.hypot(x1-x0, y1-y0))
        ang = abs(np.degrees(np.arctan2(y1-y0, x1-x0)))
        cx, cy = (x0+x1)/2.0, (y0+y1)/2.0
        rows.append({
            "label": i,
            "centroid_x": cx, "centroid_y": cy,
            "major_axis_length": L, "minor_axis_length": 1.0,
            "eccentricity": 0.99,
            "orientation": np.radians(ang),
            "angle_deg": ang,
        })

        print(f"[edgecast] kept={len(lines)}  angle={ANGLE_DEG} ray={RAY_LEN} thr%={THRESH_PCTL}")

    df_props = pd.DataFrame(rows)
    return df, df_props, lines






# --- helpers: estimate coating edge & prune lines --------------------------------

def _estimate_coating_edge_polyline(img):
    """
    Robust coating-edge tracker.
    Returns: x_edge[y]  (1D float array, length = image height)
    """
    import numpy as np
    from skimage.filters import gaussian

    a = img.astype("float64")
    H, W = a.shape[:2]

    finite = np.isfinite(a)
    if not finite.any():
        return np.full(H, W - 5.0)

    p5, p95 = np.nanpercentile(a[finite], (5, 95))
    an = np.clip((a - p5) / (p95 - p5 + 1e-9), 0.0, 1.0)
    g = gaussian(an, sigma=(1.0, 3.0), preserve_range=True)

    gx = np.gradient(g, axis=1)
    gx[~np.isfinite(gx)] = -np.inf

    x_start = int(W * 0.25)
    band = gx[:, x_start:]
    idx = np.argmax(band, axis=1)
    x_edge = x_start + idx

    pos = np.maximum(band, 0.0)
    if (pos > 0).any():
        thr = np.nanpercentile(pos[pos > 0], 75.0)
        weak = pos[np.arange(H), idx] < 0.5 * thr
        if np.any(weak):
            ridx = np.argmax(g[:, x_start:], axis=1)
            x_edge[weak] = x_start + ridx[weak]

    try:
        from scipy.signal import savgol_filter
        win = max(21, (H // 60) * 2 + 1)
        x_edge = savgol_filter(x_edge.astype("float64"), win, 3, mode="interp")
    except Exception:
        win = max(21, (H // 60) * 2 + 1)
        k = np.ones(win, dtype="float64") / win
        x_edge = np.convolve(x_edge.astype("float64"), k, mode="same")

    x_edge = np.clip(x_edge, int(W * 0.10), W - 5.0)
    return x_edge



def _clean_wrinkle_lines(img, lines,
                         x_edge,
                         ang_min=25, ang_max=75,
                         edge_tol_px=14,
                         y_spacing_px=28,
                         min_len_px=14):
    """
    Keep only diagonals that END at the coating edge (right-most endpoint near edge),
    point inward (toward smaller x), are long enough, and are spaced along Y.
    Returns pruned_lines ([(p0, p1), ...]).
    """
    import numpy as np

    H, W = img.shape[:2]
    keep = []
    for (x0, y0), (x1, y1) in lines:
        dx, dy = (x1 - x0), (y1 - y0)
        L = float(np.hypot(dx, dy))
        if L < min_len_px:
            continue

        ang = abs(np.degrees(np.arctan2(dy, dx)))  # vs horizontal
        if not (ang_min <= ang <= ang_max):
            continue

        # right-most endpoint
        if x0 >= x1:
            xr, yr = x0, y0
            xl = x1
        else:
            xr, yr = x1, y1
            xl = x0

        # clamp row & compare to edge
        yi = int(np.clip(round(yr), 0, H - 1))
        xe = float(x_edge[yi])

        # endpoint must sit AT the edge (a little to the right of it is OK),
        # and the other endpoint must be inside (left of the edge).
        if not (0 <= xr - xe <= edge_tol_px):
            continue
        if not (xl < xe - 3):
            continue

        keep.append(((x0, y0), (x1, y1), L, yr))

    # non-maximum suppression along Y: keep the longest per y-window
    keep.sort(key=lambda t: t[3])  # by yr
    pruned = []
    for seg in keep:
        _, _, L, yr = seg
        if all(abs(yr - yr2) >= y_spacing_px for *_, yr2 in pruned):
            pruned.append((*seg[:-1], yr))  # store with yr at end
        else:
            # replace shorter one in the window
            idx = None
            best_gap = 1e9
            for i, (*_, yr2) in enumerate(pruned):
                gap = abs(yr - yr2)
                if gap < y_spacing_px and gap < best_gap:
                    best_gap, idx = gap, i
            if idx is not None and L > pruned[idx][2]:
                pruned[idx] = (*seg[:-1], yr)

    # strip extra fields
    return [ (p0, p1) for (p0, p1, *_ ) in pruned ]





def ALT_call_wrinkle_detection_manual(file_path, canvas, figure, y_lower, y_upper):
    """
    Manual wrinkle evaluation (diagonal, thesis-style):
      - robust read
      - edge find in a column ROI (no mirror here)
      - preview standardized so edge is on the LEFT (remember flip state)
      - equalize rows for preview background
      - run SOBEL band detector on the FULL image (permissive first pass)
      - draw edge, edge-band, and skeleton (mirrored if preview was flipped)
      - print strong debug counters
    """
    import numpy as np
    import pandas as pd
    import wrinkle_detection_new as wrinkles
    import wrinkle_aux_funcs as wrinkle_aux

    # ------------------ 0) robust read ------------------
    def _read_heightmap_any(path):
        df_try = None
        try:
            df_try = pd.read_csv(path, header=None, sep=";", decimal=",")
        except Exception:
            pass
        if df_try is None or isinstance(df_try.iloc[0, 0], str):
            try:
                df_try = pd.read_csv(path, header=None, sep=",", decimal=".")
            except Exception:
                pass
        if df_try is None or isinstance(df_try.iloc[0, 0], str):
            df_try = pd.read_excel(path, header=None)
        df_try = df_try.apply(pd.to_numeric, errors="coerce")
        med = np.nanmedian(df_try.values)
        return df_try.fillna(med)

    df = _read_heightmap_any(file_path)

    # ------------------ 1) Y-ROI & column ROI ------------------
    y0, y1 = int(y_lower), int(y_upper)
    y0 = max(0, min(y0, df.shape[0] - 1))
    y1 = max(y0 + 1, min(int(y_upper), df.shape[0]))
    df_cleaned = df.iloc[y0:y1, :]

    c0, c1 = 25, min(380, df_cleaned.shape[1])
    if (c1 - c0) < 20:
        c0, c1 = 0, df_cleaned.shape[1]
    filtered_df = df_cleaned.iloc[:, c0:c1].reset_index(drop=True)

    # ------------------ 2) detect edge in ROI ------------------
    scnd_deriv, height_map, x_edge_roi, runtime_edge = \
        wrinkle_aux.detect_coating_edge(filtered_df, show_plot=False)

    x_edge_float = float(x_edge_roi)
    x_edge_int   = int(round(x_edge_float))

    x_edge_full_float = c0 + x_edge_float
    x_edge_full_int   = int(round(x_edge_full_float))

    try:
        import globals as vars
        vars.last_edge_full_int   = x_edge_full_int
        vars.last_edge_full_float = x_edge_full_float
        vars.last_edge_roi_int    = x_edge_int
        vars.last_edge_roi_float  = x_edge_float
    except Exception:
        pass

    # ------------------ 3) standardize preview (edge on LEFT) ------------------
    preview_flipped = False
    ncols = filtered_df.shape[1]
    if x_edge_float > (ncols / 2.0):
        filtered_df = filtered_df.iloc[:, ::-1].reset_index(drop=True)
        x_edge_float = ncols - 1 - x_edge_float
        x_edge_int   = int(round(x_edge_float))
        preview_flipped = True

    # ------------------ 4) equalize rows for the preview bg ------------------
    eq_gamma, _, _ = wrinkle_aux.equalize_rows(filtered_df, x_edge_int)
    bg = eq_gamma.values
    H, W = bg.shape

    # ------------------ 4b) choose coating side ------------------
    try:
        left_slice  = df_cleaned.iloc[:, max(0, x_edge_full_int - 30):x_edge_full_int]
        right_slice = df_cleaned.iloc[:, x_edge_full_int:min(df_cleaned.shape[1], x_edge_full_int + 30)]
        mean_left  = float(left_slice.values.mean())  if left_slice.shape[1]  > 3 else np.nan
        mean_right = float(right_slice.values.mean()) if right_slice.shape[1] > 3 else np.nan
        coating_side = "right" if not np.isnan(mean_left) and not np.isnan(mean_right) and (mean_right >= mean_left) else "left"
    except Exception:
        coating_side = "right"

    # ------------------ 5) run SOBEL detector (permissive) ------------------
    # call without band restriction
    out = wrinkles.detect_wrinkles_sobel_band(
        path=file_path,
        y_lower=y0, y_upper=y1,
        px_per_mm=10.0,
        edge_band_px=None,  # or 0
        sobel_sigma=1.2,
        thr_percentile=80,
        min_len_px=8,
        angle_center_deg=45.0,
        angle_tol_deg=45,
        small_remove_px=0,
        coating_side=coating_side,
    )

    # --- strong debug counters
    print("[DBG] bw_px =", out.get("_dbg_bw_pixels", -1),
          " skel_px =", out.get("_dbg_skel_pixels", -1),
          " n_stats =", 0 if out.get("stats") is None else len(out["stats"]))

    sk_full = out.get("mask_skel", None)
    sk_total = int(sk_full.sum()) if sk_full is not None else -1
    sk_roi   = int(sk_full[y0:y1, c0:c1].sum()) if sk_full is not None else -1
    print(f"[DBG] skel_total={sk_total}  skel_in_roi={sk_roi}  flipped_preview={preview_flipped}  side={coating_side}")

    # ------------------ 6) draw ------------------
    figure.clear()
    ax = figure.add_subplot(111)
    ax.imshow(bg, cmap="gray", aspect="auto", interpolation="nearest", origin="upper")

    ax.axvline(x_edge_float, color="#0a2f6b", linewidth=2.0, label="Beschichtungskante")

    band_half = 180 // 2
    if coating_side == "right":
        b_left, b_right = x_edge_float, x_edge_float + band_half
    else:
        b_left, b_right = x_edge_float - band_half, x_edge_float
    b_left  = max(0.0, b_left)
    b_right = min(float(W - 1), b_right)
    if b_right > b_left:
        ax.axvspan(b_left, b_right, color="orange", alpha=0.22, label="Edge-Band")

    showed_skel = False
    if sk_full is not None:
        sk_view = sk_full[y0:y1, c0:c1].copy()
        if preview_flipped:
            sk_view = sk_view[:, ::-1]
        if np.any(sk_view):
            ax.contour(sk_view, levels=[0.5], colors="cyan", linewidths=0.6)
            showed_skel = True

    if not showed_skel:
        ax.text(6, 12, "no skel in ROI", color="red", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    ax.set_xlim(0, W - 1)
    ax.set_title(f"Wrinkledetektion (SOBEL) – side={coating_side}")
    ax.set_xlabel("Spalten"); ax.set_ylabel("Zeilen")
    try:
        ax.legend(loc="upper right", fontsize=8)
    except Exception:
        pass
    canvas.draw()

    # ------------------ 7) return stats ------------------
    stats_df = out.get("stats", None)
    if stats_df is None:
        stats_df = pd.DataFrame([])
    return stats_df





def call_wrinkle_detection_manual(file_path, canvas, figure, y_lower, y_upper):
    """
    Manual, thesis-style diagonal wrinkle detection:
    - robust CSV load
    - mirror if edge is on right
    - ROI 25:380
    - detect coating edge -> standardize to LEFT
    - equalize rows for display
    - run Top-Hat + edge-band detector on the ORIGINAL file (full frame)
    - draw bg + edge line + skeleton + endpoints
    - return stats DataFrame (length_px, length_mm, angle_deg)
    """
    import os
    import numpy as np
    import pandas as pd
    from skimage.morphology import disk, binary_dilation
    import wrinkle_aux_funcs as wrinkle_aux
    import wrinkle_detection_new as wrn

    print("[MANUAL] start:", file_path, "roi:", y_lower, y_upper)
    file_name = os.path.basename(file_path)[:-4]

    # --- 0) Robust CSV load (comma/semicolon; dot/comma decimals) ---
    try:
        df = pd.read_csv(file_path, decimal=',', header=None, sep=';')
        if isinstance(df.iloc[0, 0], str):
            raise ValueError
    except Exception:
        df = pd.read_csv(file_path, decimal='.', header=None, sep=',')
    df = df.apply(pd.to_numeric, errors="coerce")

    arr = df.to_numpy()
    if not np.isfinite(arr).any():
        raise ValueError("Loaded table is all NaN. Check separators/decimals.")

    med = np.nanmedian(arr[np.isfinite(arr)])
    arr = np.where(np.isfinite(arr), arr, med)
    df = pd.DataFrame(arr)

    H, W = df.shape
    # --- 1) sanitize Y range ---
    y0 = int(max(0, min(int(y_lower), int(y_upper))))
    y1 = int(min(H, max(int(y_lower), int(y_upper))))
    if y1 - y0 < 10:
        y0 = max(0, y0 - 50)
        y1 = min(H, y1 + 50)
    df_cleaned = df.iloc[y0:y1, :]

    # --- 2) Mirror BEFORE ROI if edge looks right ---
    if df_cleaned.shape[1] >= 40 and df_cleaned.iloc[:, 20].sum() < df_cleaned.iloc[:, -20].sum():
        print("[MANUAL] edge appears right → mirror")
        df_cleaned = df_cleaned.iloc[:, ::-1]

    # --- 3) ROI selection (safe bounds) ---
    c0, c1 = 25, min(380, df_cleaned.shape[1])
    if c1 - c0 < 20:
        c0, c1 = 0, df_cleaned.shape[1]
    filtered_df = df_cleaned.iloc[:, c0:c1].reset_index(drop=True)

    # --- 4) Edge detection on ROI ---
    try:
        _, _, x_edge_raw, _ = wrinkle_aux.detect_coating_edge(filtered_df, show_plot=False)
    except Exception as e:
        print("[WARN] detect_coating_edge failed:", e)
        x_edge_raw = filtered_df.shape[1] // 6

    # Keep edge on LEFT in the *displayed* ROI
    ncols = filtered_df.shape[1]
    if float(x_edge_raw) > (ncols / 2.0):
        filtered_df = filtered_df.iloc[:, ::-1].reset_index(drop=True)
        x_edge_raw = ncols - 1 - float(x_edge_raw)
        print(f"[MANUAL] standardized edge to LEFT: x_edge={x_edge_raw:.1f}")

    # --- use BOTH: float (plot) and int (index) ---
    x_edge_float = float(x_edge_raw)
    x_edge_int   = max(0, min(int(round(x_edge_float)), filtered_df.shape[1] - 1))
    print(f"[MANUAL] edge float={x_edge_float:.2f} → int={x_edge_int}")

    # --- 5) Equalize rows around edge for nicer display background ---
    eq_gamma, eq_sigmoid, eq_exp = wrinkle_aux.equalize_rows(filtered_df, x_edge_int)
    bg = eq_gamma.values  # background to imshow

    # --- 6) Run Top-Hat + edge-band detector on the ORIGINAL file (full frame) ---
    #     (It builds masks from the full image using y0:y1 crop info.)
    out = wrn.detect_wrinkles_tophat_edgeband(
        path=file_path,
        y_lower=y0, y_upper=y1,
        px_per_mm=10.0,
        edge_band_px=40,
        tophat_rad=5,
        min_len_px=100,
        endpoint_dist_px=15,
        angle_center_deg=45.0,   # diagonal
        angle_tol_deg=12,        # tighten/loosen as needed
        small_remove_px=64,
        coating_side="right"     # change to "left" if your coating is left of the edge
    )

    # view the returned full-size skeleton in our display ROI (y0:y1, c0:c1)
    sk_full = out.get("mask_skel", None)
    if sk_full is None:
        sk_full = np.zeros((H, W), dtype=bool)
    sk_view = sk_full[y0:y1, c0:c1]

    # Thin mask for measurement overlay (avoid huge width)
    mask_wrinkles = binary_dilation(sk_view, footprint=disk(1))

    # --- 7) Draw (always show bg + edge line) ---
    figure.clear()
    ax = figure.add_subplot(111)
    ax.imshow(bg, cmap='gray', aspect='auto', interpolation='nearest', origin='upper')
    ax.axvline(x_edge_float, color='#0a2f6b', linewidth=2.0, label="Beschichtungskante")

    ys2, xs2 = np.where(sk_view)
    if ys2.size:
        ax.scatter(xs2, ys2, s=1, c="cyan", alpha=0.9, label="Skelett")

    endpoints = out.get("wrinkle_points", [])
    if endpoints:
        yy, xx = zip(*endpoints)
        yy = np.asarray(yy) - y0
        xx = np.asarray(xx) - c0
        ok = (yy >= 0) & (yy < bg.shape[0]) & (xx >= 0) & (xx < bg.shape[1])
        if ok.any():
            ax.scatter(xx[ok], yy[ok], s=25, c="red", label="Endpunkte")

    ax.set_title("Wrinkledetektion (AS) – TOPHAT – Beschichtungskante")
    ax.set_xlabel("Spalten"); ax.set_ylabel("Zeilen")
    ax.legend(loc="upper right", fontsize=8)
    canvas.draw()

    # --- 8) Optional: compute measurement table using INT edge index ---
    try:
        mask_lf, _ = wrinkle_aux.laengsfalten_detection_top_hat(pd.DataFrame(bg), show_plot=False)
    except Exception:
        mask_lf = np.zeros_like(bg, dtype=bool)

    try:
        table, _ = wrinkle_aux.measure(
            pd.DataFrame(bg),
            mask_wrinkles,
            x_edge_int,           # << INT for indexing!
            mask_lf,
            file_name=f"{file_name}_top_hat",
            save_results=True,
            show_plot=True,
            show_unknown_defects=False
        )
    except Exception as e:
        print("[WARN] measurement skipped:", e)
        table = out.get("stats", pd.DataFrame())

    # Prefer returning the detector's stats if available; otherwise measurement table
    stats = out.get("stats", None)
    if stats is not None and not getattr(stats, "empty", True):
        return stats.reset_index(drop=True)
    return table.reset_index(drop=True) if table is not None else pd.DataFrame([])
