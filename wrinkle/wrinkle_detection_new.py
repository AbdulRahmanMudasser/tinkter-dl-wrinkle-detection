import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, distance_transform_edt
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt


# --- Non-blocking core for GUI use: returns (df, ridge, skeleton) and DOES NOT plt.show() ---
# --- Non-blocking core for GUI use: returns (df, ridge, skeleton), no plt.show() ---
def frangi_core(data, min_size_frac=1e-6, angle_exclude=0):
    import numpy as np
    import pandas as pd
    from scipy.ndimage import gaussian_filter
    from skimage.filters import frangi, threshold_otsu
    from skimage.morphology import remove_small_objects, skeletonize
    from skimage.measure import label, regionprops

    data = np.asarray(data, dtype=float)

    # Fill NaNs for stable filtering
    mask_finite = np.isfinite(data)
    if not mask_finite.any():
        return pd.DataFrame([]), data, np.zeros_like(data, dtype=bool)
    median_val = np.nanmedian(data[mask_finite])
    data = np.where(mask_finite, data, median_val)

    # Smooth + ridge enhance
    blurred = gaussian_filter(data, sigma=1)
    ridge = frangi(blurred)

    # Adaptive threshold (robust)
    ridge_valid = ridge[np.isfinite(ridge)]
    if ridge_valid.size == 0:
        return pd.DataFrame([]), ridge, np.zeros_like(ridge, dtype=bool)
    t_otsu = threshold_otsu(ridge_valid)
    t_p90 = np.percentile(ridge_valid, 90)
    thresh = max(t_otsu, t_p90)

    binary = ridge > thresh
    min_size = max(20, int(min_size_frac * binary.size))
    cleaned = remove_small_objects(binary, min_size=min_size)

    # Skeletonize + measure non-horizontal wrinkles only
    skeleton = skeletonize(cleaned)
    labels = label(skeleton)

    rows = []
    for region in regionprops(labels):
        angle_deg = np.rad2deg(region.orientation)  # 0°~horizontal, ±90°~vertical (skimage convention)
        a = abs(angle_deg)
        if a < angle_exclude:  # only drop near-horizontal
            continue
        rows.append({
            "label": region.label,
            "length_px": region.major_axis_length,
            "angle_deg": angle_deg,
        })

    df = pd.DataFrame(rows)
    return df, ridge, skeleton



def detect_wrinkles(data):
  
    # --- 2. Smooth the data to reduce noise ---
    blurred = gaussian_filter(data, sigma=1)

    # --- 3. Enhance ridge-like structures (Frangi filter) ---
    ridge_enhanced = frangi(blurred)

    # --- 4. Threshold to isolate strong ridge responses ---
    # You can choose a fixed threshold, or use Otsu’s method
    # threshold_value = 0.1
    threshold_value = threshold_otsu(ridge_enhanced)
    binary = ridge_enhanced > threshold_value

    # --- 5. Clean binary image ---
    cleaned = remove_small_objects(binary, min_size=20)

    # --- 6. Skeletonize the cleaned ridges ---
    skeleton = skeletonize(cleaned)

    # --- 7. Label connected components ---
    labels = label(skeleton)
    props = regionprops(labels)

    results = []

    # --- 8. Measure and filter non-horizontal wrinkles ---
    for region in props:
        angle_deg = np.rad2deg(region.orientation)  # Negative = CCW from horizontal
        abs_angle = abs(angle_deg)

        # Filter out nearly horizontal lines (within ±20° of 0° or 180°)
        if abs_angle < 20 or abs_angle > 160:
            continue  # Skip horizontal lines

        # Measure wrinkle properties
        length = region.major_axis_length

        # Estimate width using distance transform on cleaned binary image
        mask = (labels == region.label)
        dist = distance_transform_edt(cleaned)
        width_vals = dist[mask]
        width = 2 * np.mean(width_vals) if width_vals.size > 0 else 0

        results.append({
            "label": region.label,
            "length": length,
            "width": width,
            "angle_deg": angle_deg
        })

    # --- 9. Display results as DataFrame ---
    df_results = pd.DataFrame(results)
    print(df_results)

    # --- 10. Visualization ---
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].imshow(data, cmap='gray')
    ax[0].set_title("Original Grayscale Data")

    ax[1].imshow(ridge_enhanced, cmap='hot')
    ax[1].set_title("Frangi Ridge Enhanced")

    ax[2].imshow(data, cmap='gray')
    ax[2].imshow(skeleton, cmap='winter', alpha=0.6)
    ax[2].set_title("Wrinkle Skeletons (Non-Horizontal)")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()




# ==== Top-Hat + Edge-Band (thesis-style) diagonal wrinkle detector ====
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import disk, opening, remove_small_objects, skeletonize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table

def _rd_read_heightmap_table(path):
    """
    Robust CSV/Excel reader with multiple format detection.
    Handles various delimiters, decimal separators, and headers.
    Fills NaNs with robust statistics.
    """
    f = str(path).lower()
    
    # For CSV/TXT files, try multiple formats
    if f.endswith((".csv", ".txt")):
        df = None
        
        # Try multiple CSV formats in order of likelihood
        formats = [
            {"sep": ";", "decimal": ",", "header": None},  # German format
            {"sep": ",", "decimal": ".", "header": None},   # English format
            {"sep": ";", "decimal": ",", "header": 0},      # With header
            {"sep": ",", "decimal": ".", "header": 0},      # With header
        ]
        
        for fmt in formats:
            try:
                # Read CSV with format-specific parameters
                read_kwargs = {"sep": fmt["sep"], "decimal": fmt["decimal"], "header": fmt["header"]}
                df = pd.read_csv(path, **read_kwargs)
                
                # Verify we got valid data (>30% non-NaN)
                df_temp = df.apply(pd.to_numeric, errors="coerce")
                valid_pct = df_temp.notna().sum().sum() / df_temp.size
                
                if valid_pct > 0.3:  # At least 30% valid data
                    df = df_temp
                    break
                    
            except Exception as e:
                continue
        
        if df is None:
            # Last resort: try with default pandas settings
            try:
                df = pd.read_csv(path)
                df = df.apply(pd.to_numeric, errors="coerce")
            except Exception:
                df = pd.DataFrame([[0]])  # Fallback to single zero
    
    elif f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        try:
            im = Image.open(path).convert('L')  # grayscale
            df = pd.DataFrame(np.asarray(im, dtype=float))
        except Exception:
            df = pd.DataFrame([[0]])
    else:  # Excel files
        try:
            df = pd.read_excel(path, header=None)
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception:
            df = pd.DataFrame([[0]])  # Fallback to single zero
    
    # Convert to numpy array
    arr = df.to_numpy()
    
    # Handle NaN values robustly
    valid_mask = np.isfinite(arr)
    
    if valid_mask.sum() > 0:  # At least some valid data
        # Calculate robust median from valid data only
        median_val = np.nanmedian(arr[valid_mask])
        
        # Fill NaNs with median
        arr = np.where(valid_mask, arr, median_val)
    else:
        # All NaN - use zeros
        print(f"[WARNING] All values in {path} are NaN, using zeros")
        arr = np.zeros_like(arr)
    
    return arr

def _rd_find_coating_edge(img, band_px, side="right"):
    """
    Smooth + per-row |dI/dx| argmax.
    side='right' means ROI is to the RIGHT of edge (coating inside is brighter on right).
    """
    # Bail out early if array is too small
    H, W = img.shape
    if W < 2 or H < 2:
        return np.zeros(H), np.zeros_like(img, dtype=bool)  # empty edge_x, empty ROI

    g = gaussian_filter(img, 1.0)
    gx = np.abs(np.gradient(g, axis=1))  # <-- now safe because W>=2
    edge_x = np.argmax(gx, axis=1)  # per-row
    roi = np.zeros_like(img, dtype=bool)
    if side == "right":
        for y, x in enumerate(edge_x):
            roi[y, x:min(x+band_px, W)] = True
    else:
        for y, x in enumerate(edge_x):
            roi[y, max(0, x-band_px):x] = True
    return edge_x, roi


def _rd_white_tophat(img, rad):
    se = disk(int(rad))
    opened = opening(img, se)
    th = img - opened
    th[th < 0] = 0
    return th

def _rd_skel_endpoints(skel):
    import numpy as np
    from scipy.signal import convolve2d
    k = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nn = convolve2d(skel.astype(np.uint8), k, mode="same", boundary="fill")
    ys, xs = np.where((skel) & (nn==1))
    return list(zip(ys, xs))

def _rd_start_end_endpoints(eps):
    """
    From a list of skeleton endpoints, return only the start (min Y) and end (max Y).
    Returns: [(y_start, x_start), (y_end, x_end)] or empty list.
    """
    import numpy as np
    if len(eps) < 2:
        return eps  # If only 1 or 0 endpoints, return as-is
    
    eps_arr = np.array(eps)
    y_coords = eps_arr[:, 0]
    
    # Find indices of min and max Y
    idx_start = np.argmin(y_coords)
    idx_end = np.argmax(y_coords)
    
    if idx_start == idx_end:
        return [eps[idx_start]]  # Same point, return once
    
    return [eps[idx_start], eps[idx_end]]

def _rd_angle_len(coords):
    """
    Safe angle/length estimator using PCA on coords with guards for tiny/degenerate sets.
    Returns (angle_from_vertical_deg, length_px).
    """
    import numpy as np
    pts = np.asarray(coords, dtype=int)
    if pts.size == 0 or pts.shape[0] < 2:
        return 0.0, 0.0
    ys, xs = pts[:, 0].astype(float), pts[:, 1].astype(float)
    if np.std(xs) < 1e-6 and np.std(ys) < 1e-6:
        return 0.0, 0.0
    xs -= xs.mean()
    ys -= ys.mean()
    try:
        cov = np.cov(np.vstack([xs, ys]))
        if not np.isfinite(cov).all():
            return 0.0, 0.0
        w, v = np.linalg.eig(cov)
        if not (np.isfinite(w).all() and np.isfinite(v).all()):
            return 0.0, 0.0
        vx, vy = v[0, np.argmax(w)], v[1, np.argmax(w)]
    except Exception:
        return 0.0, 0.0

    phi = (np.degrees(np.arctan2(vy, vx)) + 180.0) % 180.0
    ang_from_vertical = abs(90.0 - phi)
    length = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    return ang_from_vertical, length



import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt


def detect_wrinkles_tophat_edgeband(
    path,
    y_lower=None, y_upper=None,
    px_per_mm=10.0,
    edge_band_px=120,         # set to None/0 for bandless (full width)
    tophat_rad=9,
    min_len_px=40,
    endpoint_dist_px=120,
    angle_center_deg=30.0,
    angle_tol_deg=30,
    small_remove_px=40,
    coating_side="right",
    use_multiscale=True,
    super_permissive=False,
):
    """
    Top-Hat + skeleton diagonal detector. Bandless mode supported via
    edge_band_px=None/0. Returns:
      mask_skel, edge_x, roi_mask, stats(DataFrame),
      wrinkle_points, wrinkle_labels, _debug_edge_band_px
    """
    import numpy as np, pandas as pd
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from skimage.morphology import disk, opening, remove_small_objects, skeletonize
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops_table

    img = _rd_read_heightmap_table(path)
    H, W = img.shape

    # vertical crop
    if y_lower is not None and y_upper is not None:
        y0 = int(max(0, min(y_lower, y_upper)))
        y1 = int(min(H, max(y_lower, y_upper)))
    else:
        y0, y1 = 0, H
    img_c = img[y0:y1, :]

    # === edge & ROI band (safe for None/0) ===
    want_band = bool(edge_band_px) and int(edge_band_px) > 0
    band_px   = int(edge_band_px) if want_band else 1

    edge_x, roi_mask_tmp = _rd_find_coating_edge(img_c, band_px, side=coating_side)
    if want_band:
        roi_mask = roi_mask_tmp
    else:
        roi_mask = np.ones_like(img_c, dtype=bool)

    x_edge_float = float(np.nanmedian(edge_x))
    x_edge_int   = int(round(x_edge_float))

    # (optional) light smoothing
    img_c_s = gaussian_filter(img_c, 1.0)

    # === white top-hat (single or multi-scale) ===
    def _white_tophat(a, r):
        op = opening(a, disk(int(r)))
        th = a - op
        th[th < 0] = 0
        return th

    if use_multiscale:
        radii = [max(3, int(tophat_rad/2)), int(tophat_rad), max(int(tophat_rad)+4, 13)]
        ths = [_white_tophat(img_c_s, r) for r in radii]
        th = np.max(ths, axis=0)
    else:
        th = _white_tophat(img_c_s, int(tophat_rad))

    th_roi = np.where(roi_mask, th, 0)

    # === threshold with robust fallbacks ===
    vals = th_roi[th_roi > 0]
    if vals.size > 32:
        try:
            # More permissive thresholding - start with lower percentiles
            t = min(threshold_otsu(vals), np.percentile(vals, 50))  # Reduced from 65
        except Exception:
            t = np.percentile(vals, 70)  # Reduced from 85
    else:
        t = np.percentile(vals, 70) if vals.size else 0  # Reduced from 85
    bw = th_roi > t
    if bw.sum() < 2000 and vals.size:  # Increased threshold from 1000
        bw = th_roi > np.percentile(vals, 60)  # Reduced from 75
    if bw.sum() < 500 and vals.size:  # Increased threshold from 200
        bw = th_roi > np.percentile(vals, 45)  # Reduced from 60
    if bw.sum() < 100 and vals.size:  # Additional fallback
        bw = th_roi > np.percentile(vals, 35)

    # === clean & skeletonize ===
    if small_remove_px and small_remove_px > 0:
        bw = remove_small_objects(bw, small_remove_px)
    bw   = opening(bw, disk(1))
    skel = skeletonize(bw)

    # === components + geometry filter ===
    lab   = label(skel, connectivity=2)
    props = regionprops_table(lab, properties=["label", "coords"])

    # distance-to-edge (for endpoint gate); disabled if super_permissive or bandless
    do_endpoint_gate = (not super_permissive) and (endpoint_dist_px is not None) and (endpoint_dist_px > 0)
    if do_endpoint_gate:
        edge_img = np.zeros_like(skel, dtype=bool)
        for r, x in enumerate(edge_x):
            xi = int(x)
            if 0 <= xi < skel.shape[1]:
                edge_img[r, xi] = True
        dist_to_edge = distance_transform_edt(~edge_img)

    def _angle_len(coords):
        # delegate to robust global
        return _rd_angle_len(coords)

    keep, rows = [], []
    for lbl, coords in zip(props["label"], props["coords"]):
        coords = np.array(coords, dtype=int)
        comp = (lab == lbl)

        eps = _rd_skel_endpoints(comp)
        if not eps:
            continue

        if do_endpoint_gate and not any(dist_to_edge[y, x] <= endpoint_dist_px for (y, x) in eps):
            continue

        ang, length = _angle_len(coords)  # 0° vertical, 90° horizontal
        # Strict diagonal-only: 33° to 57° (45° ± 12°), reject horizontal/vertical
        if (33.0 <= ang <= 57.0) and (length >= min_len_px):
            keep.append(lbl)
            rows.append({
                "label": int(lbl),
                "length_px": float(length),
                "length_mm": float(length) / px_per_mm,
                "angle_deg": float(ang),
            })

    # super-permissive final pass if still empty
    if len(keep) == 0 and not super_permissive:
        for lbl, coords in zip(props["label"], props["coords"]):
            coords = np.array(coords, dtype=int)
            ang, length = _rd_angle_len(coords)
            # Strict diagonal-only: 33° to 57° (no horizontal/vertical)
            diag_ok = (33.0 <= ang <= 57.0)
            if diag_ok and length >= min_len_px:
                keep.append(lbl)
                rows.append({
                    "label": int(lbl),
                    "length_px": float(length),
                    "length_mm": float(length) / px_per_mm,
                    "angle_deg": float(ang),
                })

        super_permissive = True

    keep_mask = np.isin(lab, keep)
    
    # Create separate visual mask: strict sub-component filtering for clean visualization
    visual_mask = keep_mask.copy()
    if visual_mask.sum() > 0:
        from scipy.ndimage import label as ndlabel
        temp_lab = ndlabel(visual_mask)[0]
        clean_mask = np.zeros_like(visual_mask)
        for temp_lbl in np.unique(temp_lab)[1:]:
            comp_coords = np.argwhere(temp_lab == temp_lbl)
            if len(comp_coords) < 5:
                continue
            ang, _ = _rd_angle_len(comp_coords)
            if 33.0 <= ang <= 57.0:
                clean_mask[temp_lab == temp_lbl] = True
        visual_mask = clean_mask

    # endpoints back to full frame Y (only start/end per wrinkle)
    # IMPORTANT: Extract endpoints from ORIGINAL labels to avoid merging nearby wrinkles
    kept_endpoints, kept_labels = [], []
    for lbl in keep:
        comp = (lab == lbl)
        eps = _rd_skel_endpoints(comp)
        eps = _rd_start_end_endpoints(eps)  # Filter to only start/end points
        kept_endpoints.extend([(y + y0, x) for (y, x) in eps])
        kept_labels.extend([int(lbl)] * len(eps))

    stats_df = pd.DataFrame(rows)
    full_edge_x = np.full(H, np.nan); full_edge_x[y0:y1] = edge_x
    mask_full   = np.zeros_like(img, dtype=bool); mask_full[y0:y1, :] = keep_mask  # Full mask for measurements
    visual_full = np.zeros_like(img, dtype=bool); visual_full[y0:y1, :] = visual_mask  # Clean mask for viz
    roi_full    = np.zeros_like(img, dtype=bool); roi_full[y0:y1, :]  = roi_mask

    return {
        "mask_skel": mask_full,  # Full mask for measurements
        "mask_visual": visual_full,  # Clean diagonal-only mask for visualization
        "edge_x": full_edge_x,
        "roi_mask": roi_full,
        "stats": stats_df,
        "wrinkle_points": kept_endpoints,
        "wrinkle_labels": kept_labels,
        "_debug_edge_band_px": int(edge_band_px or 0),
        "_dbg_bw_pixels": int(bw.sum()),
        "_dbg_skel_pixels": int(skel.sum()),
    }


def detect_wrinkles_sobel_band(
    path,
    y_lower=None, y_upper=None,
    px_per_mm=10.0,
    edge_band_px=120,         # set to None/0 to disable the band (full width)
    sobel_sigma=1.2,          # light smoothing before Sobel
    thr_percentile=80,        # initial (permissive) percentile threshold
    min_len_px=15,            # keep short lines (first pass)
    angle_center_deg=45.0,    # diagonals; 0° = vertical, 90° = horizontal
    angle_tol_deg=25,         # +/- around the center
    small_remove_px=8,        # remove tiny dots; set 0 to keep everything
    coating_side="right",
):
    """
    Sobel-based diagonal wrinkle detector. If `edge_band_px` is None/0, the ROI
    is the full image width (no band). Returns:
      mask_skel, edge_x, roi_mask, stats(DataFrame), wrinkle_points, wrinkle_labels,
      coating_side, edge_band_px (int)
    """
    import numpy as np, pandas as pd
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from skimage.filters import sobel_h, sobel_v
    from skimage.morphology import opening, disk, remove_small_objects, skeletonize
    from skimage.measure import label, regionprops_table

    # --- helpers expected elsewhere in your codebase ---
    img = _rd_read_heightmap_table(path)             # (H, W) float heightmap
    H, W = img.shape

    # crop vertical ROI if provided
    if y_lower is not None and y_upper is not None:
        y0, y1 = int(max(0, min(y_lower, y_upper))), int(min(H, max(y_lower, y_upper)))
    else:
        y0, y1 = 0, H
    img_c = img[y0:y1, :]

    # === 1) Edge & ROI band (safe for edge_band_px=None/0) ===
    want_band = bool(edge_band_px) and int(edge_band_px) > 0
    band_px   = int(edge_band_px) if want_band else 1  # minimal band for finder

    edge_x, roi_mask_tmp = _rd_find_coating_edge(img_c, band_px, side=coating_side)

    if want_band:
        # build a symmetric band around the per-row edge position
        Hc, Wc = img_c.shape
        yy, xx = np.indices((Hc, Wc))
        edge_col_med = int(np.clip(np.nanmedian(edge_x), 0, Wc - 1))
        ex = np.nan_to_num(edge_x, nan=edge_col_med).astype(int)
        ex = np.clip(ex, 0, Wc - 1)
        half = int(band_px // 2)
        roi_sym = np.abs(xx - ex[:, None]) <= half
        roi_mask = roi_mask_tmp | roi_sym
    else:
        # bandless mode: whole width
        roi_mask = np.ones_like(img_c, dtype=bool)

    x_edge_float = float(np.nanmedian(edge_x))
    x_edge_int   = int(round(x_edge_float))

    # === 2) Pre-enhance (small white top-hat), blur, Sobel ===
    from skimage.morphology import opening as _opening, disk as _disk
    def _white_tophat(a, r):
        opened = _opening(a, _disk(int(r)))
        th = a - opened
        th[th < 0] = 0
        return th

    pre = _white_tophat(img_c, 7)
    g   = gaussian_filter(pre, sobel_sigma)
    gx  = sobel_h(g)
    gy  = sobel_v(g)
    mag = np.hypot(gx, gy)

    # === 3) Threshold inside ROI with fallbacks ===
    mag_roi = np.where(roi_mask, mag, 0)
    vals = mag_roi[mag_roi > 0]
    if vals.size:
        t = np.percentile(vals, thr_percentile)
        bw = mag_roi > t
        if bw.sum() < 4000:
            bw = mag_roi > np.percentile(vals, 70)
        if bw.sum() < 1000:
            bw = mag_roi > np.percentile(vals, 60)
        if bw.sum() < 300:
            bw = mag_roi > np.percentile(vals, 55)
    else:
        bw = mag_roi > 0

    # === 4) Clean + skeletonize ===
    if small_remove_px and small_remove_px > 0:
        bw = remove_small_objects(bw, small_remove_px)
    bw   = opening(bw, disk(1))
    skel = skeletonize(bw)

    # === 5) Components → angle/length filter (0°=vertical) ===
    lab   = label(skel, connectivity=2)
    props = regionprops_table(lab, properties=["label", "coords"])

    # distance to edge for endpoint gating (useful only if we had a band)
    need_gate = want_band
    if need_gate:
        edge_img = np.zeros_like(skel, dtype=bool)
        for r, x in enumerate(edge_x):
            xi = int(x)
            if 0 <= xi < skel.shape[1]:
                edge_img[r, xi] = True
        dist_to_edge = distance_transform_edt(~edge_img)

    def _angle_len(coords):
        # delegate to robust global
        return _rd_angle_len(coords)

    keep, rows = [], []
    for lbl, coords in zip(props["label"], props["coords"]):
        coords = np.array(coords, dtype=int)
        comp = (lab == lbl)

        # endpoints (use your helper)
        eps = _rd_skel_endpoints(comp)
        if not eps:
            continue
        if need_gate and not any(dist_to_edge[y, x] <= 22 for (y, x) in eps):
            continue

        ang, length = _rd_angle_len(coords)
        # Strict diagonal-only: 33° to 57° (45° ± 12°), reject horizontal/vertical
        diag_ok = (33.0 <= ang <= 57.0)
        if diag_ok and length >= min_len_px:
            keep.append(lbl)
            rows.append({
                "label": int(lbl),
                "length_px": float(length),
                "length_mm": float(length) / px_per_mm,
                "angle_deg": float(ang),
            })
            print(f"[DEBUG] Kept wrinkle lbl={lbl}: ang={ang:.1f}° len={length:.1f}px")

    keep_mask = np.isin(lab, keep)
    
    # Create separate visual mask: strict sub-component filtering for clean visualization
    visual_mask = keep_mask.copy()
    if visual_mask.sum() > 0:
        from scipy.ndimage import label as ndlabel
        # Re-label and check each small component's angle
        temp_lab = ndlabel(visual_mask)[0]
        clean_mask = np.zeros_like(visual_mask)
        for temp_lbl in np.unique(temp_lab)[1:]:
            comp_coords = np.argwhere(temp_lab == temp_lbl)
            if len(comp_coords) < 5:
                continue  # Too small to measure angle reliably
            ang, _ = _rd_angle_len(comp_coords)
            # Only keep components that are diagonal (33-57°)
            if 33.0 <= ang <= 57.0:
                clean_mask[temp_lab == temp_lbl] = True
        visual_mask = clean_mask

    # map kept endpoints back to full frame Y (only start/end per wrinkle)
    # IMPORTANT: Extract endpoints from ORIGINAL labels to avoid merging nearby wrinkles
    kept_endpoints, kept_labels = [], []
    for lbl in keep:
        comp = (lab == lbl)
        eps = _rd_skel_endpoints(comp)
        before_filter = len(eps)
        eps = _rd_start_end_endpoints(eps)  # Filter to only start/end points
        after_filter = len(eps)
        print(f"  [EP] lbl={lbl}: {before_filter} endpoints -> {after_filter} (start/end only)")
        kept_endpoints.extend([(y + y0, x) for (y, x) in eps])
        kept_labels.extend([int(lbl)] * len(eps))

    stats_df  = pd.DataFrame(rows)
    full_edge = np.full(H, np.nan); full_edge[y0:y1] = edge_x
    mask_full = np.zeros_like(img, dtype=bool); mask_full[y0:y1, :] = keep_mask  # Full mask for measurements
    visual_full = np.zeros_like(img, dtype=bool); visual_full[y0:y1, :] = visual_mask  # Clean mask for viz
    roi_full  = np.zeros_like(img, dtype=bool); roi_full[y0:y1, :]  = roi_mask
    
    print(f"[DEBUG] Sobel {coating_side}: kept {len(keep)} wrinkles, mask_full.sum={mask_full.sum()}, visual={visual_full.sum()}")

    return {
        "mask_skel": mask_full,  # Full mask for measurements (used by _metrics_from_res)
        "mask_visual": visual_full,  # Clean diagonal-only mask for visualization
        "edge_x": full_edge,
        "roi_mask": roi_full,
        "stats": stats_df,
        "wrinkle_points": kept_endpoints,
        "wrinkle_labels": kept_labels,
        "coating_side": coating_side,
        "edge_band_px": int(edge_band_px or 0),
        "_dbg_bw_pixels": int(bw.sum()),
        "_dbg_skel_pixels": int(skel.sum()),
    }




def detect_wrinkles_gabor_band(
    path,
    y_lower=None, y_upper=None,
    px_per_mm=10.0,
    edge_band_px=200,        # wide, so obliques stay inside
    thetas_deg=(35, 45, 55), # diagonals (~vertical±10°)
    frequencies=(0.04, 0.07, 0.1),  # ridge scales; tune if needed
    min_len_px=12,
    angle_center_deg=45.0,
    angle_tol_deg=25,
    small_remove_px=8,
    coating_side="right",
):
    """
    Gabor-bank diagonal wrinkle detector in a symmetric band around the coating edge.
    Returns dict compatible with your other detectors, plus _dbg* counters.
    """
    import numpy as np, pandas as pd
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from skimage.filters import gabor
    from skimage.morphology import opening, disk, remove_small_objects, skeletonize
    from skimage.measure import label, regionprops_table

    # --- read + crop Y ---
    img = _rd_read_heightmap_table(path)
    H, W = img.shape
    if y_lower is not None and y_upper is not None:
        y0, y1 = int(max(0, min(y_lower, y_upper))), int(min(H, max(y_lower, y_upper)))
    else:
        y0, y1 = 0, H
    img_c = img[y0:y1, :]

    # --- edge & symmetric band around edge ---
    edge_x, roi_mask0 = _rd_find_coating_edge(img_c, edge_band_px, side=coating_side)
    Hc, Wc = img_c.shape
    yy, xx = np.indices((Hc, Wc))
    edge_med = int(np.clip(np.nanmedian(edge_x), 0, Wc-1))
    ex = np.nan_to_num(edge_x, nan=edge_med).astype(int)
    ex = np.clip(ex, 0, Wc-1)
    band = int(max(6, edge_band_px // 2))
    roi_sym = np.abs(xx - ex[:, None]) <= band
    roi_mask = (roi_mask0 | roi_sym)

    # --- oriented Gabor bank, then take maximum response across bank ---
    # light blur can stabilize, but often raw works well for Gabor:
    gimg = gaussian_filter(img_c, 0.7)
    resp_max = np.zeros_like(gimg, dtype=float)

    for th_deg in thetas_deg:
        th = np.deg2rad(th_deg)
        for f in frequencies:
            # skimage.filters.gabor returns (real, imag)
            r, i = gabor(gimg, frequency=f, theta=th)
            mag  = np.hypot(r, i)
            # keep only inside band
            mag *= roi_mask
            # track max response over the bank
            resp_max = np.maximum(resp_max, mag)

    # --- adaptive threshold ladder (very permissive) ---
    vals = resp_max[roi_mask]
    dbg_pcts = {}
    def _pct(p):
        if vals.size == 0:
            return 0.0
        t = float(np.percentile(vals, p))
        dbg_pcts[p] = t
        return t

    if vals.size:
        cuts = [90, 80, 70, 60, 55, 50, 45, 40]
        bw = np.zeros_like(resp_max, bool)
        for p in cuts:
            t = _pct(p)
            bw = (resp_max > t) & roi_mask
            if bw.sum() >= 300:
                break
        if bw.sum() < 200:
            # top 35% of the band values
            thr = _pct(65)
            bw = (resp_max > thr) & roi_mask
        if bw.sum() < 100:
            # anything non-zero in the band
            bw = (resp_max > 0) & roi_mask
    else:
        bw = np.zeros_like(resp_max, bool)

    # --- gentle cleaning + skeleton ---
    if small_remove_px > 0:
        bw = remove_small_objects(bw, small_remove_px)
    bw   = opening(bw, disk(1))
    skel = skeletonize(bw)

    # --- components -> angle/length ---
    lab   = label(skel, connectivity=2)
    props = regionprops_table(lab, properties=["label", "coords"])

    def _angle_len(coords):
        # delegate to robust global
        return _rd_angle_len(coords)

    keep, rows = [], []
    for lbl, coords in zip(props["label"], props["coords"]):
        ang, length = _angle_len(coords)
        # Strict diagonal-only: 33° to 57° (45° ± 12°), reject horizontal/vertical
        if (33.0 <= ang <= 57.0) and (length >= min_len_px):
            keep.append(lbl)
            rows.append({
                "label": int(lbl),
                "length_px": float(length),
                "length_mm": float(length) / px_per_mm,
                "angle_deg": float(ang),
            })

    keep_mask = np.isin(lab, keep)
    
    # Create separate visual mask: strict sub-component filtering for clean visualization
    visual_mask = keep_mask.copy()
    if visual_mask.sum() > 0:
        from scipy.ndimage import label as ndlabel
        temp_lab = ndlabel(visual_mask)[0]
        clean_mask = np.zeros_like(visual_mask)
        for temp_lbl in np.unique(temp_lab)[1:]:
            comp_coords = np.argwhere(temp_lab == temp_lbl)
            if len(comp_coords) < 5:
                continue
            ang, _ = _rd_angle_len(comp_coords)
            if 33.0 <= ang <= 57.0:
                clean_mask[temp_lab == temp_lbl] = True
        visual_mask = clean_mask

    # --- map to full frame + endpoints (only start/end per wrinkle) ---
    # IMPORTANT: Extract endpoints from ORIGINAL labels to avoid merging nearby wrinkles
    kept_endpoints, kept_labels = [], []
    for lbl in keep:
        comp = (lab == lbl)
        eps = _rd_skel_endpoints(comp)
        eps = _rd_start_end_endpoints(eps)  # Filter to only start/end points
        kept_endpoints.extend([(y + y0, x) for (y, x) in eps])
        kept_labels.extend([int(lbl)] * len(eps))

    stats_df  = pd.DataFrame(rows)
    full_edge = np.full(H, np.nan); full_edge[y0:y1] = edge_x
    mask_full = np.zeros_like(img, bool); mask_full[y0:y1, :] = keep_mask  # Full mask for measurements
    visual_full = np.zeros_like(img, bool); visual_full[y0:y1, :] = visual_mask  # Clean mask for viz
    roi_full  = np.zeros_like(img, bool); roi_full[y0:y1, :]  = roi_mask

    return {
        "mask_skel": mask_full,  # Full mask for measurements
        "mask_visual": visual_full,  # Clean diagonal-only mask for visualization
        "edge_x": full_edge,
        "roi_mask": roi_full,
        "stats": stats_df,
        "wrinkle_points": kept_endpoints,
        "wrinkle_labels": kept_labels,
        "coating_side": coating_side,
        "edge_band_px": int(edge_band_px),
        # debug:
        "_dbg_vals_size": int(vals.size),
        "_dbg_bw_pixels": int(bw.sum()),
        "_dbg_skel_pixels": int(skel.sum()),
        "_dbg_pcts": {k: float(v) for k, v in dbg_pcts.items()},
        "_dbg_edge_med": float(edge_med),
        "_dbg_shapes": {"full": (H, W), "crop": (Hc, Wc)},
    }
