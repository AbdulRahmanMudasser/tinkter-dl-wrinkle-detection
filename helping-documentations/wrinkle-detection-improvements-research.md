# Wrinkle Detection System - Research-Based Improvement Plan

## Executive Summary
After analyzing the project codebase and conducting research on state-of-the-art ridge/wrinkle detection methods, I've identified the core issues and developed 5 robust approaches to fix the detection system.

## Current System Weaknesses (CRITICAL)

### 1. **Data I/O Fragility** 🔴
- Repeated "All-NaN slice encountered" warnings
- CSV reading fails with different delimiters (`,`, `;`)
- Inconsistent handling of headers
- NaN values treated inconsistently

### 2. **Algorithm Selection Bias** 🔴
- System defaults to **TopHatStrict** (most restrictive)
- Fallback algorithms never reached
- Parameters too conservative:
  - `min_len_px=24` (too long for actual wrinkles)
  - `angle_tol_deg=16` (too narrow)
  - `endpoint_dist_px=140` (too restrictive)

### 3. **Edge-Band Drift** 🔴
- Edge band extends into negative x-values
- Edge detection drifts from actual coating edge
- Coating side ("left" vs "right") misidentified

### 4. **No Multi-Scale Detection** ⚠️
- Single filter scale misses faint wrinkles
- No Frangi/Hessian vesselness filter
- Limited Gabor orientation coverage

### 5. **Fixed Thresholds** ⚠️
- Percentile thresholds (80th, 85th) too rigid
- No hysteresis thresholding
- No adaptive calibration per image

### 6. **No Quantitative Feedback** ⚠️
- No precision/recall metrics
- Can't tune without ground truth
- No debug output for parameters

### 7. **GUI Input Validation** ⚠️
- Empty text fields cause Tkinter errors
- No validation of ROI bounds
- No user feedback on invalid inputs

---

## Research Summary

### Relevant Industrial Techniques Found:
1. **Frangi/Hessian Vesselness Filter** - Detects ridge-like structures via 2nd-order derivatives
2. **Multi-Scale Ridge Detection** - Catches wrinkles at different scales
3. **Hysteresis Thresholding** - Connects broken ridge segments
4. **Gabor Filter Banks** - Multi-orientation texture analysis
5. **Structure Tensor Orientation** - Determines local ridge direction

### Why Facial Wrinkle Research Doesn't Apply:
- Focus on skin textures vs. material defects
- Different contrast mechanisms
- Human face has known structure vs. unknown defect patterns
- Cosmetics focus vs. quality control requirements

---

## Five Robust Approaches

### **Approach 1: Data Hygiene & Deterministic Preprocessing** 🔧
**Effort:** 1-2 days | **Impact:** HIGH | **Risk:** LOW

**Goal:** Eliminate all NaN/format issues and ensure consistent input data

**Plan:**
```python
# New function: read_heightmap_robust()
def read_heightmap_robust(path):
    # 1. Try multiple CSV formats in order
    for sep in [';', ',']:
        for header in [True, False]:
            try:
                df = pd.read_csv(path, sep=sep, header=header if header else None)
                # 2. Coerce to numeric
                df = df.apply(pd.to_numeric, errors='coerce')
                # 3. Check for valid data (>50% non-NaN)
                if df.notna().sum().sum() > df.size * 0.5:
                    # 4. Fill NaNs with robust median
                    median_val = np.nanmedian(df.values)
                    df = df.fillna(median_val)
                    return df.values
            except: continue
    raise ValueError(f"Could not read {path}")
```

**Touch Points:**
- `wrinkle_detection_new.py::_rd_read_heightmap_table()`
- Add ROI validation (y0 < y1, within image bounds)
- Add shape consistency checks

**Acceptance Criteria:**
- Zero "All-NaN" warnings on all test files
- Deterministic output (rerun gives identical results)
- Proper shape handling for all CSV variants

---

### **Approach 2: Stabilized Edge-Band & Coating Side Inference** 🎯
**Effort:** 1-2 days | **Impact:** HIGH | **Risk:** LOW

**Goal:** Keep edge band locked to coating edge, no negative x drift

**Plan:**
```python
def find_coating_edge_stabilized(df, edge_col=slice(25, 380)):
    # 1. Extract edge candidate column
    edge_data = df.iloc[:, edge_col]
    
    # 2. Detect edge using robust derivative
    edge_x = np.zeros(len(edge_data))
    for y in range(len(edge_data)):
        row = edge_data.iloc[y].values
        # Robust edge finder: maximum gradient
        grad = np.diff(row)
        edge_x[y] = np.argmax(np.abs(grad)) + edge_col.start
    
    # 3. SMOOTH edge_x to prevent drift
    from scipy.ndimage import median_filter
    edge_x_smooth = median_filter(edge_x, size=11)
    
    # 4. Check bounds [0, W-1]
    edge_x_smooth = np.clip(edge_x_smooth, 0, df.shape[1]-1)
    
    # 5. Infer coating side by comparing left vs right of edge
    left_mean = np.mean(df.values[:, :edge_x_smooth.min().astype(int)])
    right_mean = np.mean(df.values[:, edge_x_smooth.max().astype(int):])
    coating_side = "right" if right_mean > left_mean else "left"
    
    return edge_x_smooth, coating_side
```

**Touch Points:**
- `wrinkle_aux_funcs.py::detect_coating_edge()`
- `wrinkle_detection_new.py::_rd_find_coating_edge()`
- Add visualization overlay

**Acceptance Criteria:**
- Zero negative x-values in edge band
- Edge band always aligned with coating edge (blue line)
- Coating side correctly identified (left vs right)
- No band extending into empty space

---

### **Approach 3: Multi-Scale Ridge Ensemble with Hysteresis** 🔬
**Effort:** 2-3 days | **Impact:** VERY HIGH | **Risk:** MEDIUM

**Goal:** Detect wrinkles at multiple scales, even faint ones

**Plan:**
```python
def detect_wrinkles_ridge_ensemble(path, y0, y1, coating_side):
    """Multi-scale ridge detection combining Frangi, Gabor, Sobel"""
    
    # 1. Read and preprocess
    arr = read_heightmap_robust(path)
    arr_roi = arr[y0:y1, :]
    H, W = arr_roi.shape
    
    # 2. Apply Frangi vesselness (multiple sigmas)
    from skimage.filters import frangi
    frangi_responses = []
    for sigma in [0.5, 1.0, 1.5]:
        frangi_ridge = frangi(arr_roi, sigma_range=(sigma, sigma*2), 
                             beta1=0.5, beta2=15)
        frangi_responses.append(frangi_ridge)
    
    # 3. Apply Gabor filters (multi-orientation)
    from skimage.filters import gabor
    gabor_responses = []
    for theta in [35, 40, 45, 50, 55]:  # degrees
        for freq in [0.04, 0.07, 0.10]:
            _, real_filter = gabor(arr_roi, frequency=freq, 
                                  theta=np.deg2rad(theta))
            gabor_responses.append(np.abs(real_filter))
    
    # 4. Apply enhanced Sobel (multiple sigmas)
    from skimage.filters import sobel_h, sobel_v
    sobel_responses = []
    for sigma in [0.5, 1.0, 2.0]:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(arr_roi, sigma)
        sobel_v = sobel_v(blurred)
        sobel_h = sobel_h(blurred)
        sobel_mag = np.hypot(sobel_h, sobel_v)
        sobel_responses.append(sobel_mag)
    
    # 5. Fuse all responses (weighted combination)
    frangi_max = np.max(frangi_responses, axis=0)
    gabor_max = np.max(np.array(gabor_responses), axis=0)
    sobel_max = np.max(np.array(sobel_responses), axis=0)
    
    # Normalize each to [0, 1]
    frangi_norm = (frangi_max - frangi_max.min()) / (frangi_max.max() - frangi_max.min() + 1e-10)
    gabor_norm = (gabor_max - gabor_max.min()) / (gabor_max.max() - gabor_max.min() + 1e-10)
    sobel_norm = (sobel_max - sobel_max.min()) / (sobel_max.max() - sobel_max.min() + 1e-10)
    
    # Weighted fusion
    fused = 0.4*frangi_norm + 0.4*gabor_norm + 0.2*sobel_norm
    
    # 6. HYSTERESIS thresholding (connects broken segments)
    from skimage.filters import threshold_otsu, threshold_local
    high_thresh = np.percentile(fused, 70)
    low_thresh = np.percentile(fused, 50)
    
    strong_mask = fused > high_thresh
    weak_mask = (fused > low_thresh) & (fused <= high_thresh)
    
    from scipy.ndimage import binary_dilation
    # Connect strong to weak if they're adjacent
    strong_dilated = binary_dilation(strong_mask, structure=np.ones((3,3)))
    connected = strong_mask | (weak_mask & strong_dilated)
    
    # 7. Skeletonize and filter
    from skimage.morphology import skeletonize
    skel = skeletonize(connected)
    
    # 8. Filter by angle and length
    angles, lengths = extract_wrinkle_angles_lengths(skel, coating_side)
    filtered_skel = filter_by_geometry(skel, angles, lengths, 
                                       min_len_px=10, angle_tol_deg=60)
    
    return {"mask_skel": filtered_skel, 
            "fused_response": fused,
            "edge_band_px": None}  # Full image search
```

**Touch Points:**
- New function in `wrinkle_detection_new.py`
- Add to algorithm candidate list in `main.py`
- Test on your Sensor_0 and Sensor_2 CSVs

**Acceptance Criteria:**
- Detects wrinkles that TopHat missed
- Higher `skel_px` count than current methods
- Visual output matches thesis figures (continuous red lines)
- Can find faint wrinkles (low contrast)

---

### **Approach 4: Self-Calibrating Thresholds Per Image** 🎛️
**Effort:** 1-2 days | **Impact:** MEDIUM | **Risk:** LOW

**Goal:** Adapt thresholds to each image's characteristics

**Plan:**
```python
def calibrate_thresholds_per_image(response_map, target_false_positive_rate=0.01):
    """Find threshold that gives target FP rate"""
    
    # 1. Calculate robust statistics
    valid_pixels = response_map[~np.isnan(response_map)]
    median_val = np.median(valid_pixels)
    iqr_val = np.percentile(valid_pixels, 75) - np.percentile(valid_pixels, 25)
    
    # 2. Use Otsu's method for initial guess
    from skimage.filters import threshold_otsu
    otsu_thresh = threshold_otsu(valid_pixels)
    
    # 3. Grid search around Otsu
    candidates = np.linspace(otsu_thresh*0.5, otsu_thresh*1.5, 50)
    best_thresh = otsu_thresh
    best_score = float('inf')
    
    for thresh in candidates:
        mask = response_map > thresh
        fp_rate = mask.sum() / response_map.size
        
        # Score: distance from target FP rate
        score = abs(fp_rate - target_false_positive_rate)
        
        if score < best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh
```

**Touch Points:**
- Replace fixed percentiles (80th, 85th) with calibrated values
- Add logging to show chosen thresholds
- Cache parameters for reproducibility

**Acceptance Criteria:**
- Different images get different thresholds
- Logs show adaptive selection
- Reduces false positives on noisy images
- Consistent detection across datasets

---

### **Approach 5: Geometry-Coherent Ridge Tracing** 📐
**Effort:** 2-3 days | **Impact:** MEDIUM | **Risk:** MEDIUM

**Goal:** Trace wrinkle lines by tracking orientation coherence

**Plan:**
```python
def trace_ridges_with_coherence(skeleton, angle_constraint_deg=30):
    """Trace ridges while maintaining orientation coherence"""
    
    # 1. Find all endpoints
    from skimage.morphology import skeletonize_endpoints
    endpoints = skeletonize_endpoints(skeleton)
    
    # 2. For each endpoint, trace line forward
    ridges = []
    for endpoint in np.argwhere(endpoints):
        ridge = [tuple(endpoint)]
        current = endpoint
        direction = None
        
        # Grow ridge while following coherent direction
        for _ in range(10000):  # max length
            neighbors = get_8_neighbors(current, skeleton.shape)
            valid_neighbors = [n for n in neighbors if skeleton[tuple(n)]]
            
            if len(valid_neighbors) == 0:
                break
            
            # Choose neighbor that maintains direction
            if direction is None:
                next_point = valid_neighbors[0]
                direction = next_point - current
            else:
                # Find neighbor closest to current direction
                angles = [np.arctan2(*(n-current)) for n in valid_neighbors]
                direction_angle = np.arctan2(*direction)
                angle_diffs = [abs(a - direction_angle) for a in angles]
                next_point = valid_neighbors[np.argmin(angle_diffs)]
                direction = next_point - current
            
            ridge.append(tuple(next_point))
            current = next_point
            
            # Check angle coherence
            if len(ridge) > 10:
                recent_angles = [np.arctan2(ridge[i+1][0]-ridge[i][0], 
                                          ridge[i+1][1]-ridge[i][1]) 
                                for i in range(len(ridge)-11, len(ridge)-1)]
                angle_std = np.std(recent_angles)
                if angle_std > np.deg2rad(angle_constraint_deg):
                    break  # Too much deviation
            
            if len(valid_neighbors) > 1:
                break  # Junction reached
        
        if len(ridge) >= 5:  # Minimum length
            ridges.append(np.array(ridge))
    
    return ridges
```

**Touch Points:**
- New function in `wrinkle_detection_new.py`
- Use instead of simple skeleton filtering
- Combine with angle/length filtering

**Acceptance Criteria:**
- Produces continuous ridge lines
- Respects local orientation
- Connects broken segments properly
- Handles junctions robustly

---

## Implementation Priority

### **Phase 1: Critical Fixes (Week 1)** 🔥
1. Fix `_rd_read_heightmap_table()` - eliminate NaN warnings
2. Fix edge band boundary issues
3. Make algorithm parameters more permissive
4. Add GUI input validation

**Expected Result:** No crashes, more detections

### **Phase 2: Robust Detection (Week 2)** 🔬
1. Implement Approach 3 (Multi-Scale Ridge Ensemble)
2. Add hysteresis thresholding
3. Test on Sensor_0 and Sensor_2 files

**Expected Result:** Higher detection rate, faint wrinkles found

### **Phase 3: Refinement (Week 3)** ⚙️
1. Implement Approach 4 (Self-Calibrating Thresholds)
2. Add Approach 5 (Geometry-Coherent Tracing)
3. Performance tuning

**Expected Result:** Consistent, reliable detection

---

## Key Research Findings Applied

### From Ridge Detection Literature:
- **Multi-scale approach** catches varied wrinkle sizes
- **Hysteresis thresholding** connects broken segments
- **Frangi filter** sensitive to ridge-like structures
- **Structure tensor** provides local orientation

### Adapted for Industrial Materials:
- Can't assume skin-like texture
- Must handle unknown defect patterns
- Focus on edge-guided search (coating edge)
- Robust to noise and artifacts

---

## References
- Frangi, A.F. et al. (1998): "Multi-scale Vesselness Filter"
- Lindeberg, T. (1998): "Feature Detection with Scale-Space"
- Canny, J. (1986): "Edge Detection with Hysteresis"
- Scikit-image documentation: Ridge detection, skeletonization

