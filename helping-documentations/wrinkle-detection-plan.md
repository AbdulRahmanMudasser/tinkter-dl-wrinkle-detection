# Wrinkle Detection System - Implementation Plan

## 🎯 Goal
Fix the electrode calendering wrinkle detection system to reliably detect wrinkles in Sensor_0 and Sensor_2 data.

---

## 📊 Current Status Analysis

### What's Working ✅
- CSV file reading (basic functionality)
- GUI interface loads and runs
- Multi-algorithm framework exists (TopHat, Sobel, Gabor)
- Edge detection finds coating edge (blue line)
- Row equalization works (gamma correction)

### What's Broken 🔴
1. **Algorithm Selection:** Uses TopHatStrict (most restrictive) → finds 0 wrinkles
2. **Parameter Rigidity:** `min_len_px=24`, `angle_tol_deg=16` → too strict
3. **No Multi-Scale:** Single filter scale misses faint wrinkles
4. **Data I/O Errors:** Repeated "All-NaN" warnings
5. **Edge Band Drift:** Extends into negative x, wrong coating side
6. **No Fallback Logic:** If strict fails, doesn't try permissive

---

## 🛠️ Implementation Plan

### **Phase 1: Immediate Fixes (Done ✓)**
✅ Made TopHat parameters more permissive in `main.py`
✅ Added Sobel fallback with permissive parameters
✅ Fixed array dimension mismatches in plotting

### **Phase 2: Data I/O Fixes (Next)**
**Files:** `wrinkle_detection_new.py::_rd_read_heightmap_table()`

**Changes:**
```python
def _rd_read_heightmap_table(path):
    """Robust CSV/Excel reader with format detection"""
    
    # Try multiple formats
    df = None
    formats = [
        (";", ",", False),  # German format: semicolon, comma decimal
        (",", ".", False),  # English format: comma, period decimal
        (";", ",", True),   # With header
        (",", ".", True),   # With header
    ]
    
    for sep, dec, has_header in formats:
        try:
            if has_header:
                df = pd.read_csv(path, sep=sep, decimal=dec, index_col=0)
            else:
                df = pd.read_csv(path, sep=sep, decimal=dec, header=None)
            
            # Verify we got valid data
            if df.notna().sum().sum() > df.size * 0.3:  # At least 30% valid
                break
        except:
            continue
    
    if df is None:
        raise ValueError(f"Could not read {path}")
    
    # Convert to numeric, fill NaNs
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Fill NaNs with robust statistics
    valid_mask = ~np.isnan(df.values)
    if valid_mask.sum() > 0:
        median = np.nanmedian(df.values)
        df = df.fillna(median)
    else:
        # All NaN - use zeros
        df = pd.DataFrame(np.zeros_like(df.values))
    
    return df.values
```

**Expected:** Zero "All-NaN" warnings

---

### **Phase 3: Edge Band Stabilization**
**Files:** `wrinkle_detection_new.py::_rd_find_coating_edge()`

**Changes:**
```python
def _rd_find_coating_edge(edge_x, band_px, W, H, side):
    """Find coating edge band with stabilization"""
    
    # 1. CLAMP edge_x to valid range [0, W-1]
    edge_x = np.clip(edge_x, 0, W-1)
    
    # 2. SMOOTH edge_x to prevent drift
    from scipy.ndimage import median_filter
    edge_x_smooth = median_filter(edge_x, size=11)
    
    # 3. Create ROI mask with smooth edge
    roi = np.zeros((H, W), dtype=bool)
    
    for y in range(H):
        x = int(edge_x_smooth[y])
        x = max(0, min(x, W-1))  # Extra safety
        
        if side == "right":
            # Search to the RIGHT of edge
            x_start = x
            x_end = min(x + band_px, W)
            roi[y, x_start:x_end] = True
        else:
            # Search to the LEFT of edge
            x_start = max(0, x - band_px)
            x_end = x
            roi[y, x_start:x_end] = True
    
    return roi
```

**Expected:** No negative x-values, stable edge band

---

### **Phase 4: Multi-Scale Ridge Detection**
**Files:** New function in `wrinkle_detection_new.py`

**Changes:**
```python
def detect_wrinkles_multiscale_ridge(path, y0, y1, coating_side, 
                                    min_len_px=5, angle_tol_deg=60):
    """Multi-scale ridge detection with hysteresis"""
    
    # 1. Read data robustly
    arr = _rd_read_heightmap_table(path)
    arr_roi = arr[y0:y1, :]
    
    # 2. Find coating edge
    edge_x = find_coating_edge(arr_roi)
    band_mask = find_edge_band(edge_x, band_px=150, coating_side)
    
    # 3. Apply multi-scale filters
    
    # A) Frangi vesselness (ridge detector)
    from skimage.filters import frangi
    frangi_responses = []
    for sigma in [0.5, 1.0, 1.5]:
        resp = frangi(arr_roi, sigma_range=(sigma, sigma*2))
        frangi_responses.append(resp)
    frangi_max = np.max(frangi_responses, axis=0)
    
    # B) Gabor filters (multi-orientation)
    from skimage.filters import gabor
    gabor_responses = []
    for theta in [35, 40, 45, 50, 55]:
        _, real = gabor(arr_roi, frequency=0.06, theta=np.deg2rad(theta))
        gabor_responses.append(np.abs(real))
    gabor_max = np.max(gabor_responses, axis=0)
    
    # C) Enhanced Sobel (multi-scale)
    from scipy.ndimage import gaussian_filter
    from skimage.filters import sobel_v, sobel_h
    sobel_responses = []
    for sigma in [0.5, 1.0, 2.0]:
        blurred = gaussian_filter(arr_roi, sigma)
        sv = sobel_v(blurred)
        sh = sobel_h(blurred)
        sm = np.hypot(sh, sv)
        sobel_responses.append(sm)
    sobel_max = np.max(sobel_responses, axis=0)
    
    # 4. Fuse responses
    frangi_norm = normalize(frangi_max)
    gabor_norm = normalize(gabor_max)
    sobel_norm = normalize(sobel_max)
    
    fused = 0.4*frangi_norm + 0.4*gabor_norm + 0.2*sobel_norm
    
    # 5. HYSTERESIS thresholding
    strong_thresh = np.percentile(fused, 70)
    weak_thresh = np.percentile(fused, 50)
    
    strong_mask = fused > strong_thresh
    weak_mask = (fused > weak_thresh) & (fused <= strong_thresh)
    
    from scipy.ndimage import binary_dilation
    strong_dilated = binary_dilation(strong_mask, iterations=1)
    connected = strong_mask | (weak_mask & strong_dilated)
    
    # 6. Skeletonize
    from skimage.morphology import skeletonize
    skel = skeletonize(connected)
    
    # 7. Filter by angle/length in edge band
    skel_filtered = filter_by_angle_and_length(skel, band_mask,
                                                min_len_px, angle_tol_deg)
    
    # 8. Extract stats
    stats = extract_wrinkle_stats(skel_filtered)
    
    return {
        "mask_skel": skel_filtered,
        "stats": stats,
        "fused_response": fused,
        "band_mask": band_mask
    }
```

**Expected:** Detects faint wrinkles, higher count than TopHat

---

### **Phase 5: Integrate Multi-Scale into main.py**
**Files:** `main.py::run_one()`

**Changes:**
```python
def run_one(path, coating_side):
    cands = []
    
    # 1. Multi-scale ridge (NEW - most permissive first)
    r = wr.detect_wrinkles_multiscale_ridge(
        path=path, y_lower=y0, y_upper=y1,
        coating_side=coating_side,
        min_len_px=5, angle_tol_deg=60
    )
    cands.append(("MultiScaleRidge", r, _score(r)))
    
    # 2. Permissive TopHat
    r = wr.detect_wrinkles_tophat_edgeband(...)
    cands.append(("TopHatPerm", r, _score(r)))
    
    # 3. Sobel
    r = wr.detect_wrinkles_sobel_band(...)
    cands.append(("Sobel", r, _score(r)))
    
    # ... rest of algorithms
    
    # Select best candidate
    best = max(cands, key=lambda x: x[2])
    return best
```

**Expected:** Multi-scale selected first, finds more wrinkles

---

## 📋 Testing Procedure

### Test Case 1: Sensor_0 + Sensor_2 (Your current issue)
1. Start application
2. Set ROI: y0=100, y1=500
3. Click "manuelle Wrinkleauswertung"
4. Select Sensor_0_*.csv
5. Select Sensor_2_*.csv
6. **Expected:**
   - Console: "chosen: MultiScaleRidge / MultiScaleRidge"
   - Console: "[DBG] n_stats = X" where X > 0
   - GUI: Cyan skeleton dots visible
   - GUI: Red endpoint dots visible

### Test Case 2: Full Image
1. Set ROI: y0=0, y1=10000 (full image)
2. Run detection
3. **Expected:** More wrinkles found (full coverage)

### Test Case 3: Edge Band Visualization
1. Check orange overlay on GUI
2. **Expected:**
   - Orange band always adjacent to blue edge line
   - No negative x-values
   - Band doesn't extend into empty space

---

## 🎯 Success Criteria

### Must Have ✅
- [ ] Zero "All-NaN" warnings
- [ ] Wrinkle count > 0 on Sensor_0/Sensor_2
- [ ] Edge band visualization correct (orange overlay)
- [ ] No negative x-values in edge band
- [ ] Console shows algorithm selection

### Nice to Have 👍
- [ ] Multiple algorithms tried (fallback logic works)
- [ ] Detects faint wrinkles
- [ ] Performance < 5 seconds per image
- [ ] Logs show adaptive thresholds

### Future Work 🔮
- Add precision/recall metrics
- User can tune parameters via GUI
- Export detection results to CSV
- Real-time processing mode

---

## 🚀 Next Steps

**Right Now:**
1. Implement Phase 2 (Data I/O fixes)
2. Test on Sensor_0/Sensor_2
3. If still 0 wrinkles → implement Phase 4 (Multi-Scale Ridge)

**This Week:**
1. Complete Phases 2, 3, 4, 5
2. Test thoroughly
3. Document results

**Next Week:**
1. Refine parameters
2. Add GUI for tuning
3. Performance optimization

---

## 📝 Notes

- **Your current files:** `Sensor_0_2025-03-26T10_09_25_00_00.csv` and `Sensor_2_*.csv`
- **Current issue:** 0 wrinkles detected despite visual presence
- **Root cause:** Algorithm selection biased toward TopHatStrict (too restrictive)
- **Solution:** Multi-scale ridge detection + permissive parameters

Good luck! 🚀

