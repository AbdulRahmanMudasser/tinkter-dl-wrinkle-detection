# 🔄 PROJECT FLOW GUIDE
## Understanding the Wrinkle Detection System - Step by Step

---

## 🎯 **WHAT THIS PROJECT DOES**

**Simple Answer**: This is a **quality control system** for a manufacturing process that makes battery electrodes. It takes pictures of the electrode surface and automatically finds wrinkles (defects) in the coating.

---

## 📊 **THE BIG PICTURE**  

```
MANUFACTURING LINE → LASER SENSORS → COMPUTER → ANALYSIS → RESULTS
```

1. **Manufacturing Line**: Makes battery electrodes by coating material on metal foil
2. **Laser Sensors**: Take height measurements of the electrode surface
3. **Computer**: Runs this software to analyze the measurements
4. **Analysis**: Finds wrinkles, measures them, and reports quality
5. **Results**: Shows if the product is good or needs adjustment

---

## 🔍 **DETAILED FLOW**

### **STEP 1: DATA COLLECTION** 📸
```
Real Sensors (or Test Files) → CSV Files
```
- **In Production**: 3 laser sensors scan the electrode surface
- **For Testing**: You use pre-recorded CSV files from the `test/` folder
- **Data Format**: Height measurements in a grid (like a 3D picture)

### **STEP 2: FILE SELECTION** 📁
```
User Interface → File Dialog → Select Sensor Files
```
- **AS Side**: `Sensor_0_*.csv` (Air Side - one edge of the electrode)
- **BS Side**: `Sensor_2_*.csv` (Base Side - other edge of the electrode)
- **Middle**: `Sensor_1_*.csv` (Optional - center area)

### **STEP 3: REGION SETUP** 🎯
```
User Sets ROI (Region of Interest) → Software Crops Data
```
- **Lower Y-Value**: Starting row (e.g., 100)
- **Upper Y-Value**: Ending row (e.g., 500)
- **Why Crop**: Focus on specific area, faster processing

### **STEP 4: EDGE DETECTION** 🔵
```
Raw Data → Find Coating Edge → Blue Line on Screen
```
- **What it does**: Finds where the coating starts/stops
- **Why important**: Wrinkles only occur in the coated area
- **Visual**: Blue line shows the detected edge

### **STEP 5: WRINKLE SEARCH** 🔍
```
Coated Area → Search for Wrinkles → Find Skeleton Lines
```
- **Search Area**: Orange band around the coating edge
- **Algorithm**: Top-Hat filter finds ridge-like features
- **Result**: Skeleton lines representing wrinkles

### **STEP 6: FILTERING** ⚙️
```
Raw Wrinkles → Filter by Size/Angle → Valid Wrinkles Only
```
- **Size Filter**: Remove tiny noise, keep significant wrinkles
- **Angle Filter**: Only diagonal wrinkles (±25° from vertical)
- **Length Filter**: Minimum length to be considered real

### **STEP 7: VISUALIZATION** 📊
```
Results → Display on Screen → Show Statistics
```
- **Cyan Dots**: Skeleton of detected wrinkles
- **Red Dots**: Endpoints of wrinkles
- **Statistics**: Count, length, angle, height

---

## 🖥️ **USER INTERFACE FLOW**

### **What You See on Screen**

#### **Left Panel (Controls)**
```
┌─────────────────────────┐
│ ROI Settings            │ ← Set analysis area
│ [100] [500]             │
│                         │
│ [Manual Analysis]       │ ← Start button
│ [Automatic Analysis]    │ ← Continuous mode
│ [Stop]                  │ ← Stop button
│                         │
│ Results Display:        │
│ Wrinkles: 0             │ ← Shows count
│ Length: 0.0             │ ← Shows average
│ Height: 0.0             │ ← Shows height
│ Angle: 0.0              │ ← Shows orientation
└─────────────────────────┘
```

#### **Right Panel (Visualization)**
```
┌─────────────────────────┐
│ Gray Background         │ ← Raw sensor data
│ Blue Line              │ ← Detected coating edge
│ Orange Band            │ ← Search area for wrinkles
│ Cyan Dots              │ ← Detected wrinkle skeleton
│ Red Dots               │ ← Wrinkle endpoints
└─────────────────────────┘
```

---

## 🚀 **HOW TO USE IT (Step by Step)**

### **For Testing (What You're Doing Now)**

1. **Start the Program**
   ```
   Click: main.py → Application opens
   ```

2. **Set Analysis Area**
   ```
   Unterer Y-Wert: 100
   Oberer Y-Wert: 500
   ```

3. **Start Analysis**
   ```
   Click: "manuelle Wrinkleauswertung"
   ```

4. **Select Files**
   ```
   First Dialog: Choose Sensor_0_*.csv (AS side)
   Second Dialog: Choose Sensor_2_*.csv (BS side)
   ```

5. **View Results**
   ```
   Look for: Cyan dots = wrinkles found
   Check: Wrinkle count > 0
   ```

### **For Production Use**

1. **Connect Real Sensors**
   ```
   Hardware → Software → Live data
   ```

2. **Set Process Parameters**
   ```
   Temperature, Speed, Tension, etc.
   ```

3. **Start Continuous Monitoring**
   ```
   Click: "Automatische Auswertung"
   ```

4. **Monitor Quality**
   ```
   Watch: Wrinkle counts, adjust process
   ```

---

## 🔧 **TECHNICAL ARCHITECTURE**

### **File Structure**
```
wrinkle/
├── main.py                 ← Main application (GUI)
├── functions.py            ← Core processing logic
├── wrinkle_detection_new.py ← Detection algorithms
├── wrinkle_aux_funcs.py    ← Helper functions
├── config.ini              ← Settings
├── test/                   ← Sample data files
│   ├── Sensor_0_*.csv      ← AS side data
│   ├── Sensor_1_*.csv      ← Middle data
│   └── Sensor_2_*.csv      ← BS side data
└── venv/                   ← Python environment
```

### **Algorithm Pipeline**
```
CSV File → Read Data → Edge Detection → ROI Selection → 
Top-Hat Filter → Thresholding → Skeletonization → 
Geometry Filtering → Statistics → Display
```

---

## 🎯 **WHY YOU'RE GETTING 0 WRINKLES**

### **Current Problem**
```
TopHatStrict Algorithm → Too Restrictive → 0 Wrinkles Found
```

### **What's Happening**
1. **Algorithm**: Using "TopHatStrict" (most restrictive)
2. **Parameters**: Too strict for your test data
3. **Result**: No wrinkles detected, even though they exist

### **Our Improvements**
1. **Better Parameters**: More permissive settings
2. **Fallback System**: Try multiple algorithms
3. **Sobel Algorithm**: Alternative detection method

---

## 🚨 **CURRENT STATUS**

### **What's Working**
- ✅ Application launches
- ✅ File loading works
- ✅ Edge detection works (blue line visible)
- ✅ ROI selection works
- ✅ Visualization works

### **What's Not Working**
- ❌ Wrinkle detection (0 count)
- ❌ Using restrictive algorithm
- ❌ Parameters too strict for test data

### **What We Fixed**
- ✅ Improved parameters in `functions.py`
- ✅ Added fallback system
- ✅ Better thresholding
- ✅ Multiple algorithm support

---

## 🎓 **KEY CONCEPTS**

### **AS vs BS**
- **AS (Air Side)**: Top surface of electrode
- **BS (Base Side)**: Bottom surface of electrode
- **Why Both**: Check quality on both sides

### **ROI (Region of Interest)**
- **What**: Area of the image to analyze
- **Why**: Faster processing, focus on important area
- **How**: Set Y-values to crop the data

### **Top-Hat Algorithm**
- **What**: Mathematical filter for ridge detection
- **Why**: Good at finding wrinkle-like features
- **How**: Compares local features to background

### **Skeleton**
- **What**: Simplified line representation of wrinkles
- **Why**: Easier to measure and analyze
- **How**: Thins the detected features to center lines

---

## 🎯 **NEXT STEPS**

### **Immediate Actions**
1. **Try smaller ROI**: 100-500 instead of 0-10000
2. **Look for fallback messages**: Should see "[FALLBACK]" in console
3. **Check algorithm selection**: Should see different algorithm names

### **Expected Results**
- **Wrinkles detected**: Count > 0
- **Visual feedback**: Cyan and red dots
- **Console messages**: Algorithm selection info

### **If Still Not Working**
- **Check file format**: Ensure CSV files are valid
- **Try different ROI**: Different areas might have wrinkles
- **Use manual parameters**: Adjust detection settings

---

*This guide explains the complete flow from manufacturing to results. The system is designed to automatically detect wrinkles in battery electrode coatings for quality control.*

