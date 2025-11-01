# 🇬🇧 English Interface Translation Guide
## Wrinkle Detection System - German to English Translation

---

## 📋 **MAIN INTERFACE ELEMENTS**

### **Top Menu Bar**
| German | English | Description |
|--------|---------|-------------|
| **Datei** | **File** | Main file menu |
| **Ausgabeverzeichnis festlegen** | **Set Output Directory** | Choose where to save results |
| **Beenden** | **Exit** | Close the application |

---

## 🎛️ **WRINKLE DETECTION FRAME** (Left Panel)

### **ROI (Region of Interest) Settings**
| German | English | Description |
|--------|---------|-------------|
| **Unterer Y-Wert** | **Lower Y-Value** | Starting row for analysis |
| **Oberer Y-Wert** | **Upper Y-Value** | Ending row for analysis |
| **Y-Werte** | **Y-Values** | Row range settings |

### **Detection Controls**
| German | English | Description |
|--------|---------|-------------|
| **manuelle Wrinkleauswertung** | **Manual Wrinkle Analysis** | Blue button to start analysis |
| **Automatische Auswertung** | **Automatic Analysis** | Continuous monitoring mode |
| **Stoppen** | **Stop** | Stop current analysis |

### **Measurement Controls**
| German | English | Description |
|--------|---------|-------------|
| **Starte Messung** | **Start Measurement** | Green button to begin data acquisition |
| **Beende Messung** | **End Measurement** | Red button to stop measurement |
| **Batchgröße** | **Batch Size** | Number of measurements per batch (default: 10) |
| **Batchmessung** | **Batch Measurement** | Green button to start batch processing |
| **Ergebnis ausgeben** | **Output Result** | Blue button to export/save results |
| **Versuchsergebnisse laden** | **Load Test Results** | Blue button to load previous results |

### **Statistics Display**
| German | English | Description |
|--------|---------|-------------|
| **Anzahl Falten** | **Wrinkle Count** | Number of detected wrinkles |
| **Durchschnittliche Länge** | **Average Length** | Mean wrinkle length |
| **Durchschnittliche Höhe** | **Average Height** | Mean wrinkle height |
| **Dominanter Winkel** | **Dominant Angle** | Most common wrinkle orientation |

### **Process Parameters Display**
| German | English | Description |
|--------|---------|-------------|
| **Linienlast (N/mm)** | **Line Load (N/mm)** | Current linear load in Newtons per millimeter |
| **Temperatur (°C)** | **Temperature (°C)** | Current process temperature in Celsius |
| **Bahngeschwindigkeit (m/min)** | **Web Speed (m/min)** | Current material speed in meters per minute |
| **Bahnzug (N)** | **Web Tension (N)** | Current web tension in Newtons |
| **Aufwickler Kraft (N)** | **Winder Force (N)** | Force exerted by the winder in Newtons |
| **Abwickler Kraft (N)** | **Unwinder Force (N)** | Force exerted by the unwinder in Newtons |

---

## 📊 **PLOT FRAME** (Right Panel)

### **Plot Titles**
| German | English | Description |
|--------|---------|-------------|
| **Sobel/Top-Hat, verdicktes Skelett (Diagonal)** | **Sobel/Top-Hat, Thickened Skeleton (Diagonal)** | Main plot title |
| **Beschichtungskante** | **Coating Edge** | Blue line showing detected edge |
| **Skelett** | **Skeleton** | Cyan dots showing wrinkle skeleton |
| **Endpunkte** | **Endpoints** | Red dots showing wrinkle endpoints |

### **Axis Labels**
| German | English | Description |
|--------|---------|-------------|
| **Spalten** | **Columns** | X-axis (horizontal pixels) |
| **Zeilen** | **Rows** | Y-axis (vertical pixels) |

---

## 🔧 **CONFIGURATION FILE** (config.ini)

### **Main Sections**
| German | English | Description |
|--------|---------|-------------|
| **Zeiteinstellungen** | **Time Settings** | Date and time configuration |
| **Arbeitsverzeichnisse** | **Working Directories** | File path settings |
| **Anlagengeometrie** | **Plant Geometry** | Machine layout parameters |
| **Materialdaten** | **Material Data** | Material specifications |
| **Protokoll** | **Protocol** | Experiment logging |
| **thresholds** | **thresholds** | Detection thresholds |

### **Time Settings**
| German | English | Description |
|--------|---------|-------------|
| **current_date** | **current_date** | Today's date |

### **Working Directories**
| German | English | Description |
|--------|---------|-------------|
| **main_output_folder** | **main_output_folder** | Main results folder |
| **result_number_in_directory** | **result_number_in_directory** | Result numbering |

### **Plant Geometry**
| German | English | Description |
|--------|---------|-------------|
| **distance_roller_to_large_laser** | **distance_roller_to_large_laser** | Distance to main laser |
| **distnace_roller_to_small_laser** | **distance_roller_to_small_laser** | Distance to secondary laser |

### **Material Data**
| German | English | Description |
|--------|---------|-------------|
| **material_type** | **material_type** | Type of material |
| **material_supplier** | **material_supplier** | Material supplier |
| **thickness_uncalandered** | **thickness_uncalandered** | Uncalendered thickness |
| **total_width** | **total_width** | Total material width |
| **coating_width** | **coating_width** | Coated area width |
| **substrate_width_as** | **substrate_width_as** | AS side substrate width |
| **substrate_width_bs** | **substrate_width_bs** | BS side substrate width |
| **substrate_thickness** | **substrate_thickness** | Substrate thickness |
| **intermitting** | **intermitting** | Intermittent coating |
| **two_side_coating** | **two_side_coating** | Double-sided coating |
| **intermittend_coating_len** | **intermittent_coating_len** | Intermittent coating length |
| **intemittend_gap** | **intermittent_gap** | Gap between coatings |
| **thickness_calandered_as** | **thickness_calandered_as** | AS calendered thickness |
| **thickness_calandered_bs** | **thickness_calandered_bs** | BS calendered thickness |
| **compaction_as** | **compaction_as** | AS side compaction |
| **compaction_bs** | **compaction_bs** | BS side compaction |

### **Process Parameters**
| German | English | Description |
|--------|---------|-------------|
| **roller_offset_as** | **roller_offset_as** | AS roller offset |
| **roller_offset_bs** | **roller_offset_bs** | BS roller offset |
| **web_tension_abwickler** | **web_tension_unwinder** | Unwinder tension |
| **web_tension_aufwickler** | **web_tension_rewinder** | Rewinder tension |
| **web_tension_zugwerk** | **web_tension_tensioner** | Tensioner tension |
| **wickelcharakteristik** | **winding_characteristic** | Winding behavior |
| **sollkraft_rb** | **target_force_rb** | Target roller force |
| **sollposition_hz** | **target_position_hz** | Target cylinder position |
| **bahngeschwindigkeit** | **web_speed** | Material speed |
| **temperatur** | **temperature** | Process temperature |
| **linienlast** | **line_load** | Linear load |

---

## 🎯 **OPERATIONAL INSTRUCTIONS**

### **How to Use the System**

1. **Set Analysis Region:**
   - **Unterer Y-Wert** (Lower Y-Value): Set starting row (e.g., 100)
   - **Oberer Y-Wert** (Upper Y-Value): Set ending row (e.g., 500)

2. **Start Analysis:**
   - Click **"manuelle Wrinkleauswertung"** (Manual Wrinkle Analysis)
   - Select **Sensor_0** file (AS side)
   - Select **Sensor_2** file (BS side)

3. **View Results:**
   - Check **"Anzahl Falten"** (Wrinkle Count) - should show > 0
   - Observe **"Skelett"** (Skeleton) - cyan dots for wrinkles
   - Look for **"Endpunkte"** (Endpoints) - red dots for boundaries

### **File Selection Dialog**
| German | English | Description |
|--------|---------|-------------|
| **Öffnen** | **Open** | Select file button |
| **Abbrechen** | **Cancel** | Cancel file selection |
| **Dateiname** | **Filename** | File name field |
| **Dateityp** | **File Type** | File type filter |

---

## 🚨 **STATUS MESSAGES**

### **System Messages**
| German | English | Description |
|--------|---------|-------------|
| **Programm gestartet, Sensoren initialisiert** | **Program started, sensors initialized** | Startup message |
| **Light start** | **Light start** | Quick startup mode |
| **[BTN] manuelle Wrinkleauswertung clicked** | **[BTN] manual wrinkle analysis clicked** | Button click log |
| **[BTN] ROI rows: 0:10000** | **[BTN] ROI rows: 0:10000** | Region of interest log |
| **[WRK] AS skel sum= 0  BS skel sum= 0** | **[WRK] AS skeleton sum= 0  BS skeleton sum= 0** | Detection results |
| **chosen: TopHatStrict / TopHatStrict** | **chosen: TopHatStrict / TopHatStrict** | Algorithm selection |
| **Exportverzeichnis ausgewählt** | **Export directory selected** | Output folder selection message |

### **Fallback Messages**
| German | English | Description |
|--------|---------|-------------|
| **[FALLBACK] No wrinkles detected, trying more permissive parameters...** | **[FALLBACK] No wrinkles detected, trying more permissive parameters...** | Parameter relaxation |
| **[FALLBACK] Trying Sobel algorithm...** | **[FALLBACK] Trying Sobel algorithm...** | Algorithm switching |

---

## 📁 **FILE STRUCTURE**

### **Sensor Files**
- **Sensor_0_*.csv** = AS side (Air Side) sensor data
- **Sensor_1_*.csv** = Middle sensor data
- **Sensor_2_*.csv** = BS side (Base Side) sensor data

### **File Naming Convention**
```
Sensor_[ID]_[DATE]T[TIME]_[TIMESTAMP].csv
```
- **ID**: 0=AS, 1=Middle, 2=BS
- **DATE**: YYYY-MM-DD format
- **TIME**: HH_MM_SS format
- **TIMESTAMP**: Additional timing info

---

## ⚙️ **TROUBLESHOOTING**

### **Common Issues**

1. **"AS skel sum= 0  BS skel sum= 0"**
   - **Problem**: No wrinkles detected
   - **Solution**: Try different ROI values or check fallback messages

2. **"RuntimeWarning: All-NaN slice encountered"**
   - **Problem**: Data reading issues
   - **Solution**: Check CSV file format and content

3. **File selection issues**
   - **Problem**: Can't select sensor files
   - **Solution**: Navigate to correct folder (usually `test/` directory)

### **Parameter Tuning Guide**

| Parameter | German | English | Typical Range | Purpose |
|-----------|--------|---------|---------------|---------|
| **Unterer Y-Wert** | Lower Y-Value | 0-5000 | Set analysis start row |
| **Oberer Y-Wert** | Upper Y-Value | 100-10000 | Set analysis end row |
| **edge_band_px** | Edge Band Pixels | 40-120 | Search area width |
| **angle_tol_deg** | Angle Tolerance | 12-35 | Angular acceptance range |

---

## 🎓 **QUICK REFERENCE**

### **Essential Buttons**
- 🔵 **manuelle Wrinkleauswertung** = Start analysis
- ⚫ **Stoppen** = Stop analysis
- 📁 **Ausgabeverzeichnis festlegen** = Set output folder

### **Key Settings**
- 📊 **ROI Values**: 100-500 for testing, 0-10000 for full analysis
- 🎯 **File Selection**: Always select Sensor_0 and Sensor_2 for AS/BS analysis
- 📈 **Results**: Look for skeleton dots (cyan) and endpoints (red)

### **Success Indicators**
- ✅ **Anzahl Falten > 0** = Wrinkles detected successfully
- ✅ **Cyan dots visible** = Skeleton detected
- ✅ **Red dots visible** = Endpoints found
- ✅ **Blue line** = Coating edge detected correctly

---

*This guide covers all major interface elements and should help English speakers navigate the German wrinkle detection system effectively.*
