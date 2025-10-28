import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import globals as vars
from PIL import Image
import os
import time
from multiprocessing import Process, Value, Array, Queue, Pipe
from multiprocessing import Manager, current_process
from datetime import datetime, timezone, date
import globals as vars
import functions as funcs
from configparser import ConfigParser
import threading
import queue



###### Programm Initialisierung #######
#######################################

import functions as funcs, inspect
print("USING functions from:", inspect.getfile(funcs))
print("USING handler:", funcs.ALT_call_wrinkle_detection_manual)
# route legacy name to the new handler
funcs.call_wrinkle_detection_manual = funcs.ALT_call_wrinkle_detection_manual

### GUI Einstellungen ###
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

### Datei Einstellungen ###
config = ConfigParser()
config.read('config.ini')
config.set('Zeiteinstellungen', 'current_date', str(date.today().strftime("%d.%m.%Y")))
with open('config.ini', 'w') as cfg:
    config.write(cfg)

### Setup Multiprocessing ###
# Sensor BUFFER MODE MUST BE OFF!
Status = Queue()
Trigger_S1 = Queue()  # Trigger for Sensor 1->Id 0
Trigger_S2 = Queue()  # Trigger for Sensor 2->Id 1
Trigger_S3 = Queue()  # Trigger for Sensor 3->Id 2

S1Process = Process(target=funcs.handle_sensor, args=([192, 168, 1, 101], 0, 2, 1, Status, Trigger_S1))  # 3
S2Process = Process(target=funcs.handle_sensor,
                    args=([192, 168, 2, 101], 1, 2, 2, Status, Trigger_S2))  # Scale setting 2 default
S3Process = Process(target=funcs.handle_sensor, args=([192, 168, 3, 101], 2, 2, 1, Status, Trigger_S3))

processes = [S1Process, S2Process, S3Process]
triggers = [Trigger_S1, Trigger_S2, Trigger_S3]

### OPC UA Data ###
URL = "opc.tcp://127.0.0.1:4840"

### Klassendefinition ###
# Hauptprogramm
class AutPUT(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after(0, lambda: self.state('zoomed'))
        try:
            # self.after_idle(lambda:funcs.initialize_sensors(processes, triggers, Status))
            print("Light start")
        except:
            pass
        #######################################
        ### Menüleiste ###
        self.menu = tk.Menu(self)
        self.menu_file = tk.Menu(self.menu, tearoff=0)
        self.menu_file.add_command(label="Ausgabeverzeichnis festlegen", command=lambda: set_export_directory(self))
        self.config(menu=self.menu)

        self.menu_file.add_command(label="Beenden", command=lambda: [funcs.end_program(processes), self.destroy()])
        self.config(menu=self.menu)
        self.menu.add_cascade(label="Datei", menu=self.menu_file)

        #######################################
        ### Widgets ###
        self.columnconfigure(1, minsize=920, weight=2)
        self.columnconfigure(3, minsize=1000, weight=2)

        self.report_window = None

        self.frame_terminal = TerminalFrame(master=self, width=1880)
        self.frame_terminal.grid(row=6, column=1, columnspan=8, padx=5, pady=0, sticky="NSEW")

        self.frame_wrinkle_detection = WrinkleDetectionFrame(master=self)
        self.frame_wrinkle_detection.grid(row=1, column=1, rowspan=5, padx=5, pady=5, sticky="NSEW")

        self.frame_output_plots = PlotFrame(self)
        self.frame_output_plots.grid(row=1, column=3, rowspan=5, padx=5, pady=5, sticky="NSEW")

        funcs.add_terminal_line(self.frame_terminal.terminal, "Programm gestartet, Sensoren initialisiert")

class WrinkleDetectionFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        # Überschrift
        self.wrinkle_frame_head_label = ctk.CTkLabel(self, text="Wrinkle-Erkennung")
        self.wrinkle_frame_head_label.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky="NSEW")

        # Canvas AS
        self.wrinkles_AS_fig = Figure(figsize=(4.5, 3.5), dpi=100)
        self.canvas_wrinkles_AS = FigureCanvasTkAgg(self.wrinkles_AS_fig, master=self)
        self.canvas_wrinkles_AS.get_tk_widget().grid(row=1, column=1, padx=5, pady=5, sticky="NSEW")
        self.label_wrinkles_AS = tk.Label(
            self, text="Wrinkledetektion (AS)",
            background=self.canvas_wrinkles_AS.get_tk_widget()["background"], font=("Arial", 14),
        )
        self.label_wrinkles_AS.grid(row=1, column=1, padx=20, pady=10, sticky="NW")

        # Canvas BS
        self.wrinkles_BS_fig = Figure(figsize=(4.5, 3.5), dpi=100)
        self.canvas_wrinkles_BS = FigureCanvasTkAgg(self.wrinkles_BS_fig, master=self)
        self.canvas_wrinkles_BS.get_tk_widget().grid(row=1, column=2, padx=5, pady=5, sticky="NSEW")
        self.label_wrinkles_BS = tk.Label(
            self, text="Wrinkledetektion (BS)",
            background=self.canvas_wrinkles_BS.get_tk_widget()["background"], font=("Arial", 14),
        )
        self.label_wrinkles_BS.grid(row=1, column=2, padx=20, pady=10, sticky="NW")

        # Manuelle Auswertung
        self.y_lower_label = ctk.CTkLabel(self, text="Unterer Y-Wert: ")
        self.y_lower_label.grid(row=2, column=1, padx=10, pady=0, sticky="W")

        self.entry_y_lower_value = tk.StringVar(self); self.entry_y_lower_value.set(str(vars.lower_y_value))
        self.entry_y_lower = ctk.CTkEntry(self, textvariable=self.entry_y_lower_value, width=100)
        self.entry_y_lower.grid(row=2, column=1, padx=10, pady=0, sticky="E")

        self.y_upper_label = ctk.CTkLabel(self, text="Oberer Y-Wert: ")
        self.y_upper_label.grid(row=2, column=2, padx=10, pady=0, sticky="W")

        self.entry_y_upper_value = tk.StringVar(self); self.entry_y_upper_value.set(str(vars.upper_y_value))
        self.entry_y_upper = ctk.CTkEntry(self, textvariable=self.entry_y_upper_value, width=100)
        self.entry_y_upper.grid(row=2, column=2, padx=10, pady=0, sticky="E")

        # Button
        self.button_manuelle_auswertung = ctk.CTkButton(
            self, text="manuelle Wrinkleauswertung", command=self.start_manual_wrinkle_detection
        )
        self.button_manuelle_auswertung.grid(row=3, column=1, padx=10, pady=10, sticky="W")

        # KPIs (use StringVar + formatted strings; right-aligned with width to avoid clipping)
        self.wrinkle_count_label = ctk.CTkLabel(self, text="Anzahl Wrinkle: ")
        self.wrinkle_count_label.grid(row=6, column=1, columnspan=2, padx=10, pady=0, sticky="W")
        self.wrinkle_count_val = tk.StringVar(self); self.wrinkle_count_val.set("0")
        self.wrinkle_count_val_label = ctk.CTkLabel(self, textvariable=self.wrinkle_count_val, width=80, anchor="e")
        self.wrinkle_count_val_label.grid(row=6, column=1, columnspan=2, padx=30, pady=0, sticky="E")

        self.wrinkle_len_label = ctk.CTkLabel(self, text="Durchschnittliche Wrinklelänge (mm): ")
        self.wrinkle_len_label.grid(row=7, column=1, columnspan=2, padx=10, pady=0, sticky="W")
        self.wrinkle_len_val = tk.StringVar(self); self.wrinkle_len_val.set("0.00")
        self.wrinkle_len_val_label = ctk.CTkLabel(self, textvariable=self.wrinkle_len_val, width=80, anchor="e")
        self.wrinkle_len_val_label.grid(row=7, column=1, columnspan=2, padx=30, pady=0, sticky="E")

        self.wrinkle_height_label = ctk.CTkLabel(self, text="Durchschnittliche Wrinklehöhe (mm): ")
        self.wrinkle_height_label.grid(row=8, column=1, columnspan=2, padx=10, pady=0, sticky="W")
        self.wrinkle_height_val = tk.StringVar(self); self.wrinkle_height_val.set("0.00")
        self.wrinkle_height_val_label = ctk.CTkLabel(self, textvariable=self.wrinkle_height_val, width=80, anchor="e")
        self.wrinkle_height_val_label.grid(row=8, column=1, columnspan=2, padx=30, pady=0, sticky="E")

        self.wrinkle_angle_label = ctk.CTkLabel(self, text="Vorherrschende Orientierung (°): ")
        self.wrinkle_angle_label.grid(row=9, column=1, columnspan=2, padx=10, pady=0, sticky="W")
        self.wrinkle_angle_val = tk.StringVar(self); self.wrinkle_angle_val.set("0.0")
        self.wrinkle_angle_val_label = ctk.CTkLabel(self, textvariable=self.wrinkle_angle_val, width=80, anchor="e")
        self.wrinkle_angle_val_label.grid(row=9, column=1, columnspan=2, padx=30, pady=0, sticky="E")

        self.wrinkle_type_label = ctk.CTkLabel(self, text="Kurz/Lang Wrinkle: ")
        self.wrinkle_type_label.grid(row=10, column=1, columnspan=2, padx=10, pady=0, sticky="W")
        self.wrinkle_type_val = tk.StringVar(self); self.wrinkle_type_val.set("0/0")
        self.wrinkle_type_val_label = ctk.CTkLabel(self, textvariable=self.wrinkle_type_val, width=80, anchor="e")
        self.wrinkle_type_val_label.grid(row=10, column=1, columnspan=2, padx=30, pady=0, sticky="E")

    # ---------------------------
    # Manual detection handler
    # ---------------------------
    def start_manual_wrinkle_detection(self):
        """Manual wrinkle detection (thread+queue; UI never blocks)."""
        import os, queue, threading, traceback
        from tkinter import filedialog, messagebox
        import numpy as np
        import pandas as pd
        import wrinkle_detection_new as wr

        print("[BTN] manuelle Wrinkleauswertung clicked")

        # --- 1) ROI ---
        try:
            y_lo_str = self.entry_y_lower_value.get()
            y_hi_str = self.entry_y_upper_value.get()
            
            # Handle empty values gracefully
            if not y_lo_str or y_lo_str.strip() == "":
                y_lo = vars.lower_y_value
            else:
                y_lo = int(y_lo_str)
                
            if not y_hi_str or y_hi_str.strip() == "":
                y_hi = vars.upper_y_value
            else:
                y_hi = int(y_hi_str)
        except ValueError:
            messagebox.showerror("Wrinkle-Erkennung", "Bitte ganze Zahlen für Y-Werte eingeben.")
            return
        if y_lo > y_hi:
            y_lo, y_hi = y_hi, y_lo
        vars.lower_y_value = y_lo
        vars.upper_y_value = y_hi
        print(f"[BTN] ROI rows: {y_lo}:{y_hi}")

        # --- 2) Ensure AS/BS file paths ---
        as_path = getattr(vars, "current_sensor_0_file_path", None)
        if not (isinstance(as_path, str) and os.path.exists(as_path)):
            as_path = filedialog.askopenfilename(
                title="AS-Datei wählen",
                filetypes=[("CSV/TXT", "*.csv *.txt *.dat"), ("Alle Dateien", "*.*")]
            )
            if not as_path:
                messagebox.showerror("Wrinkle-Erkennung", "Keine AS-Datei gewählt.")
                return
            vars.current_sensor_0_file_path = as_path

        bs_path = getattr(vars, "current_sensor_2_file_path", None)
        if not (isinstance(bs_path, str) and os.path.exists(bs_path)):
            bs_path = filedialog.askopenfilename(
                title="BS-Datei wählen",
                filetypes=[("CSV/TXT", "*.csv *.txt *.dat"), ("Alle Dateien", "*.*")]
            )
            if not bs_path:
                messagebox.showerror("Wrinkle-Erkennung", "Keine BS-Datei gewählt.")
                return
            vars.current_sensor_2_file_path = bs_path

        # --- 3) Immediate visual feedback ---
        def _clicked(canvas, fig, label):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"CLICKED ({label})", ha="center", va="center",
                    fontsize=28, color="orange")
            ax.set_axis_off()
            canvas.draw()

        _clicked(self.canvas_wrinkles_AS, self.wrinkles_AS_fig, "AS")
        _clicked(self.canvas_wrinkles_BS, self.wrinkles_BS_fig, "BS")

        # --- 4) Disable button while running ---
        self.button_manuelle_auswertung.configure(state="disabled")

        # --- 5) Queue + poller (define before first .after) ---
        q = queue.Queue()

        def _metrics_from_res(res):
            """KPIs computed ONLY from the wrinkles we actually label (wrinkle_points)."""
            from skimage.measure import label

            wr_pts = res.get("wrinkle_points", [])
            mask = res.get("mask_skel", None)
            if mask is None or mask.sum() == 0 or not wr_pts:
                return 0, 0.0, 0.0, 0.0

            lab = label(mask, connectivity=2)
            px_per_mm = float(getattr(vars, "PX_PER_MM_X", 10.0))

            count = 0
            lens_mm, heights_mm, angles_deg = [], [], []

            for (ry, rx) in wr_pts:
                if ry < 0 or rx < 0 or ry >= lab.shape[0] or rx >= lab.shape[1]:
                    continue
                lab_id = int(lab[ry, rx])
                if lab_id == 0:
                    continue

                coords = np.argwhere(lab == lab_id)
                if coords.shape[0] < 2:
                    continue

                y = coords[:, 0].astype(float)
                x = coords[:, 1].astype(float)
                # PCA orientation + length along principal axis
                xc, yc = x - x.mean(), y - y.mean()
                Sxx = (xc * xc).sum()
                Syy = (yc * yc).sum()
                Sxy = (xc * yc).sum()
                theta = 0.5 * np.arctan2(2 * Sxy, (Sxx - Syy) + 1e-9)
                ang = abs(np.rad2deg(theta))
                ux, uy = np.cos(theta), np.sin(theta)
                proj = xc * ux + yc * uy

                length_px = float(proj.max() - proj.min())
                height_px = float(y.max() - y.min())

                lens_mm.append(length_px / px_per_mm)
                heights_mm.append(height_px / px_per_mm)
                angles_deg.append(ang)
                count += 1

            if count == 0:
                return 0, 0.0, 0.0, 0.0
            return (
                count,
                float(np.mean(lens_mm)),
                float(np.mean(heights_mm)),
                float(np.median(angles_deg)),
            )

        def _poll():
            """Pump worker results into UI; reschedules itself until a message arrives."""
            try:
                msg = q.get_nowait()
            except queue.Empty:
                self.after(80, _poll)
                return

            if not msg.get("ok", False):
                messagebox.showerror("Wrinkle-Erkennung (Thread-Fehler)", msg.get("error", "Unbekannter Fehler"))
                self.button_manuelle_auswertung.configure(state="normal")
                return

            best_as = msg["best_as"]  # (name, side, res, score)
            best_bs = msg["best_bs"]
            y0, y1 = msg["y0"], msg["y1"]
            as_p, bs_p = msg["as_path"], msg["bs_path"]

            def draw_one(canvas, fig, path, res):
                img = wr._rd_read_heightmap_table(path)
                H, W = img.shape

                def _norm(z):
                    lo, hi = np.percentile(z, [1, 99])
                    return z if hi <= lo else np.clip((z - lo) / (hi - lo), 0, 1)

                fig.clear()
                ax = fig.add_subplot(111)
                bg = _norm(img[y0:y1, :])
                ax.imshow(bg, cmap="gray", origin="upper", interpolation="nearest")
                ax.set_title("Sobel/Top-Hat, verdicktes Skelett (Diagonal)")

                # edge line
                edge_x = np.asarray(res["edge_x"], dtype=float)
                yy = np.arange(y1 - y0)
                
                # Ensure edge_x has the right length
                if len(edge_x) != (y1 - y0):
                    # If edge_x is shorter, pad it or interpolate
                    if len(edge_x) < (y1 - y0):
                        # Pad with the last valid value or median
                        if len(edge_x) > 0:
                            pad_value = np.nanmedian(edge_x[np.isfinite(edge_x)]) if np.isfinite(edge_x).any() else W / 2.0
                            edge_x = np.pad(edge_x, (0, (y1 - y0) - len(edge_x)), mode='constant', constant_values=pad_value)
                        else:
                            edge_x = np.full(y1 - y0, W / 2.0, dtype=float)
                    else:
                        # If edge_x is longer, crop it
                        edge_x = edge_x[:y1 - y0]
                
                ex = edge_x
                if np.isfinite(ex).any():
                    ex = np.where(np.isfinite(ex), ex, float(np.nanmedian(ex[np.isfinite(ex)])))
                else:
                    ex = np.full_like(yy, W / 2.0, dtype=float)
                    
                # Ensure both arrays have the same length
                min_len = min(len(ex), len(yy))
                ex = ex[:min_len]
                yy = yy[:min_len]
                
                ax.plot(ex, yy, color="blue", linewidth=2, label="Beschichtungskante")

                # skeleton overlay
                mask = res["mask_skel"]
                ys, xs = np.where(mask[y0:y1, :])
                if ys.size:
                    ax.scatter(xs, ys, s=4, linewidths=0, c="#00FFFF", alpha=0.9)

                # endpoints only (decluttered overlay)
                for (ry, rx) in res.get("wrinkle_points", []):
                    if y0 <= ry < y1:
                        ax.scatter([rx], [ry - y0], s=28, c="red", edgecolors="none")

                ax.legend(loc="upper right", framealpha=0.85)
                ax.set_xlabel("X-axis [pixels]");
                ax.set_ylabel("")
                canvas.draw()

            draw_one(self.canvas_wrinkles_AS, self.wrinkles_AS_fig, as_p, best_as[2])
            draw_one(self.canvas_wrinkles_BS, self.wrinkles_BS_fig, bs_p, best_bs[2])

            # --- KPIs from visible wrinkles only (respect current ROI y0:y1) ---
            def _metrics_from_res(res, y0, y1):
                import numpy as np
                from skimage.measure import label

                # only endpoints inside the ROI
                wr_pts = [(ry, rx) for (ry, rx) in res.get("wrinkle_points", [])
                          if y0 <= ry < y1]
                mask = res.get("mask_skel", None)
                if mask is None or mask.sum() == 0 or not wr_pts:
                    return 0, 0.0, 0.0, 0.0

                lab = label(mask, connectivity=2)
                px_per_mm = float(getattr(vars, "PX_PER_MM_X", 10.0))

                count = 0
                lens_mm, heights_mm, angles_deg = [], [], []
                for (ry, rx) in wr_pts:
                    if not (0 <= ry < lab.shape[0] and 0 <= rx < lab.shape[1]):
                        continue
                    lab_id = int(lab[ry, rx])
                    if lab_id == 0:
                        continue

                    coords = np.argwhere(lab == lab_id)
                    if coords.shape[0] < 2:
                        continue

                    y = coords[:, 0].astype(float)
                    x = coords[:, 1].astype(float)
                    # PCA for angle & length
                    xc, yc = x - x.mean(), y - y.mean()
                    Sxx = (xc * xc).sum();
                    Syy = (yc * yc).sum();
                    Sxy = (xc * yc).sum()
                    theta = 0.5 * np.arctan2(2 * Sxy, (Sxx - Syy) + 1e-9)
                    ang = abs(np.rad2deg(theta))
                    ux, uy = np.cos(theta), np.sin(theta)
                    proj = xc * ux + yc * uy

                    length_px = float(proj.max() - proj.min())
                    height_px = float(y.max() - y.min())

                    lens_mm.append(length_px / px_per_mm)
                    heights_mm.append(height_px / px_per_mm)
                    angles_deg.append(ang)
                    count += 1

                if count == 0:
                    return 0, 0.0, 0.0, 0.0, 0, 0
                # Classify: Long (>=20mm) vs Short (<20mm)
                long_count = sum(1 for L in lens_mm if L >= 20.0)
                short_count = count - long_count
                return (
                    count,
                    float(np.mean(lens_mm)),
                    float(np.mean(heights_mm)),
                    float(np.median(angles_deg)),
                    short_count,
                    long_count,
                )

            # compute KPIs with the same ROI the plots use
            c1, l1, h1, a1, s1, lg1 = _metrics_from_res(best_as[2], y0, y1)
            c2, l2, h2, a2, s2, lg2 = _metrics_from_res(best_bs[2], y0, y1)

            count = c1 + c2
            avg_len = (l1 + l2) / 2.0 if count > 0 and (l1 > 0 or l2 > 0) else 0.0
            avg_h = (h1 + h2) / 2.0 if count > 0 and (h1 > 0 or h2 > 0) else 0.0
            dom_ang = (a1 + a2) / 2.0 if count > 0 and (a1 > 0 or a2 > 0) else 0.0
            short_total = s1 + s2
            long_total = lg1 + lg2

            self.wrinkle_count_val.set(f"{int(count)}")
            self.wrinkle_len_val.set(f"{avg_len:.2f}")
            self.wrinkle_height_val.set(f"{avg_h:.2f}")
            self.wrinkle_angle_val.set(f"{dom_ang:.1f}")
            self.wrinkle_type_val.set(f"{short_total}/{long_total}")

            # re-enable the button
            self.button_manuelle_auswertung.configure(state="normal")

        # --- 6) Background worker (no Tk calls here) ---
        def start_worker(as_path, bs_path, y0, y1):
            def _score(res):
                # prefer more skeleton pixels, then more mask pixels
                return (int(res.get("_dbg_skel_pixels", 0)),
                        int(res.get("_dbg_bw_pixels", 0)))

            def run_one(path, coating_side):
                cands = []

                # 1) Balanced Top-Hat (stricter defaults)
                r = wr.detect_wrinkles_tophat_edgeband(
                    path=path, y_lower=y0, y_upper=y1,
                    coating_side=coating_side,
                    edge_band_px=100, tophat_rad=8,
                    min_len_px=30, endpoint_dist_px=40,
                    angle_center_deg=45.0, angle_tol_deg=18,
                    small_remove_px=20, super_permissive=False
                )
                cands.append(("TopHatBalanced", r, _score(r)))

                # 2) Strict Top-Hat (fewer false positives)
                r = wr.detect_wrinkles_tophat_edgeband(
                    path=path, y_lower=y0, y_upper=y1,
                    coating_side=coating_side,
                    edge_band_px=80, tophat_rad=10,
                    min_len_px=35, endpoint_dist_px=30,
                    angle_center_deg=45.0, angle_tol_deg=16,
                    small_remove_px=25, super_permissive=False
                )
                cands.append(("TopHatStrict", r, _score(r)))

                # 3) Balanced Sobel (good sensitivity/selectivity)
                r = wr.detect_wrinkles_sobel_band(
                    path=path, y_lower=y0, y_upper=y1,
                    coating_side=coating_side,
                    edge_band_px=120, sobel_sigma=1.2,
                    thr_percentile=75, min_len_px=30,
                    angle_center_deg=45.0, angle_tol_deg=18,
                    small_remove_px=16
                )
                cands.append(("SobelBalanced", r, _score(r)))

                # 4) Gabor (alternative frequency domain)
                r = wr.detect_wrinkles_gabor_band(
                    path=path, y_lower=y0, y_upper=y1,
                    coating_side=coating_side,
                    edge_band_px=280, thetas_deg=(40, 45, 50),
                    frequencies=(0.05, 0.08), min_len_px=30,
                    angle_center_deg=45.0, angle_tol_deg=18,
                    small_remove_px=18
                )
                cands.append(("Gabor", r, _score(r)))

                # 5) Fallback: permissive Sobel (last resort)
                r = wr.detect_wrinkles_sobel_band(
                    path=path, y_lower=y0, y_upper=y1,
                    coating_side=coating_side,
                    edge_band_px=None, sobel_sigma=1.0,
                    thr_percentile=70, min_len_px=20,
                    angle_center_deg=45.0, angle_tol_deg=25,
                    small_remove_px=12
                )
                cands.append(("SobelFallback", r, _score(r)))

                name, best_res, sc = max(cands, key=lambda z: z[2])
                return (name, coating_side, best_res, sc)

            try:
                best_as = run_one(as_path, "left")  # AS: edge at left
                best_bs = run_one(bs_path, "right")  # BS: edge at right

                # light stats just for logging
                print("[WRK] AS skel sum=", best_as[2]["mask_skel"].sum(),
                      " BS skel sum=", best_bs[2]["mask_skel"].sum(),
                      " | chosen:", best_as[0], "/", best_bs[0])

                q.put({"ok": True,
                       "best_as": best_as, "best_bs": best_bs,
                       "y0": y_lo, "y1": y_hi,
                       "as_path": as_path, "bs_path": bs_path})
            except Exception:
                q.put({"ok": False, "error": traceback.format_exc()})

        # --- 7) Start worker thread & kick off poller ---
        t = threading.Thread(target=start_worker, args=(as_path, bs_path, y_lo, y_hi), daemon=True)
        t.start()
        self.after(80, _poll)


class TerminalFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.terminal_head_label = ctk.CTkLabel(self, text="Ausgabe Terminal")
        self.terminal_head_label.grid(row=1, column=1, padx=5, pady=0, sticky="NW")
        self.terminal = ctk.CTkTextbox(self, width=1880, height=120, activate_scrollbars=True)
        self.terminal.grid(row=2, column=1, padx=5, pady=0, sticky="S")

class PlotFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.columnconfigure(1, weight=5)
        self.columnconfigure(2, weight=5)
        self.columnconfigure(3, weight=5)
        self.columnconfigure(4, weight=5)
        self.columnconfigure(5, weight=1)

        # Canvas oben links (sensor 0)
        self.sensor_0_fig = Figure(figsize=(4.5, 3.5), dpi=100)
        self.canvas_sensor_0 = FigureCanvasTkAgg(self.sensor_0_fig, master=self)
        self.canvas_sensor_0.get_tk_widget().grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="NSEW")
        self.label_canvas_sensor_0 = tk.Label(self, text="Plot Sensor 0 (AS)",
                                              background=self.canvas_sensor_0.get_tk_widget()["background"],
                                              font=("Arial", 16))
        self.label_canvas_sensor_0.grid(row=1, column=1, padx=20, pady=10, sticky="NW")
        self.toolbar_s_0_frame = ctk.CTkFrame(master=self, bg_color=self.canvas_sensor_0.get_tk_widget()["background"])
        self.toolbar_s_0_frame.grid(row=1, column=1, padx=10, pady=10, sticky="SE")
        self.toolbar_s_0 = CanvasToolbar(self.canvas_sensor_0, self.toolbar_s_0_frame)
        self.toolbar_s_0.config(background=self.canvas_sensor_0.get_tk_widget()["background"])
        self.toolbar_s_0.update()
        for button in self.toolbar_s_0.winfo_children():
            button.config(background=self.canvas_sensor_0.get_tk_widget()["background"])

        # Canvas oben rechts (sensor 1)
        self.sensor_1_fig = Figure(figsize=(4.5, 3.5), dpi=100)
        self.canvas_sensor_1 = FigureCanvasTkAgg(self.sensor_1_fig, master=self)
        self.canvas_sensor_1.get_tk_widget().grid(row=1, column=3, columnspan=2, padx=10, pady=10, sticky="NSEW")
        self.label_canvas_sensor_1 = tk.Label(self, text="Plot Sensor 1 (Mitte)",
                                              background=self.canvas_sensor_1.get_tk_widget()["background"],
                                              font=("Arial", 16))
        self.label_canvas_sensor_1.grid(row=1, column=3, padx=20, pady=10, sticky="NW")
        self.toolbar_s_1_frame = ctk.CTkFrame(master=self, bg_color=self.canvas_sensor_1.get_tk_widget()["background"])
        self.toolbar_s_1_frame.grid(row=1, column=3, padx=10, pady=10, sticky="SE")
        self.toolbar_s_1 = CanvasToolbar(self.canvas_sensor_1, self.toolbar_s_1_frame)
        self.toolbar_s_1.config(background=self.canvas_sensor_1.get_tk_widget()["background"])
        self.toolbar_s_1.update()
        for button in self.toolbar_s_1.winfo_children():
            button.config(background=self.canvas_sensor_1.get_tk_widget()["background"])

        # Canvas unten links (sensor 2)
        self.sensor_2_fig = Figure(figsize=(4.5, 3.5), dpi=100)
        self.canvas_sensor_2 = FigureCanvasTkAgg(self.sensor_2_fig, master=self)
        self.canvas_sensor_2.get_tk_widget().grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky="NSEW")
        self.label_canvas_sensor_2 = tk.Label(self, text="Plot Sensor 2 (BS)",
                                              background=self.canvas_sensor_2.get_tk_widget()["background"],
                                              font=("Arial", 16))
        self.label_canvas_sensor_2.grid(row=2, column=1, padx=20, pady=10, sticky="NW")
        self.toolbar_s_2_frame = ctk.CTkFrame(master=self, bg_color=self.canvas_sensor_1.get_tk_widget()["background"])
        self.toolbar_s_2_frame.grid(row=2, column=1, padx=10, pady=10, sticky="SE")
        self.toolbar_s_2 = CanvasToolbar(self.canvas_sensor_2, self.toolbar_s_2_frame)
        self.toolbar_s_2.config(background=self.canvas_sensor_2.get_tk_widget()["background"])
        self.toolbar_s_2.update()
        for button in self.toolbar_s_2.winfo_children():
            button.config(background=self.canvas_sensor_2.get_tk_widget()["background"])

        # Canvas unten rechts (opcua)
        self.opcua_fig = Figure(figsize=(4.5, 3.5), dpi=100)
        self.canvas_opcua = FigureCanvasTkAgg(self.opcua_fig, master=self)
        self.canvas_opcua.get_tk_widget().grid(row=2, column=3, columnspan=2, padx=10, pady=10, sticky="NSEW")
        self.label_canvas_opcua = tk.Label(self, text="Plot OPC-UA Daten",
                                           background=self.canvas_opcua.get_tk_widget()["background"],
                                           font=("Arial", 16))
        self.label_canvas_opcua.grid(row=2, column=3, padx=20, pady=10, sticky="NW")
        self.toolbar_opcua_frame = ctk.CTkFrame(master=self, bg_color=self.canvas_opcua.get_tk_widget()["background"])
        self.toolbar_opcua_frame.grid(row=2, column=3, padx=10, pady=10, sticky="SE")
        self.toolbar_opcua = CanvasToolbar(self.canvas_opcua, self.toolbar_opcua_frame)
        self.toolbar_opcua.config(background=self.canvas_opcua.get_tk_widget()["background"])
        self.toolbar_opcua.update()
        for button in self.toolbar_opcua.winfo_children():
            button.config(background=self.canvas_opcua.get_tk_widget()["background"])

        ### Steuerknöpfe
        self.button_start_measurement = ctk.CTkButton(master=self, text="Starte Messung",
                                                      command=lambda: funcs.start_measurement(triggers),
                                                      fg_color="green", hover_color="#026e21")
        self.button_start_measurement.grid(row=3, column=1, padx=20, pady=20, sticky='W')

        self.button_stop_measurement = ctk.CTkButton(master=self, text="Beende Messung",
                                                     command=lambda: funcs.stop_measurement(triggers), fg_color="red",
                                                     hover_color="#9e0303")
        self.button_stop_measurement.grid(row=3, column=2, padx=20, pady=20, sticky="W")

        self.button_start_batch_measurement = ctk.CTkButton(master=self, text="Batchmessung",
                                                            command=lambda: [load_output_directory(),
                                                                             funcs.batch_measurement(
                                                                                 batch_len=self.current_batch_len.get(),
                                                                                 terminal=master.frame_terminal.terminal,
                                                                                 triggers=triggers, master=master),
                                                                             funcs.set_experiment_parameters(self)],
                                                            fg_color="green", hover_color="#026e21")
        self.button_start_batch_measurement.grid(row=3, column=4, padx=20, pady=20, sticky='W')
        self.current_batch_len = ctk.IntVar(self)
        self.current_batch_len.set(vars.default_batch_len)
        self.batch_len_label = ctk.CTkLabel(self, text="Batchgröße (Sekunden): ").grid(row=3, column=3, padx=20,
                                                                                       pady=20, sticky='W')
        self.batch_len_entry = ctk.CTkEntry(self, textvariable=self.current_batch_len, width=80).grid(row=3, column=3,
                                                                                                      padx=20, pady=20,
                                                                                                      sticky='E')

        self.button_create_report = ctk.CTkButton(master=self, text="Ergebnis ausgeben",
                                                  command=lambda: create_report(master))
        self.button_create_report.grid(row=4, column=1, padx=20, pady=20, sticky='W')

        self.button_select_result = ctk.CTkButton(master=self, text="Versuchsergebnisse laden",
                                                  command=lambda: funcs.load_result_directory(self))
        self.button_select_result.grid(row=4, column=2, padx=20, pady=20, sticky='W')

        ## Anzeige der wichtigsten Kalanderparameter des geladenen Versuchs
        self.exp_line_load_label = ctk.CTkLabel(master=self, text="Linienlast (N/mm): ").grid(row=4, column=3, padx=20,
                                                                                              sticky="NW")
        self.exp_line_load_var = ctk.IntVar(self); self.exp_line_load_var.set(vars.exp_line_load)
        self.exp_line_load_value_label = ctk.CTkLabel(master=self, textvariable=self.exp_line_load_var).grid(row=4,
                                                                                                             column=3,
                                                                                                             padx=40,
                                                                                                             sticky="NE")

        self.exp_temperature_label = ctk.CTkLabel(master=self, text="Temperatur (°C): ").grid(row=4, column=3, padx=20,
                                                                                              sticky="W")
        self.exp_temperature_var = ctk.IntVar(self); self.exp_temperature_var.set(vars.exp_temperature)
        self.exp_temperature_value_label = ctk.CTkLabel(master=self, textvariable=self.exp_temperature_var).grid(row=4,
                                                                                                                 column=3,
                                                                                                                 padx=40,
                                                                                                                 sticky="E")

        self.exp_line_speed_label = ctk.CTkLabel(master=self, text="Bahngeschwindigkeit (m/min): ").grid(row=4,
                                                                                                         column=3,
                                                                                                         padx=20,
                                                                                                         sticky="SW")
        self.exp_line_speed_var = ctk.IntVar(self); self.exp_line_speed_var.set(vars.exp_web_speed)
        self.exp_line_speed_value_label = ctk.CTkLabel(master=self, textvariable=self.exp_line_speed_var).grid(row=4,
                                                                                                               column=3,
                                                                                                               padx=40,
                                                                                                               sticky="SE")

        self.exp_web_tension_label = ctk.CTkLabel(master=self, text="Bahnzug (N): ").grid(row=4, column=4, padx=20,
                                                                                          sticky="NW")
        self.exp_web_tension_var = ctk.IntVar(self); self.exp_web_tension_var.set(vars.exp_web_tension)
        self.exp_web_tension_value_label = ctk.CTkLabel(master=self,
                                                        textvariable=self.exp_web_tension_var).grid(row=4, column=4,
                                                                                                     padx=20,
                                                                                                     sticky="NE")

        self.exp_rewinder_tension_label = ctk.CTkLabel(master=self, text="Aufwickler Kraft (N): ").grid(row=4, column=4,
                                                                                                        padx=20,
                                                                                                        sticky="W")
        self.exp_rewinder_tension_var = ctk.IntVar(self); self.exp_rewinder_tension_var.set(vars.exp_rewinder_tension)
        self.exp_rewinder_tension_value_label = ctk.CTkLabel(master=self,
                                                             textvariable=self.exp_rewinder_tension_var).grid(row=4,
                                                                                                              column=4,
                                                                                                              padx=20,
                                                                                                              sticky="E")

        self.exp_unwinder_tension_label = ctk.CTkLabel(master=self, text="Abwickler Kraft (N): ").grid(row=4, column=4,
                                                                                                       padx=20,
                                                                                                       sticky="SW")
        self.exp_unwinder_tension_var = ctk.IntVar(self); self.exp_unwinder_tension_var.set(vars.exp_unwinder_tension)
        self.exp_unwinder_tension_value_label = ctk.CTkLabel(master=self,
                                                             textvariable=self.exp_unwinder_tension_var).grid(row=4,
                                                                                                              column=4,
                                                                                                              padx=20,
                                                                                                              sticky="SE")

        # Anzeigeoptionen
        self.button_switch_to_2D_reduced = ctk.CTkButton(master=self, text="2D-R", width=50, height=50,
                                                         command=lambda: [funcs.update_plots(self, image_mode="2D_reduced")])
        self.button_switch_to_2D_reduced.grid(row=1, column=5, padx=10, pady=10, sticky="N")

        self.button_switch_to_2D_full = ctk.CTkButton(master=self, text="2D-F", width=50, height=50,
                                                      command=lambda: [funcs.update_plots(self, image_mode="2D_full")])
        self.button_switch_to_2D_full.grid(row=1, column=5, padx=10, pady=70, sticky="N")

        self.button_switch_to_3D_reduced = ctk.CTkButton(master=self, text="3D-R", width=50, height=50,
                                                         command=lambda: [funcs.update_plots(self, image_mode="3D_reduced")])
        self.button_switch_to_3D_reduced.grid(row=1, column=5, padx=10, pady=130, sticky="N")

class CreateReport(ctk.CTkToplevel):
    def __init__(self, master, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x300")
        self.attributes('-topmost', 1)
        self.master = master

        self.current_date = ctk.StringVar(self)
        self.current_date.set(vars.default_date)
        self.date_label = ctk.CTkLabel(self, text="Datum (DD.MM.JJJJ):").grid(row=1, column=1, padx=20, pady=20)
        self.date_entry = ctk.CTkEntry(self, textvariable=self.current_date).grid(row=1, column=2, padx=20, pady=20)

        self.starttime = ctk.StringVar(self)
        self.starttime_entry = ctk.CTkEntry(self, textvariable=self.starttime).grid(row=2, column=2, padx=20, pady=20)
        self.starttime_label = ctk.CTkLabel(self, text="Startzeitpunkt (hh,mm,ss):").grid(row=2, column=1, padx=20, pady=20)

        self.endtime = ctk.StringVar(self)
        self.endtime_entry = ctk.CTkEntry(self, textvariable=self.endtime).grid(row=3, column=2, padx=20, pady=20)
        self.endtime_label = ctk.CTkLabel(self, text="Endzeitpunkt (hh,mm,ss):").grid(row=3, column=1, padx=20, pady=20)

        self.button_export_result = ctk.CTkButton(master=self, text="Ergebnisse exportieren",
                                                  command=lambda: [load_output_directory(), self.destroy(),
                                                                   funcs.access_influxDB(self.current_date.get(),
                                                                                         self.starttime.get(),
                                                                                         self.endtime.get(),
                                                                                         mode="manual",
                                                                                         terminal=app.frame_terminal.terminal),
                                                                   funcs.call_output_plots(master,
                                                                                           directory=vars.output_folder)])
        self.button_export_result.grid(row=4, column=2, padx=20, pady=20)

class CanvasToolbar(NavigationToolbar2Tk):
    def set_message(self, s):
        pass

class WrinkleDetectionPlotWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Plot Window")
        self.geometry("1000x600")
        self.canvas_dict = {}
        self.figure_dict = {}
        self.canvas_list = []
        self.figure_list = []
        self.create_canvases()

    def create_canvases(self):
        for i in range(2):
            for j in range(3):
                fig = Figure(figsize=(4, 3), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=self)
                widget = canvas.get_tk_widget()
                widget.grid(row=i, column=j, padx=10, pady=10)
                self.canvas_list.append(canvas)
                self.figure_list.append(fig)

############################################################################################
### Menüfunktionen ###
def set_export_directory(self):
    folder_selected = filedialog.askdirectory(title="Ordner für CSV-Export auswählen")
    if folder_selected:
        self.output_folder = folder_selected
        print(f"Exportverzeichnis ausgewählt: {self.output_folder}")
        config.set('Arbeitsverzeichnisse', 'main_output_folder', self.output_folder)
        with open('config.ini', 'w') as cfg:
            config.write(cfg)
    else:
        messagebox.showwarning("Einstellungen", "Kein Exportverzeichnis ausgewählt")

########################################################################################
### GUI Funktionen ###
def create_report(self):
    if self.report_window is None or not self.report_window.winfo_exists():
        self.report_window = CreateReport(self)
    else:
        self.report_window.focus()
    time.sleep(0.1)
    self.report_window.focus()

# Ausgabeverzeichnis für Ergebnisexport
def load_output_directory():
    path = config.get('Arbeitsverzeichnisse', 'main_output_folder')
    newpath = path.replace('/', '\\')
    datestring = str(date.today())
    datepath = newpath + '\\' + datestring
    datepath.encode('unicode-escape').decode()

    if not os.path.exists(datepath):
        config.set('Arbeitsverzeichnisse', 'result_number_in_directory', str(1))
        with open('config.ini', 'w') as cfg:
            config.write(cfg)
        current_folder_in_directory = int(config.get('Arbeitsverzeichnisse', 'result_number_in_directory'))
        os.makedirs(datepath)
        resultpath = datepath + f"\\{datestring}_Versuch_{current_folder_in_directory}"
        os.makedirs(resultpath)
        vars.output_folder = resultpath
        config.set('Arbeitsverzeichnisse', 'result_number_in_directory', str(current_folder_in_directory + 1))
        with open('config.ini', 'w') as cfg:
            config.write(cfg)
    else:
        current_folder_in_directory = int(config.get('Arbeitsverzeichnisse', 'result_number_in_directory'))
        resultpath = datepath + f"\\{datestring}_Versuch_{current_folder_in_directory}"
        os.makedirs(resultpath)
        vars.output_folder = resultpath
        config.set('Arbeitsverzeichnisse', 'result_number_in_directory', str(current_folder_in_directory + 1))
        with open('config.ini', 'w') as cfg:
            config.write(cfg)

############################################################################################
# Zweites Fenster zum Erstellen eines Reports
if __name__ == '__main__':
    app = AutPUT()
    app.mainloop()
