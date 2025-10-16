import os
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter, maximum_filter, laplace, binary_dilation, label, convolve, sobel, \
    binary_fill_holes, binary_erosion, gaussian_filter1d
import time
import matplotlib.pyplot as plt
from skimage.morphology import binary_opening, rectangle, opening, binary_closing, remove_small_objects, \
    skeletonize  # ,binary_erosion
from skimage.measure import regionprops_table
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from itertools import combinations
import functions as funcs


def detect_coating_edge(input_dataframe, show_plot=False):  # OPTIMIZED
    """
    Detects the coating edge in a height map using Laplacian-like 2nd derivative analysis.

    Parameters:
        data (DataFrame): Input data with height values.
        show_plot (bool): Whether to display diagnostic plots.

    Returns:
        scnd_derivative (ndarray): Second derivative of the smoothed column mean profile.
        height_map (ndarray): Normalized height map.
        x_of_coating_edge (int): Estimated column index of the coating edge.
        runtime (float): Processing time in seconds.
    """

    # --- Parameters ---
    smoothing_sigma = 3
    threshold_drop_1stdiff = 0.03  # unused currently but kept for future use
    threshold_drop_2nddiff = 0.01  # unused currently but kept for future use
    x_offset = 5
    min_column = 15

    start_time = time.time()

    # --- Step 1: Normalize height map ---
    height_map = input_dataframe.values
    norm_map = (height_map - np.mean(height_map)) / np.std(height_map)

    # --- Step 2: Compute column-wise mean and smooth it ---
    column_mean = norm_map.mean(axis=0)
    smoothed_mean = gaussian_filter1d(column_mean, sigma=smoothing_sigma)

    # --- Step 3: First and second derivatives ---
    first_derivative = np.diff(smoothed_mean)
    scnd_derivative = np.diff(first_derivative)

    # --- Step 4: Detect coating edge by minimum of 2nd derivative ---
    scnd_derivative[:min_column] = 0  # mask out early noisy region
    x_of_coating_edge = np.argmin(scnd_derivative) + x_offset

    # --- Compute runtime ---
    runtime = round(time.time() - start_time, 2)

    # --- Optional: Plotting ---
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Second derivative
        ax1.plot(scnd_derivative, label='Second Derivative')
        ax1.axhline(-0.03, color='gray', linestyle='--', label='Drop Threshold')
        ax1.axvline(x_of_coating_edge, color='red', linestyle='--', label='Detected Edge')
        ax1.set_title("Second Derivative of Smoothed Intensity Profile")
        ax1.set_xlabel("Column Index")
        ax1.set_ylabel("2nd Derivative Value")
        ax1.legend()

        # Plot 2: Height map with edge overlay
        im = ax2.imshow(height_map, cmap='gray')
        ax2.axvline(x=x_of_coating_edge, color='blue', linestyle='--', label=f'Wrinkle Start (col {x_of_coating_edge})')
        ax2.set_title("Height Map with Detected Edge")
        ax2.legend()

        # Colorbar
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Height')

        plt.tight_layout()
        if show_plot:
            plt.show()
        else:
            try:
                plt.close(fig)
            except Exception:
                plt.close()

    return scnd_derivative, height_map, x_of_coating_edge, runtime


def equalize_rows(input_dataframe, x_coating_edge):
    # 1️⃣ Add this line immediately after the def:
    x_coating_edge = int(round(float(x_coating_edge)))   # <--- add this line

    # 2️⃣ Keep your old line but now it uses the new int variable:
    right_border_of_averaging = int(x_coating_edge + (340 - x_coating_edge) / 2)

    gamma = 1.4
    alpha = 7
    beta = 0.4

    epsilon = 1.5

    average_value_of_df = input_dataframe.values.mean()
    print(average_value_of_df)
    rowwise_avg_right_of_coating_edge = input_dataframe.iloc[:, x_coating_edge + 1:right_border_of_averaging].mean(
        axis=1)
    correction_factors = average_value_of_df / rowwise_avg_right_of_coating_edge.replace(0, pd.NA)
    correction_factors = np.where(correction_factors < 0.1, 0.1, correction_factors)
    correction_factors = np.where(correction_factors > 5, 5, correction_factors)

    scaled_df = input_dataframe.mul(correction_factors, axis=0)
    scaled_df_trimmed = scaled_df.iloc[:, x_coating_edge + 1:]

    added_df = input_dataframe.iloc[:, x_coating_edge + 1:].add(scaled_df_trimmed)

    scaled_added_df = (added_df - added_df.min().min()) / (added_df.max().max() - added_df.min().min())
    # scaled_added_df = added_df

    final_df_gamma = scaled_added_df ** gamma
    final_df_sigmoid = 1 / (1 + np.exp(-alpha * (scaled_added_df - beta)))
    final_df_exponential = np.expm1(epsilon * scaled_added_df) / np.expm1(epsilon)
    '''
    # Compute row-wise mean from column index 1 to x (inclusive)
    rowwise_avg_left_of_coating_edge = input_dataframe.iloc[:, 1:x_coating_edge].mean(axis=1)
    rowwise_avg_right_of_coating_edge = input_dataframe.iloc[:, x_coating_edge+1:right_border_of_averaging].mean(axis=1)

    # Compute inverse of row-wise average left of tghe coating edge (handle division by zero safely)
    inverse_avg_left_of_coating_edge = 1/rowwise_avg_left_of_coating_edge.replace(0, pd.NA)
    inverse_avg_left_of_coating_edge = np.where(inverse_avg_left_of_coating_edge < 0.1, 0.1, inverse_avg_left_of_coating_edge)
    inverse_avg_left_of_coating_edge = np.where(inverse_avg_left_of_coating_edge > 5, 5, inverse_avg_left_of_coating_edge)

    # Compute inverse of row-wise average right of tghe coating edge (handle division by zero safely)
    inverse_avg_right_of_coating_edge = 1/rowwise_avg_right_of_coating_edge.replace(0, pd.NA)
    inverse_avg_right_of_coating_edge = np.where(inverse_avg_right_of_coating_edge < 0.1, 0.1, inverse_avg_right_of_coating_edge)
    inverse_avg_right_of_coating_edge = np.where(inverse_avg_right_of_coating_edge > 5, 5, inverse_avg_right_of_coating_edge) 

    scaled_df_left = input_dataframe.mul(inverse_avg_left_of_coating_edge, axis=0)
    scaled_df_right = input_dataframe.mul(inverse_avg_right_of_coating_edge, axis=0)

    #scaled_df_right_equ = pd.DataFrame(funcs.equalize_row_height(scaled_df_right,2))

    scaled_df_trimmed_equalized = funcs.equalize_row_height(scaled_df_right.iloc[:, x_coating_edge+1:], 1)
    '''
    # Create subplots: 1 row, 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    # --- Plot 1: Grayscale heightmap of the scaled DataFrame ---
    im = ax1.imshow(final_df_gamma, cmap='gray', aspect='auto')
    ax1.set_title("Heightmap of Scaled DataFrame")
    ax1.set_xlabel("Columns")
    ax1.set_ylabel("Rows")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Value")

    # --- Plot 1: Grayscale heightmap of the scaled DataFrame ---
    im = ax2.imshow(final_df_sigmoid, cmap='gray', aspect='auto')
    ax2.set_title("Heightmap of Scaled DataFrame")
    ax2.set_xlabel("Columns")
    ax2.set_ylabel("Rows")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Value")

    # --- Plot 1: Grayscale heightmap of the scaled DataFrame ---
    im = ax3.imshow(final_df_exponential, cmap='gray', aspect='auto')
    ax3.set_title("Heightmap of Scaled DataFrame")
    ax3.set_xlabel("Columns")
    ax3.set_ylabel("Rows")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Value")
    '''
    # --- Plot 2: Row-wise average and inverse ---
    ax2.plot(correction_factors, label='Row-wise Average left')
    #ax2.plot(inverse_avg_left_of_coating_edge, label='Inverse of Average left')
    ax2.set_title(f'Row-wise Avg and Inverse LEFT(Cols 1 to {x_coating_edge})')
    ax2.set_xlabel("Row Index")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)

    # --- Plot 2: Row-wise average and inverse ---
    ax3.plot(rowwise_avg_right_of_coating_edge, label='Row-wise Average right')
    ax3.plot(inverse_avg_right_of_coating_edge, label='Inverse of Average right')
    ax3.set_title(f'Row-wise Avg and Inverse RIGHT(Cols {x_coating_edge} to 340)')
    ax3.set_xlabel("Row Index")
    ax3.set_ylabel("Value")
    ax3.legend()
    ax3.grid(True)
    '''
    # Improve layout and show
    # Improve layout and close (no popups)
    plt.tight_layout()
    try:
        plt.close(fig)
    except Exception:
        plt.close()

    return final_df_gamma, final_df_sigmoid, final_df_exponential


def calculate_median_mean_diff_uncoated(data, x_of_coating_edge):
    median = np.median(data.iloc[x_of_coating_edge:, :])
    average_height = np.mean(data.iloc[x_of_coating_edge:, :])
    median_mean_diff_uncoated = average_height - median
    return np.round(median_mean_diff_uncoated, 2)


def laengsfalten_detection_top_hat(data, show_plot=False):
    """
    führt die Längsfalten Detektion mit dem Top-Hat-Filter aus und gibt eine vorverarbeteite Maske und die Laufzeit zurück

    Args:
        data: Aufnahme des Profils
        show_plot: boolean ob Vorgang geplottet werden soll

    Returns:
        mask_Längsfalte_closed: Maske der erkannten Längsfalten
        runtime: Laufzeit
    """
    print("Erkennung der Längsfalten mittels Top-hat-Filter...")
    start_time = time.time()
    data = data.to_numpy()
    # schwache Erhöhungen elimineren
    data = opening(data, rectangle(4, 4))
    # Top-hat Operation durchführen
    opened_data = opening(data, rectangle(3, 60))
    # subtrahiere den Datensatz ohne kleine Strukturen von dem Ausgangsdatensatz
    data_top_hat = data - opened_data

    # alle Pixel deren Wert größer als der Schwellwert ist werden Defekten zugeordnet
    threshold_Längsfalte = 0.0000001
    mask_Längsfalte = data_top_hat > threshold_Längsfalte

    # Pixel die Längsfalten zugeordnet werden, jedoch viel zu weit links liegen um tatsächlich Längsfaltes zu zeigen werden auf 0 gesetzt
    mask_Längsfalte[:, :70] = 0

    # durch Opening werden kleine Elemente und Ausreißer gelöscht, (oder so verkleinert dass sie im späteren Verlauf des Programms gelöscht werden)
    #  die aufgrund ihrer Form und Ausdehnung keinen Längsfalte darstellen können
    mask_Längsfalte_opened = binary_opening(mask_Längsfalte, rectangle(3, 3))
    mask_Längsfalte_opened = binary_opening(mask_Längsfalte_opened, rectangle(10, 2))
    mask_Längsfalte_opened = remove_small_objects(mask_Längsfalte_opened, min_size=65000)
    mask_Längsfalte_closed = binary_closing(mask_Längsfalte_opened, rectangle(45, 1))

    runtime = np.round(time.time() - start_time, 2)
    if show_plot:
        data_dict = {
            # "Original Data": {"data": data, "3d": False, "distortion_free": True, "colourful": False},
            "top Hat Data": {"data": data_top_hat, "3d": False, "distortion_free": False, "colourful": False,
                             "endpoints": None, "labels": None},
            "mask Längsfalte": {"data": mask_Längsfalte, "3d": False, "distortion_free": False, "colourful": False,
                                "endpoints": None, "labels": None},
            # "mask Längsfalte opened": {"data": mask_Längsfalte_opened, "3d": False, "distortion_free": False, "colourful": False, "endpoints":None, "labels": None},
            "mask Längsfalte closed": {"data": mask_Längsfalte_closed, "3d": False, "distortion_free": False,
                                       "colourful": False, "endpoints": None, "labels": None},
        }
        # Aufruf der Plot-Funktion
        plot_subplots(data_dict, plot_in_mm=False, one_row=True)
    return mask_Längsfalte_closed, runtime


def wrinkle_detection_top_hat(data, coating_edge, show_plot=False):
    """
    führt die Wrinkle-Detektion mit dem Top-Hat-Filter aus und gibt eine vorverarbeteite Maske und die Laufzeit zurück

    Args:
        data: Aufnahme des Profils
        show_plot: boolean ob Vorgang geplottet werden soll

    Returns:
        mask_wrinkle_opened: Maske der detektierten Wrinkles
        runtime. Laufzeit
    """
    print("Erkennung der Defekte mittels Top-Hat-Filter...")
    coating_edge = coating_edge - coating_edge
    start_time = time.time()
    data = data.to_numpy()

    # Top-hat Operation durchführen
    # opened_data = opening(data, rectangle(10,25))
    opened_data = opening(data, rectangle(10, 25))
    data_top_hat = data - opened_data

    # alle Pixel deren Wert größer als der Schwellwert ist werden Defekten zugeordnet
    threshold_wrinkle = 0.0005  # 0.0005
    mask_wrinkle = data_top_hat > threshold_wrinkle

    # Pixel die Wrinkles zugeordnet werden, jedoch viel zu weit links liegen um tatsächlich wrinkles zu zeigen werden auf 0 gesetzt
    mask_wrinkle[:, :coating_edge] = 0

    # durch Opening werden kleine Elemente und Ausreißer gelöscht, (oder so verkleinert dass sie im späteren Verlauf des Programms gelöscht werden)
    #  die aufgrund ihrer Form und Ausdehnung keinen wrinkle darstellen können
    mask_wrinkle_opened = binary_opening(mask_wrinkle, rectangle(3, 3))  # 3,3
    mask_wrinkle_opened = remove_small_objects(mask_wrinkle_opened, min_size=100)  # 4000
    runtime = np.round(time.time() - start_time, 2)
    if show_plot:
        data_dict = {
            # "Original Data": {"data": data, "3d": False, "distortion_free": True, "colourful": False},
            "top Hat Data": {"data": data_top_hat * 10, "3d": False, "distortion_free": False, "colourful": False,
                             "endpoints": None, "labels": None},
            "mask wrinkles": {"data": mask_wrinkle, "3d": False, "distortion_free": False, "colourful": False,
                              "endpoints": None, "labels": None},
            "mask wrinkles opened": {"data": mask_wrinkle_opened, "3d": False, "distortion_free": False,
                                     "colourful": False, "endpoints": None, "labels": None},
        }
        # Aufruf der Plot-Funktion
        plot_subplots(data_dict, plot_in_mm=False, one_row=True)
    return mask_wrinkle_opened, runtime


def get_regression_angle_and_deviation(coords_list):
    """
    bestimmt mit Hilfe der Ausgleichsgeraden verschiedene Messgrößen eines Objektes

    Args:
        coords_list: Liste der Koordinaten die dem Objekt zugeordnet wurden

    returns:
        angle: Winkel der Ausgleichsgeraden in Grad
        mean_residual: mittlere Abweichung der Ausgleichsgeraden zum tatsächlichen Verlauf
        slope: Steigung der Ausgleichsgeraden

        """
    # Listen für x und y extrahieren
    x = []
    y = []
    for coord in coords_list:
        x.append(coord[0])
        y.append(coord[1])
    if len(coords_list) < 2 or np.all(x == x[0]) or np.all(y == y[0]) or np.std(x) == 0:
        angle, mean_residuals, slope = 0, 0, 0
    else:
        # Lineare Regression durchführen
        result = stats.mstats.linregress(x, y)
        slope = result.slope
        intercept = result.intercept

        # Regressionsgeraden-Werte berechnen
        y_pred = [slope * xi + intercept for xi in x]  # y-Werte der Regressionsgeraden
        residuals = [abs(yi - ypi) for yi, ypi in zip(y, y_pred)]  # Residuen (Abweichungen)
        mean_residuals = np.mean(residuals)
        # Winkel der Regressionsgeraden berechnen
        angle = np.arctan(1 / np.abs(slope)) * 180 / np.pi + 90

    return angle, mean_residuals, slope


def get_neighbors(skeleton):
    """
    zählt die Nachbarn eines jeden Pixels des Skelettes

    Arg:
        skeleton: Maske des Skelettes

    returns:
        neighbors: Matrix mit der Maske des skelettes, jedem Pixel wird die Anzahl seiner Nahcbarn zugewiesen
    """
    # Kernel zur Nachbarschaftsberechnung
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Anzahl der Nachbarn berechnen
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant')

    return neighbors


def delete_small_branches(skeleton, min_length):
    """
    löschte kleine Äste des Skelettes

    Args:
        skeleton: Maske des Skelettes welches verarbeitet wird
        min_length: Länge ab der Äste nciht mehr gelöscht werden

    return:
        cleaned_skeleton: Skelett ohne kleine Äste

    """
    neighbors = get_neighbors(skeleton)
    # Endpunkte sind Pixel der Maske die nur einen Nachbar der ebenfalls Teil der Maske ist besitzen
    endpoints = (skeleton & (neighbors == 1))

    # Kopie des Skeletts erstellen
    cleaned_skeleton = skeleton.copy()

    # Indizes der Endpunkte finden
    end_coords = np.argwhere(endpoints)

    # Für jeden Endpunkt prüfen
    for end in end_coords:
        path = []  # Liste der Pixel des aktuellen Pfades
        current = tuple(end)
        prev = None

        # Verfolge den Pfad zurück zum nächsten Verzweigungspunkt
        while True:
            path.append(current)
            neighbors_coords = []

            # Nachbarn finden
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
                           (1, 1)]:  # Schritte der Indizes um auf mögliche Nachbarn zu prüfen
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < skeleton.shape[0] and
                        0 <= neighbor[1] < skeleton.shape[1] and
                        skeleton[neighbor] and neighbor != prev):
                    neighbors_coords.append(neighbor)

            # Wenn mehr als 2 Nachbarn: Verzweigungspunkt erreicht
            if len(neighbors_coords) != 1:
                break

            # Weiter zum nächsten Pixel
            prev = current
            current = neighbors_coords[0]

        # Pfad analysieren
        if len(path) < min_length:
            # Pfad löschen, wenn er zu kurz ist
            for coord in path:
                cleaned_skeleton[coord] = 0
    return cleaned_skeleton


def measure(data, mask_wrinkle, x_coating_edge, mask_längsfalte=None, file_name="kein Dateiname übergeben",
            save_results=False, show_plot=False, show_unknown_defects=True):
    """
    führt die Vermessung der mit dem Top-Hat-Filter detektierten Wrinkles und den mit der Kombination aus Top-Hat und Sobelfilter detektierten
    Längsfalten durch

    Args:
        data: Aufnahme des Profils
        mask_wrinkle: Maske der Wrinkles
        x_coating_edge: X-koordinate der Beschichtungskante
        mask_Längsfalte: Makse der Längsfaltenerkennung
        file_name: Name der Datei unter der die Plots gespeichert werden sollen
        save_results: boolean ob Plots gespiechert werden sollen
        show_plot: boolean ob Plots gezeigt werden sollen
        show_unknown_defects: boolean ob Defekte die weder Längsfalten noch Wrrnkles zugeordnet werden können angezeigt werden sollen
        show_plot: boolean ob Vorgang geplottet werden soll

    Returns:
        table: Tabelle mit den Ergebnissen
        runtime: Laufzeit
    """
    print('Vermessung der erkannten Defekte...')
    start_time = time.time()
    if mask_längsfalte is not None:
        mask = np.logical_or(mask_wrinkle, mask_längsfalte).astype(int)
    else:
        mask = mask_wrinkle
    labeled_array, num_features = label(mask)

    # Bild für Hüllen initialisieren
    all_hulls = np.zeros_like(mask, dtype=bool)

    # Dataframe mit Eigenschaften der gelabelten Objekte erstellen
    regions = regionprops_table(labeled_array, intensity_image=np.array(data), properties=(
    'area', 'coords', 'label', 'bbox', 'image', 'centroid', 'intensity_mean', 'intensity_max', 'intensity_min'),
                                cache=True)
    # Zählvariablen für Objektnummern initialisieren
    wrinkle_number = 0
    falten_number = 0
    unknown_number = 0
    max_dis_wrinkle = 0

    # Variabeln für Ergebniss initialisieren
    rows = []
    labels_data = []
    all_ends = []
    dis_max = []
    covered_volume_wrinkles = 0
    covered_volume_längsfalte = 0

    # mm_per_pixel für Angabe der Messwerte in metrischen Größen holen
    mm_per_pixel_x, mm_per_pixel_y = get_measurement_data()

    # iteriere über jeden möglichen Defekt, die Objekte weren einzeln ausgwertet und im Anschluss wieder zusammegführt (außer Objekte die nciht Defekten zugeordnet werden)
    for i in regions['label']:
        centroid_y = regions["centroid-0"][i - 1]
        centroid_x = regions["centroid-1"][i - 1]
        minr, minc, maxr, maxc = regions['bbox-0'][i - 1], regions['bbox-1'][i - 1], regions['bbox-2'][i - 1], \
        regions['bbox-3'][i - 1],
        b_height = (maxr - minr) * mm_per_pixel_y
        b_width = (maxc - minc) * mm_per_pixel_x
        defect_height = np.round(regions['intensity_max'][i - 1] - regions['intensity_min'][i - 1], 2)
        area = regions["area"][i - 1] * mm_per_pixel_x * mm_per_pixel_y
        # zu kleine oder zu niedrige Objekte  werden gelöscht
        if b_height < 2 or area < 1 or defect_height < 0.045:  # 4,5,0.045
            continue
        else:
            # jedes einzelne Objekt skelettieren
            skeleton = skeletonize(regions["image"][i - 1])
            # Verzweigungen des Skeletts löschen
            min_length = 10  # 50
            skeleton = delete_small_branches(skeleton, min_length)

            # Winkel und mittlere Abweichung von approximierter Geraden berechnen
            angle, mean_residual, slope = get_regression_angle_and_deviation(np.argwhere(skeleton == 1))
            distance_to_coating = (maxc - x_coating_edge) * mm_per_pixel_x
            covered_volume = area * (regions["intensity_mean"][i - 1] - regions["intensity_min"][i - 1])

            if b_height > 56 and area > 30 and b_width > 0.4 and defect_height > 0.1 and angle > 173 and centroid_x > x_coating_edge - 10:
                obj_type = "Längsfalte"
                falten_number += 1
                obj_number = falten_number
                covered_volume_längsfalte += covered_volume
                min_length = 1000
                skeleton = delete_small_branches(skeleton, min_length)
            elif distance_to_coating < 5.7 and slope < 0 and angle < 178.8 and centroid_x > x_coating_edge + 10 and b_height < 60 and area < 64:
                # zu slope: wenn die Ausgleichsgerade von unten links nach rechts oben geht, kann es sich nicht um einen Wrinkle handeln -> diese Wrinkles werden gelöscht
                obj_type = "Wrinkle"
                covered_volume_wrinkles += covered_volume
                wrinkle_number += 1
                obj_number = wrinkle_number
                dis_max.append(distance_to_coating)
            elif show_unknown_defects:
                obj_type = "unb. Def."
                unknown_number += 1
                obj_number = unknown_number
            else:
                continue

            # Endpunkte des verzweigungsarmen Skeletts bestimmen
            neighbors = get_neighbors(skeleton)
            endpoints = (skeleton & (neighbors == 1))
            # Indizes der Endpunkte extrahieren
            end_coords = np.argwhere(endpoints)
            # Abstand zwischen den Endpunkten berechnen

            try:
                end_coords_cleaned = [None, None]
                if len(end_coords) < 2:
                    end_coords_cleaned = [(0, 0), (0, 0)]
                else:
                    index_combinations = list(combinations(list(range(0, len(end_coords))), 2))
                    distance = 0
                    for pairs in index_combinations:
                        index_1, index_2 = pairs
                        delta_x = abs(end_coords[index_1][1] - end_coords[index_2][1])
                        delta_y = abs(end_coords[index_1][0] - end_coords[index_2][0])
                        distance_temp = np.sqrt((delta_x * mm_per_pixel_x) ** 2 + (delta_y * mm_per_pixel_y) ** 2)
                        if distance_temp > distance:
                            distance = distance_temp
                            end_coords_cleaned[0], end_coords_cleaned[1] = end_coords[index_1], end_coords[index_2]

            except IndexError as e:
                print("IndexError occurred during distance calculation:", e)
                print(f"end_coords: {end_coords}")
                distance = 0

            end_coords_cleaned = [(x + minr, y + minc) for x, y in end_coords_cleaned]

            all_ends.append(end_coords_cleaned)

            # einfügen der Label in den Plot
            labels_data.append((centroid_x, centroid_y, obj_type, obj_number))

            # Hülle der Defekte erzeugen
            # durch Erosion die Ränder der gelabten Bereiche löschen
            eroded = binary_erosion(regions["image"][i - 1], structure=rectangle(5, 5))
            # das erodierte Bild vom nicht erodierten Bild abziehen um nur die Ränder zu erhalten
            hull = regions["image"][i - 1] ^ eroded

            # alle Hüllen zu einem Binärbild zusammenfügen
            all_hulls[minr:maxr, minc:maxc] |= hull

            # gemessene Größen an Ergebnisliste anhängen
            rows.append({
                "    Objekttyp    ": obj_type,
                "Objektnummer": obj_number,
                "bbox Breite [mm]": np.round(b_width, 1),
                "bbox Höhe [mm]": np.round(b_height, 1),
                "Fläche [mm^2]": np.round(area, 2),
                "Winkel [°]": np.round(angle, 2),
                "Kurvigkeit": np.round(mean_residual, 2),
                "Länge [mm]": np.round(distance, 2),
                "Defekthöhe [mm]": defect_height,
                "max. Abstand Beschichtung [mm]": np.round(distance_to_coating, 1),
                "eingeschlossenes Volumen [mm^3]": np.round(covered_volume, 2),
            })

    # Ergebnisliste in Tabelle umwandeln
    table = pd.DataFrame(rows)
    if wrinkle_number > 5:
        max_dis_wrinkle = np.round(np.percentile(dis_max, 80), 2)
    else:
        max_dis_wrinkle = "zu wenige Wrinkles um Aussage treffen zu können"

    # Tabelle auf mindestens drei Zeilen auffüllen
    if len(table) < 3:
        mising_rows = 3 - len(table)
        for _ in range(mising_rows):
            table = pd.concat([table, pd.DataFrame([[""] * len(table.columns)], columns=table.columns)],
                              ignore_index=True)

    table['Anzahl der erkannten Defekte (Top_Hat)'] = [f"Wrinkles:  {wrinkle_number}"] + [
        f"Längsfalten:  {falten_number}"] + [f"Unbekannte:  {unknown_number}"] + [""] * (len(table) - 3)
    table['max Abstand Beschichtung (80% Perzentil) [mm]'] = [max_dis_wrinkle] + [""] * (len(table) - 1)

    # gewichtete Summe als Indexbewertung verwenden
    factor_volume_wrinkles = 3
    factor_volume_längsfalten = 1
    table['Grad der Defekte        '] = [f"Gewichtung Längsfalten:  {factor_volume_längsfalten}"] + [
        f"Gewichtung Wrinkles:  {factor_volume_wrinkles}"] + [(
                                                                          covered_volume_längsfalte * factor_volume_längsfalten + covered_volume_wrinkles * factor_volume_wrinkles) / (
                                                                          factor_volume_längsfalten + factor_volume_wrinkles)] + [
                                            ""] * (len(table) - 3)

    # Liste mit Endpunkten in numpy array umwandeln, Liste flach machen (falls verschachtelt)
    if any(isinstance(elem, list) or isinstance(elem, np.ndarray) for elem in all_ends):
        all_ends_flat = [coord for sublist in all_ends for coord in sublist]
    else:
        all_ends_flat = all_ends
    all_ends = np.array(all_ends_flat)

    # Erstelle eine RGB-Darstellung des Grauwertbilds (um Hüllen und Endpunkte farbig darstellen zu können)
    data_plus = data + abs(np.min(data))
    data_plus = data_plus / np.max(data_plus)  # Normierung auf [0, 1]
    colored_hulls = np.stack([data_plus] * 3, axis=-1)  # Grauwertbild zu RGB erweitern

    hulls_color = [1, 0, 0]  # rot
    for i, color in enumerate(hulls_color):
        colored_hulls[all_hulls == 1, i] = color

    data_dict = {
        "Top-Hat, Label": {"data": data, "3d": False, "distortion_free": False, "colourful": False,
                           "endpoints": all_ends,
                           "labels": [(centroid_x, centroid_y, f"{obj_type}: {obj_number}") for
                                      centroid_x, centroid_y, obj_type, obj_number in labels_data]},
        "Top-Hat, Umrisse der Defekte": {"data": colored_hulls, "3d": False, "distortion_free": False,
                                         "colourful": True, "endpoints": None,
                                         "labels": [(centroid_x, centroid_y, f"{obj_number}") for
                                                    centroid_x, centroid_y, obj_type, obj_number in labels_data]},
    }
    # Aufruf der Plot-Funktion
    plot_subplots(data_dict, file_name=file_name, save_results=save_results, plot_in_mm=True, one_row=True,
                  show_plot=show_plot)  # x_coating_edge,
    runtime = (round(time.time() - start_time, 2))
    return table, runtime


def get_measurement_data():
    detection_rate = 100  # in Hz
    speed = 1  # in m/min
    mm_per_pixel_y = speed / detection_rate * 1000 / 60  # in mm

    measuring_width = 15  # in mm
    number_of_pixels = 800
    mm_per_pixel_x = measuring_width / number_of_pixels  # in mm
    return mm_per_pixel_x, mm_per_pixel_y


def plot_subplots(data_dict, x_coating_edge=None, draw_plane=False, one_row=False, plot_in_mm=True, save_results=False,
                  show_plot=False, Fourier=False, file_name="kein Dateiname übergeben"):
    """
    Plotte beliebig viele Subplots, verschiedene Einstellmöglichkeiten, Speichermöglichkeit

    Args:
        data_dict (dict): Ein dictionary mit folgendem Format:
                          - "data": Die Daten für den Plot (2D-Array).
                          - "3d": Boolean, ob der Plot in 3D (True) oder 2D (False) gezeichnet werden soll.
                          - "distortion_free": Boolean, ob der Plot verzerrungsfrei dargestellt werden soll.
                          - "colourful": Boolean, ob der Plot farbig sein soll.
                          - "endpoints": Koordinaten der Endpunkte.
                          - "Label": Koordinaten und Name des Labels
        x_coating_edge (float, optional): Die X-Position der Beschichtungskante. Falls None, wird die Ebene nicht gezeichnet.
        draw_plane (bool, optional): Ob die durch `x_coating_edge` definierte Ebene gezeichnet wird.
        one_row (bool, optional): Ob alle Subplots in einer einzigen Zeile dargestellt werden.
        plot_in_mm (bool, optional): Ob die Achsenbeschriftung in mm erfolgen soll.
        save_results (bool, optional): Ob die Plots als Datei gespeichert werden sollen.
        file_name (str, optional): Name der Datei für gespeicherte Plots.
    """
    if plot_in_mm:
        mm_per_pixel_x, mm_per_pixel_y = get_measurement_data()
        unit = "mm"
    else:
        mm_per_pixel_x = 1
        mm_per_pixel_y = 1
        unit = "pixels"

    num_plots = len(data_dict)

    if one_row:
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        rows = 1
        cols = num_plots
    else:
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = int(np.ceil(num_plots / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if num_plots == 1:
        axes = [axes]  # Achsen in eine Liste packen, wenn nur ein Plot vorhanden ist.
    else:
        axes = axes.flatten()

        # alle subplots durchgehen un dEigenschaften festlegen
    for i, (title, plot_info) in enumerate(data_dict.items()):

        ax = axes[i]
        data = plot_info["data"]
        is_3d = plot_info["3d"]
        is_colourful = plot_info["colourful"]
        is_distortion_free = plot_info["distortion_free"]
        endpoints = plot_info["endpoints"]
        labels = plot_info["labels"]
        data = np.array(data)
        if is_colourful:
            if Fourier:
                cmap = "turbo"
            else:
                cmap = 'viridis'

        else:
            cmap = 'gray'

        if is_distortion_free:
            aspect = 'equal'
        else:
            aspect = 'auto'

        if is_3d:
            # 3D-Plot
            ax.remove()
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            x = np.arange(0, (data.shape[1] * mm_per_pixel_x), mm_per_pixel_x)
            y = np.arange(0, (data.shape[0] * mm_per_pixel_y), mm_per_pixel_y)
            x, y = np.meshgrid(x, y)
            ax.plot_surface(x, y, data, cmap="viridis")
            if draw_plane and x_coating_edge is not None:
                y_vals = np.linspace(*ax.get_ylim(), 100)
                z_vals = np.linspace(*ax.get_zlim(), 100)
                Y_plane, Z_plane = np.meshgrid(y_vals, z_vals)
                ax.plot_surface(x_coating_edge * mm_per_pixel_x, Y_plane, Z_plane, color="black", alpha=0.7)
                mid_y = np.mean(ax.get_ylim())
                mid_z = np.mean(ax.get_zlim())
                ax.text(x_coating_edge * mm_per_pixel_x, mid_y, mid_z, "Beschichtungskante", color="blue", fontsize=10,
                        rotation=90)
            ax.set_title(title)
            ax.set_xlabel(f"X-axis [{unit}]")
            ax.set_ylabel(f"Y-axis [{unit}]")
            ax.set_zlabel(f"Z-axis [{unit}]")
        else:
            # 2D-Plot
            x_pixels = data.shape[1]
            y_pixels = data.shape[0]
            extent = [0, x_pixels * mm_per_pixel_x, 0, y_pixels * mm_per_pixel_y]
            ax.imshow(data, cmap=cmap, aspect=aspect, origin="lower", interpolation="none", extent=extent)
            if endpoints is not None:
                if endpoints.size != 0:
                    ax.scatter(endpoints[:, 1] * mm_per_pixel_x, endpoints[:, 0] * mm_per_pixel_y, color='red', s=50,
                               label="Endpunkte")
                    ax.legend(loc="upper right")

            if labels is not None:
                for label_info in labels:
                    coord_x, coord_y, label_text = label_info
                    ax.text(coord_x * mm_per_pixel_x, coord_y * mm_per_pixel_y, label_text, color="red", fontsize=10,
                            ha='center', va='center', transform=ax.transData)
            ax.set_title(title)
            ax.set_xlabel(f"X-axis [{unit}]", fontsize=18)
            ax.set_ylabel(f"Y-axis [{unit}]", fontsize=18)
            if x_coating_edge is not None:
                ax.axvline(x=x_coating_edge * mm_per_pixel_x, color='blue', linewidth=2, label="Beschichtungskante")
                ax.legend(loc="upper right")

    # Entferne ungenutzte Achsen
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_results:
        save_folder = r"Bilder Auswertung"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{file_name}_subplots.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Speichern als PNG
    # Close figure to avoid popups
    try:
        plt.close(fig)
    except Exception:
        plt.close()
