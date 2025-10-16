import customtkinter as ctk

global measurement_running
measurement_running = [0]

global status_list
status_list = []

global image_mode
image_mode = "2D_full" #options: "2D_reduced", "2D_full", "3D_reduced"

global current_selected_test_directory
current_selected_test_directory = None

global temp_selected_test_directory
temp_selected_test_directory = None

global current_sensor_0_file_path
current_sensor_0_file_path = None

global current_sensor_1_file_path
current_sensor_1_file_path = None

global current_sensor_2_file_path
current_sensor_2_file_path = None

global initialization_complete
initialization_complete = False

global Trigger_S1, Trigger_S2, Trigger_S3
Trigger_S1, Trigger_S2, Trigger_S3 = None, None, None

global query_api
query_api = None

global default_date
default_date = ""

global default_batch_len
default_batch_len = 10

global output_folder
output_folder = r"C=\Users\FKoeadmin\Desktop"

global lower_y_value
lower_y_value = 0

global upper_y_value
upper_y_value = 10000

global nodes
nodes = {"LINE_ON" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Line_On_n_0"',
    "STOERUNG_AKTIV" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Stoerung_Aktiv"',
    "SAFETY_MELDUNG_AKTIV" :'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Safety_Meldung_Aktiv"',
    "AKT_SPS_ZEIT" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Aktuelle_SPS_Zeit"',
    "GESCHWINDIGKEIT_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Linespeed"."SPEED_Act"',
    "GRUNDGESCHWINDIGKEIT" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Linespeed"."v_Line_Base"',
    "PRODUKTIONSGESCHWINDIGKEIT" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Linespeed"."v_Line_Fabrication"',
    "SYSTEMDRUCK_OK" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Hydraulik"."Sytemdruck_OK"',
    "SYSTEMDRUCK_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Hydraulik"."IST_Sytemdruck"',
    "HKG_AN" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."HKG"."On"',
    "HKG_WALZENFREIGABE" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."HKG"."FRG_Walzentemperatur"',
    "HKG_HEIZT" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."HKG"."Heizen_Aktiv"',
    "HKG_KUEHLT" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."HKG"."Kuehlen_Aktiv"',
    "TEMP_SOLL" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."HKG"."Temperature_Set"',
    "TEMP_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."HKG"."Temperature_Act"',
    "ABW_BAHNSPANNUNG_SOLL" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Abwickler"."Bahnspannung_SOLL"',
    "STARTPOS_RB_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."STARTPOSITION_RB_AS"',
    "SOLLKRAFT_RB_AS_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Sollkraft_RB_AS_BS"',
    "PRODUKTBREITE" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Produktbreite"',
    "LINIENLAST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Linienlast"',
    "SOLLPOS_HZYL_AS_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."SOLLPOS_HAUPTZYL_AS_BS"',
    "MOTOR_OBEN_DREHMOMENT_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Motor_Oben_Torque_Act"',
    "MOTOR_UNTEN_DREHMOMENT_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Motor_Unten_Torque_Act"',
    "MAN_OFFSET_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."AS_MAN_OFFSET_MAN"',
    "MAN_OFFSET_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."BS_MAN_OFFSET_MAN"',
    "IST_POS_HP_ZYL_EINL_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_POS_HP_ZYL_EINL_AS"',
    "IST_POS_HP_ZYL_AUSL_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_POS_HP_ZYL_AUSL_AS"',
    "IST_POS_HP_ZYL_EINL_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_POS_HP_ZYL_EINL_BS"',
    "IST_POS_HP_ZYL_AUSL_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_POS_HP_ZYL_AUSL_BS"',
    "IST_KRAFT_HP_ZYL_EINL_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_KRAFT_HP_ZYL_EINL_AS"',
    "IST_KRAFT_HP_ZYL_AUSL_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_KRAFT_HP_ZYL_AUSL_AS"',
    "IST_KRAFT_HP_ZYL_EINL_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_KRAFT_HP_ZYL_EINL_BS"',
    "IST_KRAFT_HP_ZYL_AUSL_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Pauly"."IST_KRAFT_HP_ZYL_AUSL_BS"',
    "SCHIEF_OFFSET_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."OFFSET_AS"',
    "SCHIEF_OFFSET_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."OFFSET_BS"',
    "SCHIEF_TEMP_OFFSET" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."TEMP_OFFSET_WALZEN"',
    "SCHIEF_DURCHMESSER_OFFSET" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."Walzedurchmesser_OFFSET"',
    "SCHIEF_GES_OFFSET_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."Gesamt_OFFSET_AS"',
    "SCHIEF_GES_OFFSET_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."Gesamt_OFFSET_BS"',
    "SCHIEF_SOLLPOS_HZYL_AS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."Sollpos_Hauptzylinder_AS"',
    "SCHIEF_SOLLPOS_HZYL_BS" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."Schiefstellung"."Sollpos_Hauptzylinder_BS"',
    "D_WALZE_OBEN" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."D_Walze_oben"',
    "D_WALZE_UNTEN" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."D_Walze_unten"',
    "OFFSET_WALZENDURCHMESSER" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Kalander"."OFFSET_Walzedurchmesser"',
    "ZUGWERK_ZUGKRAFT_SOLL" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Zugwerk"."Tension_Set"',
    "ZUGWERK_ZUGKRAFT_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Zugwerk"."Tension_Act"',
    "ZUGWERK_DREHMOMENT_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Zugwerk"."Drehmoment_IST"',
    "AUFWICKLER_DREHMOMENT_IST" : 'ns=2;s=CPU 315F-2 PN/DP."OPC-UA_STATUS"."Aufwickler"."Drehmoment_IST"'}

global single_values_opcua
single_values_opcua = {}

global exp_line_load, exp_temperature, exp_web_speed, exp_web_tension, exp_unwinder_tension, exp_rewinder_tension
exp_line_load = 1
exp_temperature = 2
exp_web_speed = 3
exp_web_tension = 4
exp_unwinder_tension = 5
exp_rewinder_tension = 6