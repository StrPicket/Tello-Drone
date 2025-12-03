import time
import math
from collections import deque
import threading

import cv2
import csv
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN DE WAYPOINTS
# ============================================================

wpX = []
wpY = []

arrayErrorX = []
arrayErrorY = []
referenceX = []
referenceY = []
manipulationX = []
manipulationY = []
measuredX = []
measuredY = []

already_lifted = False

try:
    with open("waypoints.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)   # Saltar encabezado
        for row in reader:
            wpX.append(int(row[0]))
            wpY.append(int(row[1]))

    print("Waypoints leídos correctamente.")
    print("X:", wpX)
    print("Y:", wpY)

except FileNotFoundError:
    print("\nERROR: No se encontró 'waypoints.csv'.")
    print("Asegúrate de que esté en la misma carpeta que este .py\n")
    raise

wpX.append(0)
wpY.append(0)

positionX = 0
positionY = 0
heading = 0   # grados, 0 = eje X positivo del "mundo"


# ============================================================
# CONFIGURACIÓN VISIÓN / CONTROL
# ============================================================
MODEL_PATH = "best.pt"      # tu modelo YOLO
TARGET_CLASS = "Fire"       # clase objetivo

# Control lateral PD
KP = 0.25
KD = 0.2
DEADZONE = 20
MAX_SPEED = 18              # velocidad lateral máx

CORRECTION_TIME = 20.0      # ⬅️ más tiempo de corrección por waypoint (s)
INFERENCE_SIZE = 224
FRAME_SKIP = 2              # saltar frames para no saturar CPU

WAYPOINT_HOVER_TIME = 5.0   # ⬅️ tiempo de "espera" en cada waypoint (s)

# Parámetros para decidir cuándo "ya lo centró"
CENTER_TOLERANCE = 25       # píxeles alrededor del centro
DETECTION_STABLE_FRAMES = 5 # nº de frames seguidos centrado

# Altura de la maniobra al encontrar Fire
ALTITUDE_DELTA_CM = 30     # baja 20cm y luego sube 20cm

# Transformación de imagen (igual que tu script que ya funciona)
FLIP_HORIZONTAL = True     # Mirror
FLIP_VERTICAL = False      # flip adicional dentro de fix_image


# ============================================================
# GLOBALES
# ============================================================
tello = None
model = None
frame_read = None
video_thread = None
running = False
current_status = "Inicializando..."


# ============================================================
# HILO DE VIDEO CONTINUO
# ============================================================
def video_display_thread():
    """
    Hilo que corre continuamente mostrando la cámara
    sin importar qué esté haciendo el dron.
    """
    global frame_read, running, current_status
    
    while running:
        try:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue

            # Aplicar transformaciones
            frame = cv2.flip(frame, 0)
            img = fix_image(frame)

            # Agregar información de estado
            cv2.putText(img, current_status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Tello Camera - Live View", img)
            
            # Permitir cerrar con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.01)  # ~100 FPS máx para no saturar
            
        except Exception as e:
            print(f"Error en video thread: {e}")
            time.sleep(0.1)


# ============================================================
# FUNCIONES UTILITARIAS DE NAVEGACIÓN
# ============================================================
def normalize_angle(angle_deg: float) -> int:
    """Normaliza un ángulo a rango [-180, 180]."""
    return -int((angle_deg + 180) % 360 - 180)


def rotate_tello(t: Tello, angle: int):
    """
    Giro del Tello:
      angle > 0 → clockwise
      angle < 0 → counter_clockwise
    """
    global current_status
    current_status = f"Girando {angle}°..."
    
    if angle > 0:
        t.rotate_clockwise(angle)
    elif angle < 0:
        t.rotate_counter_clockwise(-angle)


def fix_image(frame):
    """
    MISMO pipeline que el code que ya te funciona:
    - frame viene en BGR (djitellopy)
    - Convertimos a RGB
    - Hacemos mirror horizontal y/o vertical
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def plot_control_graphs():
    # ============================================================
    # FIGURA 1 — CONTROL EN X
    # ============================================================
    plt.figure(figsize=(12, 10))

    # --- Referencia vs medida ---
    plt.subplot(3, 1, 1)
    plt.plot(referenceX, label="Referencia X", linewidth=2)
    plt.plot(measuredX, label="Medida X", linewidth=2)
    plt.title("Control en X — Referencia vs Medida")
    plt.xlabel("Muestra")
    plt.ylabel("Pixel")
    plt.legend()
    plt.grid(True)

    # --- Error ---
    plt.subplot(3, 1, 2)
    plt.plot(arrayErrorX, label="Error X", color='red')
    plt.title("Error en X")
    plt.xlabel("Muestra")
    plt.ylabel("Pixel")
    plt.grid(True)

    # --- Manipulación ---
    plt.subplot(3, 1, 3)
    plt.plot(manipulationX, label="Salida de control X", color='green')
    plt.title("Manipulación (u) en X")
    plt.xlabel("Muestra")
    plt.ylabel("Velocidad RC")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ============================================================
    # FIGURA 2 — CONTROL EN YAW
    # ============================================================
    plt.figure(figsize=(12, 10))

    # --- Referencia vs medida ---
    plt.subplot(3, 1, 1)
    plt.plot(referenceY, label="Referencia Yaw", linewidth=2)
    plt.plot(measuredY, label="Medida Yaw", linewidth=2)
    plt.title("Control en YAW — Referencia vs Medida")
    plt.xlabel("Muestra")
    plt.ylabel("Pixel")
    plt.legend()
    plt.grid(True)

    # --- Error ---
    plt.subplot(3, 1, 2)
    plt.plot(arrayErrorY, label="Error Yaw", color='red')
    plt.title("Error en Yaw")
    plt.xlabel("Muestra")
    plt.ylabel("Pixel")
    plt.grid(True)

    # --- Manipulación ---
    plt.subplot(3, 1, 3)
    plt.plot(manipulationY, label="Salida de control Yaw", color='green')
    plt.title("Manipulación (u) en Yaw")
    plt.xlabel("Muestra")
    plt.ylabel("Velocidad RC")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ============================================================
# FASE DE CORRECCIÓN: CENTRAR LA CLASE "Fire"
# ============================================================
def correction_phase():
    """
    Lógica de corrección con detección YOLO en el hilo de video.
    """
    global frame_read, model, tello, current_status, already_lifted

    print("  → correction_phase INICIADA")
    current_status = "Buscando Fire..."

    prev_error = 0
    prev_time = time.time()
    frame_counter = 0
    stable_center_frames = 0
    error_bufferX = deque(maxlen=4)
    error_bufferY = deque(maxlen=4)
    start_time = time.time()

    while True:
        # Timeout general
        if time.time() - start_time > CORRECTION_TIME:
            tello.send_rc_control(0, 0, 0, 0)
            current_status = "Timeout - pasando al siguiente waypoint"
            print("  → Tiempo de corrección agotado, saliendo.")
            return

        frame = frame_read.frame
        if frame is None:
            continue

        frame = cv2.flip(frame, 0)
        img = fix_image(frame)

        h, w = img.shape[:2]
        center_x = w // 2
        center_y = h // 2

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        current_time = time.time()
        dt = max(current_time - prev_time, 1e-3)
        prev_time = current_time

        # ---------------- YOLO ----------------
        results = model(img, conf=0.2, verbose=False)

        found = False
        cx = cy = None
        best_conf = 0.0

        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            class_names = r.names

            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                for i, cls_id in enumerate(cls_ids):
                    class_name = class_names[cls_id]
                    conf = float(confs[i])

                    if class_name.lower() == TARGET_CLASS.lower():
                        if conf > best_conf:
                            best_conf = conf
                            x1, y1, x2, y2 = xyxy[i].astype(int)
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            found = True

        # ---------------- LÓGICA SI NO ENCUENTRA ----------------
        if not found:
            current_status = "NO FIRE detectado"
            tello.send_rc_control(0, 0, 0, 0)

            if not already_lifted:
                print("  → No se ve Fire, SUBIENDO 40 cm...")
                current_status = "Subiendo para buscar Fire..."
                already_lifted = True
                time.sleep(0.5)
                tello.move_up(ALTITUDE_DELTA_CM)
                time.sleep(1.0)
                continue

            continue

        # ---------------- LÓGICA SI ENCUENTRA ----------------
        error = cx - center_x
        errorY = cy - center_y
        referenceX.append(cx)
        referenceY.append(cy)
        measuredX.append(center_x)
        measuredY.append(center_y)
        error_bufferX.append(error)
        error_bufferY.append(errorY)
        err_filtered = int(np.mean(error_bufferX))
        err_filteredY = int(np.mean(error_bufferY))
        arrayErrorX.append(err_filtered)   
        arrayErrorY.append(err_filteredY)     

        if abs(err_filtered) < DEADZONE and abs(err_filteredY) < DEADZONE:
            control = int(np.clip(err_filtered * 0.08, -8, 8))
            controlY = int(np.clip(err_filteredY * 0.08, 8, 8))
        else:
            p_term = KP * err_filtered
            d_term = KD * (err_filtered - prev_error) / dt
            control = p_term + d_term
            control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))
            controlY = int(np.clip(KP*-err_filteredY, -MAX_SPEED, MAX_SPEED))
            manipulationX.append(control)
            manipulationY.append(controlY)

        prev_error = err_filtered

        tello.send_rc_control(control, controlY, 0, 0)
        current_status = f"Fire detectado - Centrando (err={err_filtered}, errY={controlY})"

        # Comprobar centrado
        if abs(err_filtered) < CENTER_TOLERANCE and abs(err_filteredY) < CENTER_TOLERANCE :
            stable_center_frames += 1
        else:
            stable_center_frames = 0

        # Si lleva varios frames centrado → maniobra y salir
        if stable_center_frames >= DETECTION_STABLE_FRAMES:
            break

    print("  ✓ Fire centrado. Ejecutando maniobra bajada/subida...")
    current_status = "Fire centrado - Ejecutando maniobra"
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)

    current_status = "Bajando 40cm..."
    tello.move_down(ALTITUDE_DELTA_CM)
    time.sleep(3.0)
    if already_lifted:
        tello.move_down(ALTITUDE_DELTA_CM)
        time.sleep(3.0)
        already_lifted = False
    
    current_status = "Subiendo 40cm..."
    tello.move_up(ALTITUDE_DELTA_CM)
    time.sleep(0.5)

    print("  ✓ Maniobra completada. Saliendo de correction_phase.")
    tello.send_rc_control(0, 0, 0, 0)
    current_status = "Maniobra completada"
    return


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def main():
    global tello, model, frame_read, video_thread, running, current_status
    global positionX, positionY, heading

    # ----- Inicializar Tello -----
    tello = Tello()
    tello.connect()
    print("Batería:", tello.get_battery(), "%")

    # ----- Video -----
    current_status = "Iniciando video..."
    tello.streamon()
    time.sleep(2)
    frame_read = tello.get_frame_read()

    # ----- Iniciar hilo de video continuo -----
    running = True
    video_thread = threading.Thread(target=video_display_thread, daemon=True)
    video_thread.start()
    print("Hilo de video iniciado")
    time.sleep(1)

    # ----- Cargar modelo YOLO -----
    print("Cargando modelo YOLO...")
    current_status = "Cargando modelo YOLO..."
    model = YOLO(MODEL_PATH)
    print("Modelo YOLO cargado.")

    try:
        # Despegue
        current_status = "Despegando..."
        tello.takeoff()
        time.sleep(3)

        # (opcional) subir un poco
        current_status = "Subiendo a altura inicial..."
        tello.move_up(40)
        time.sleep(3)

        # Calcular el índice del último waypoint (retorno a base)
        total_waypoints = len(wpX)
        last_waypoint_index = total_waypoints - 1

        # Recorrer waypoints
        for i in range(1):
            dx = wpX[i] - positionX
            dy = wpY[i] - positionY

            # Ángulo global hacia el siguiente waypoint
            target_angle = math.degrees(math.atan2(dy, dx))
            turn_angle = target_angle - heading
            turn_angle = normalize_angle(turn_angle)

            # Distancia (en celdas * 60 cm)
            distance = int(math.sqrt(dx**2 + dy**2) * 60)

            print(f"\n=== Waypoint {i+1}/{total_waypoints} → ({wpX[i]}, {wpY[i]}) ===")
            print(f"  Δx={dx}, Δy={dy}")
            print(f"  Giro relativo: {turn_angle}°")
            print(f"  Distancia:     {distance} cm (aprox)")

            # Verificar si es el waypoint de retorno a base
            is_return_to_base = (i == last_waypoint_index)

            # Giro
            if abs(turn_angle) > 5:
                rotate_tello(tello, turn_angle)
                time.sleep(1)

            # Avance (limitado a rango seguro de Tello)
            distance_clamped = max(20, min(500, distance))
            current_status = f"Avanzando {distance_clamped}cm al waypoint {i+1}..."
            tello.move_forward(distance_clamped)
            time.sleep(1)
            print("  ✓ Waypoint alcanzado")

            # ----- HOVER CON CÁMARA ACTIVA -----
            if is_return_to_base:
                current_status = f"Base alcanzada - Preparando aterrizaje"
            else:
                current_status = f"Waypoint {i+1} - Hover ({WAYPOINT_HOVER_TIME}s)"
            
            hover_start = time.time()
            while time.time() - hover_start < WAYPOINT_HOVER_TIME:
                time.sleep(0.1)
            # ------------------------------------------

            # Actualizar pose "ideal"
            positionX = wpX[i]
            positionY = wpY[i]
            heading = target_angle

            # Fase de corrección: SOLO si NO es el último waypoint (retorno a base)
            if not is_return_to_base:
                correction_phase()

        print("\nRuta completada. Aterrizando...")
        current_status = "Aterrizando..."
        tello.land()

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario (KeyboardInterrupt).")
        current_status = "Interrumpido - Aterrizando"
        try:
            tello.land()
        except Exception:
            pass

    except Exception as e:
        print("\nError en ejecución:", e)
        current_status = f"Error: {e}"
        try:
            tello.land()
        except Exception:
            pass

    finally:
        running = False
        time.sleep(0.5)
        tello.streamoff()
        cv2.destroyAllWindows()
        tello.end()
        print("Recursos liberados.")


if __name__ == "__main__":
    main()