import time
import math
from collections import deque

import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

# ============================================================
# CONFIGURACIÓN DE WAYPOINTS
# ============================================================
wpX = [2, 5, 6, 0]
wpY = [0, 1, 4, 0]

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
DEADZONE = 60
MAX_SPEED = 18              # velocidad lateral máx

CORRECTION_TIME = 15.0      # ⬅️ más tiempo de corrección por waypoint (s)
INFERENCE_SIZE = 224
FRAME_SKIP = 2              # saltar frames para no saturar CPU

WAYPOINT_HOVER_TIME = 5.0   # ⬅️ tiempo de “espera” en cada waypoint (s)

# Parámetros para decidir cuándo "ya lo centró"
CENTER_TOLERANCE = 25       # píxeles alrededor del centro
DETECTION_STABLE_FRAMES = 5 # nº de frames seguidos centrado

# Altura de la maniobra al encontrar Fire
ALTITUDE_DELTA_CM = 40      # baja 20cm y luego sube 20cm

# Transformación de imagen (igual que tu script que ya funciona)
FLIP_HORIZONTAL = True     # Mirror
FLIP_VERTICAL = False      # flip adicional dentro de fix_image


# ============================================================
# GLOBALES
# ============================================================
tello = None
model = None
frame_read = None


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

    if FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)
    if FLIP_VERTICAL:
        frame = cv2.flip(frame, 0)

    return frame


# ============================================================
# FASE DE CORRECCIÓN: CENTRAR LA CLASE "Fire"
# ============================================================
def correction_phase():
    """
    Lógica:

    1. Siempre muestra la imagen.
    2. Si no encuentra Fire al principio del waypoint:
         → sube 20 cm (solo una vez) y vuelve a buscar.
    3. Si después de subir y durante CORRECTION_TIME no encuentra Fire:
         → sale y pasa al siguiente waypoint.
    4. Si encuentra Fire:
         → corrección lateral (PD) hasta centrar.
         → cuando está centrado varios frames:
               baja 20 cm, espera 3s, sube 20 cm y sale.
    """

    global frame_read, model, tello

    print("  → correction_phase INICIADA")

    prev_error = 0
    prev_time = time.time()
    frame_counter = 0
    stable_center_frames = 0
    error_buffer = deque(maxlen=4)
    last_control = 0
    start_time = time.time()

    already_lifted = False  # para subir 20 cm solo una vez

    while True:
        # Timeout general
        if time.time() - start_time > CORRECTION_TIME:
            tello.send_rc_control(0, 0, 0, 0)
            print("  → Tiempo de corrección agotado, saliendo.")
            return

        frame = frame_read.frame
        if frame is None:
            continue

        # Igual que tu script que funciona:
        frame = cv2.flip(frame, 0)
        img = fix_image(frame)

        h, w = img.shape[:2]
        center_x = w // 2

        # Mostrar SIEMPRE la imagen (aunque no haya Fire)
        display = img.copy()
        cv2.putText(display, "Buscando Fire...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Correction View", display)
        cv2.waitKey(1)

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        current_time = time.time()
        dt = max(current_time - prev_time, 1e-3)
        prev_time = current_time

        # ---------------- YOLO ----------------
        results = model(img, conf=0.2, verbose=False, imgsz=INFERENCE_SIZE)

        found = False
        cx = cy = None
        best_conf = 0.0

        vis = img.copy()

        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            class_names = r.names

            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                # Dibujar TODAS las detecciones (como en tu script simple)
                for i, cls_id in enumerate(cls_ids):
                    class_name = class_names[cls_id]
                    conf = float(confs[i])
                    x1, y1, x2, y2 = xyxy[i].astype(int)

                    # caja general
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(vis, f"{class_name} {conf:.2f}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # buscamos la mejor Fire
                    if class_name.lower() == TARGET_CLASS.lower():
                        if conf > best_conf:
                            best_conf = conf
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            found = True

                # si hay Fire, remarcar en rojo
                if found:
                    cv2.putText(vis, "FIRE!", (cx - 20, cy + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ---------------- LÓGICA SI NO ENCUENTRA ----------------
        if not found:
            # texto de NO FIRE
            cv2.putText(vis, "NO FIRE", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Correction View", vis)
            cv2.waitKey(1)

            tello.send_rc_control(0, 0, 0, 0)

            # subir 20 cm solo la primera vez
            if not already_lifted:
                print("  → No se ve Fire, SUBIENDO 40 cm...")
                already_lifted = True
                time.sleep(0.5)
                tello.move_up(ALTITUDE_DELTA_CM)
                time.sleep(1.0)
                # después de subir, vuelve a intentar detección
                continue

            # si ya subió y sigue sin ver Fire, el while saldrá por timeout
            tello.move_down(ALTITUDE_DELTA_CM)
            continue

        # ---------------- LÓGICA SI ENCUENTRA ----------------
        # control lateral para centrar
        error = cx - center_x
        error_buffer.append(error)
        err_filtered = int(np.mean(error_buffer))

        if abs(err_filtered) < DEADZONE:
            control = int(np.clip(err_filtered * 0.08, -8, 8))
        else:
            p_term = KP * err_filtered
            d_term = KD * (err_filtered - prev_error) / dt
            control = p_term + d_term
            control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))

        prev_error = err_filtered

        tello.send_rc_control(-control, 0, 0, 0)

        # Visualización de centrado
        cv2.line(vis, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(vis, f"err={err_filtered}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"ctrl={control}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"conf={best_conf:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Correction View", vis)
        cv2.waitKey(1)

        # Comprobar centrado
        if abs(err_filtered) < CENTER_TOLERANCE:
            stable_center_frames += 1
        else:
            stable_center_frames = 0

        # Si lleva varios frames centrado → maniobra y salir
        if stable_center_frames >= DETECTION_STABLE_FRAMES:
            print("  ✓ Fire centrado. Ejecutando maniobra bajada/subida...")
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)

            tello.move_down(ALTITUDE_DELTA_CM)
            time.sleep(3.0)
            tello.move_up(ALTITUDE_DELTA_CM)
            time.sleep(0.5)

            print("  ✓ Maniobra completada. Saliendo de correction_phase.")
            tello.send_rc_control(0, 0, 0, 0)
            return


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def main():
    global tello, model, frame_read
    global positionX, positionY, heading

    # ----- Inicializar Tello -----
    tello = Tello()
    tello.connect()
    print("Batería:", tello.get_battery(), "%")

    # ----- Video -----
    tello.streamon()
    time.sleep(2)
    frame_read = tello.get_frame_read()

    # ----- Cargar modelo YOLO -----
    print("Cargando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    print("Modelo YOLO cargado.")

    try:
        # Despegue
        tello.takeoff()
        time.sleep(2)

        # (opcional) subir un poco
        tello.move_up(30)
        time.sleep(2)

        # Recorrer waypoints
        for i in range(len(wpX)):
            dx = wpX[i] - positionX
            dy = wpY[i] - positionY

            # Ángulo global hacia el siguiente waypoint
            target_angle = math.degrees(math.atan2(dy, dx))
            turn_angle = target_angle - heading
            turn_angle = normalize_angle(turn_angle)

            # Distancia (en celdas * 60 cm)
            distance = int(math.sqrt(dx**2 + dy**2) * 60)

            print(f"\n=== Waypoint {i+1}/{len(wpX)} → ({wpX[i]}, {wpY[i]}) ===")
            print(f"  Δx={dx}, Δy={dy}")
            print(f"  Giro relativo: {turn_angle}°")
            print(f"  Distancia:     {distance} cm (aprox)")

            # Giro
            if abs(turn_angle) > 5:
                rotate_tello(tello, turn_angle)
                time.sleep(1)

            # Avance (limitado a rango seguro de Tello)
            distance_clamped = max(20, min(500, distance))
            tello.move_forward(distance_clamped)
            time.sleep(1)
            print("  ✓ Waypoint alcanzado")

            # ----- NUEVO: HOVER CON CÁMARA ACTIVA -----
            hover_start = time.time()
            while time.time() - hover_start < WAYPOINT_HOVER_TIME:
                frame = frame_read.frame
                if frame is None:
                    continue
                frame = cv2.flip(frame, 0)
                img = fix_image(frame)
                cv2.putText(img, f"Waypoint {i+1} hover",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.imshow("Correction View", img)
                cv2.waitKey(1)
                time.sleep(0.03)
            # ------------------------------------------

            # Actualizar pose "ideal"
            positionX = wpX[i]
            positionY = wpY[i]
            heading = target_angle

            # Fase de corrección: centrarse en "Fire" + maniobra / subida si no ve
            correction_phase()

        print("\nRuta completada. Aterrizando...")
        tello.land()

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario (KeyboardInterrupt).")
        try:
            tello.land()
        except Exception:
            pass

    except Exception as e:
        print("\nError en ejecución:", e)
        try:
            tello.land()
        except Exception:
            pass

    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        tello.end()
        print("Recursos liberados.")


if __name__ == "__main__":
    main()
