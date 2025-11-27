import time
import math
from collections import deque
import csv
import threading
from queue import Queue, Empty

import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

# ============================================================
# CONFIGURACIÓN DE WAYPOINTS DESDE CSV
# ============================================================

wpX = []
wpY = []

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
MODEL_PATH = "best.pt"
TARGET_CLASS = "Fire"

KP = 0.25
KD = 0.2
DEADZONE = 60
MAX_SPEED = 18

CORRECTION_TIME = 15.0
INFERENCE_SIZE = 224

WAYPOINT_HOVER_TIME = 5.0

CENTER_TOLERANCE = 25
DETECTION_STABLE_FRAMES = 5

ALTITUDE_DELTA_CM = 40

FLIP_HORIZONTAL = True
FLIP_VERTICAL = False

# ============================================================
# GLOBALES
# ============================================================
tello = None
model = None
frame_read = None

# Thread-safe variables
latest_frame = None
frame_lock = threading.Lock()
stop_video_thread = False
video_thread = None

# ============================================================
# THREAD DE CAPTURA DE VIDEO - SIEMPRE ACTIVO
# ============================================================
def video_capture_thread():
    """Thread dedicado a capturar frames continuamente sin procesamiento pesado"""
    global latest_frame, stop_video_thread, frame_read
    
    print("[VIDEO] Thread de captura iniciado")
    consecutive_failures = 0
    max_failures = 30
    
    while not stop_video_thread:
        try:
            if frame_read is None:
                time.sleep(0.1)
                continue
                
            frame = frame_read.frame
            
            if frame is not None:
                # Solo flip básico, sin procesamiento pesado
                frame = cv2.flip(frame, 0)
                if FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)
                if FLIP_VERTICAL:
                    frame = cv2.flip(frame, 0)
                
                with frame_lock:
                    latest_frame = frame.copy()
                
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > max_failures:
                    print(f"[VIDEO] ⚠ {consecutive_failures} frames fallidos, intentando reconectar stream...")
                    try:
                        reconnect_stream()
                        consecutive_failures = 0
                    except Exception as e:
                        print(f"[VIDEO] Error al reconectar: {e}")
                
            time.sleep(0.01)  # ~100 FPS max, pero sin procesamiento
            
        except Exception as e:
            print(f"[VIDEO] Error en captura: {e}")
            time.sleep(0.1)
    
    print("[VIDEO] Thread de captura detenido")


def reconnect_stream():
    """Intenta reconectar el stream de video"""
    global frame_read, tello
    
    print("[VIDEO] Reconectando stream...")
    try:
        tello.streamoff()
        time.sleep(1)
        tello.streamon()
        time.sleep(2)
        frame_read = tello.get_frame_read()
        print("[VIDEO] ✓ Stream reconectado")
    except Exception as e:
        print(f"[VIDEO] Error en reconexión: {e}")


def get_latest_frame():
    """Obtiene el último frame de forma thread-safe"""
    with frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return None


def fix_image(frame):
    """Conversión de color simple - los flips ya se hacen en el thread"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ============================================================
# VISUALIZACIÓN LIGERA (sin YOLO)
# ============================================================
def show_simple_frame(text="Navegando...", color=(0, 255, 0)):
    """Muestra frame sin procesamiento pesado"""
    frame = get_latest_frame()
    if frame is None:
        return
    
    display = frame.copy()
    cv2.putText(display, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Tello View", display)
    cv2.waitKey(1)


# ============================================================
# FASE DE CORRECCIÓN: CENTRAR "Fire" (PROCESAMIENTO PESADO)
# ============================================================
def correction_phase():
    global model, tello

    print("  → correction_phase INICIADA")

    prev_error = 0
    prev_time = time.time()
    frame_counter = 0
    stable_center_frames = 0
    error_bufferX = deque(maxlen=4)
    error_bufferY = deque(maxlen=4)
    start_time = time.time()
    already_lifted = False

    while True:
        if time.time() - start_time > CORRECTION_TIME:
            tello.send_rc_control(0, 0, 0, 0)
            print("  → Tiempo de corrección agotado, saliendo.")
            return

        # Obtener frame del thread (siempre disponible)
        frame = get_latest_frame()
        if frame is None:
            print("  ⚠ Sin frame disponible")
            time.sleep(0.1)
            continue

        img = fix_image(frame)
        h, w = img.shape[:2]
        center_x = w // 2
        center_y = h // 2

        frame_counter += 1
        # Reducir frecuencia de YOLO para no saturar
        if frame_counter % 3 != 0:  # Procesar cada 3 frames
            show_simple_frame("Buscando Fire...", (255, 255, 0))
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

                for i, cls_id in enumerate(cls_ids):
                    class_name = class_names[cls_id]
                    conf = float(confs[i])
                    x1, y1, x2, y2 = xyxy[i].astype(int)

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(vis, f"{class_name} {conf:.2f}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if class_name.lower() == TARGET_CLASS.lower():
                        if conf > best_conf:
                            best_conf = conf
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            found = True

                if found:
                    cv2.putText(vis, "FIRE!", (cx - 20, cy + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ---------------- NO ENCUENTRA ----------------
        if not found:
            cv2.putText(vis, "NO FIRE", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Tello View", vis)
            cv2.waitKey(1)

            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.1)

            if not already_lifted:
                print("  → No se ve Fire, SUBIENDO 40 cm...")
                already_lifted = True
                try:
                    tello.move_up(ALTITUDE_DELTA_CM)
                    time.sleep(1.0)
                except Exception as e:
                    print(f"  ⚠ Error al subir: {e}")
                continue
            
            try:
                tello.move_down(ALTITUDE_DELTA_CM)
            except Exception as e:
                print(f"  ⚠ Error al bajar: {e}")
            continue

        # ---------------- SÍ ENCUENTRA ----------------
        error = cx - center_x
        errorY = cy - center_y
        error_bufferX.append(error)
        error_bufferY.append(errorY)
        err_filtered = int(np.mean(error_bufferX))
        err_filteredY = int(np.mean(error_bufferY))        

        if abs(err_filtered) < DEADZONE and abs(err_filteredY) < DEADZONE:
            control = int(np.clip(err_filtered * 0.08, -8, 8))
            controlY = 0
        else:
            p_term = KP * err_filtered
            d_term = KD * (err_filtered - prev_error) / dt
            control = p_term + d_term
            control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))
            controlY = int(np.clip(KP*-err_filteredY, -MAX_SPEED, MAX_SPEED))

        prev_error = err_filtered

        tello.send_rc_control(-control, 0, -controlY, 0)

        cv2.line(vis, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(vis, f"err={err_filtered}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"ctrl={control}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"conf={best_conf:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Tello View", vis)
        cv2.waitKey(1)

        if abs(err_filtered) < CENTER_TOLERANCE:
            stable_center_frames += 1
        else:
            stable_center_frames = 0

        if stable_center_frames >= DETECTION_STABLE_FRAMES:
            print("  ✓ Fire centrado. Ejecutando maniobra bajada/subida...")
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)

            try:
                tello.move_down(ALTITUDE_DELTA_CM)
                time.sleep(3.0)
                tello.move_up(ALTITUDE_DELTA_CM)
                time.sleep(0.5)
            except Exception as e:
                print(f"  ⚠ Error en maniobra: {e}")

            print("  ✓ Maniobra completada. Saliendo de correction_phase.")
            tello.send_rc_control(0, 0, 0, 0)
            return


# ============================================================
# COMANDOS SEGUROS CON REINTENTOS
# ============================================================
def safe_command(command_func, *args, max_retries=3, **kwargs):
    """Ejecuta comando con reintentos y manejo de errores"""
    for attempt in range(max_retries):
        try:
            command_func(*args, **kwargs)
            return True
        except Exception as e:
            print(f"  ⚠ Intento {attempt+1}/{max_retries} falló: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  ✗ Comando falló después de {max_retries} intentos")
                return False


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def main():
    global tello, model, frame_read, stop_video_thread, video_thread
    global positionX, positionY, heading

    tello = Tello()
    tello.connect()
    print("Batería:", tello.get_battery(), "%")

    # Iniciar stream
    tello.streamon()
    time.sleep(2)
    frame_read = tello.get_frame_read()
    time.sleep(1)

    # Iniciar thread de video ANTES de cualquier otra cosa
    stop_video_thread = False
    video_thread = threading.Thread(target=video_capture_thread, daemon=True)
    video_thread.start()
    time.sleep(1)  # Dar tiempo a que el thread capture frames

    print("Cargando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    print("Modelo YOLO cargado.")

    try:
        safe_command(tello.takeoff)
        time.sleep(2)

        safe_command(tello.move_up, 30)
        time.sleep(2)

        for i in range(len(wpX)):
            dx = wpX[i] - positionX
            dy = wpY[i] - positionY

            target_angle = math.degrees(math.atan2(dy, dx))
            turn_angle = int((target_angle - heading + 180) % 360 - 180)
            turn_angle = -turn_angle

            distance = int(math.sqrt(dx**2 + dy**2) * 60)

            print(f"\n=== Waypoint {i+1}/{len(wpX)} → ({wpX[i]}, {wpY[i]}) ===")
            print(f"  Δx={dx}, Δy={dy}")
            print(f"  Giro relativo: {turn_angle}°")
            print(f"  Distancia:     {distance} cm (aprox)")

            # Rotación segura
            if abs(turn_angle) > 5:
                if turn_angle > 0:
                    safe_command(tello.rotate_clockwise, abs(turn_angle))
                else:
                    safe_command(tello.rotate_counter_clockwise, abs(turn_angle))
                time.sleep(1)

            # Movimiento seguro
            distance_clamped = max(20, min(500, distance))
            safe_command(tello.move_forward, distance_clamped)
            time.sleep(1)
            print("  ✓ Waypoint alcanzado")

            # Hover con visualización ligera (sin YOLO)
            hover_start = time.time()
            while time.time() - hover_start < WAYPOINT_HOVER_TIME:
                show_simple_frame(f"Waypoint {i+1} hover", (255, 255, 0))
                time.sleep(0.03)

            positionX = wpX[i]
            positionY = wpY[i]
            heading = target_angle

            # AHORA SÍ procesamiento pesado
            correction_phase()

        print("\nRuta completada. Aterrizando...")
        safe_command(tello.land)

    except KeyboardInterrupt:
        print("\n⚠ Interrumpido por el usuario.")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        safe_command(tello.land)

    except Exception as e:
        print(f"\n✗ Error en ejecución: {e}")
        import traceback
        traceback.print_exc()
        tello.send_rc_control(0, 0, 0, 0)
        safe_command(tello.land)

    finally:
        # Detener thread de video
        stop_video_thread = True
        if video_thread:
            video_thread.join(timeout=2)
        
        try:
            tello.streamoff()
        except:
            pass
        
        cv2.destroyAllWindows()
        tello.end()
        print("Recursos liberados.")


if __name__ == "__main__":
    main()