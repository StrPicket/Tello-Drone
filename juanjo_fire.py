
from djitellopy import Tello
from ultralytics import YOLO
import numpy as np
import time
import cv2

# ================================
# Cargar modelo YOLO11
# ================================
model = YOLO("best.pt")

# ================================
# Conectar Tello
# ================================
drone = Tello()
drone.connect()
drone.streamon()
time.sleep(2)

TARGET_CLASS = "Fire"

# Listas donde se guardan los Waypoints detectados
wpX = []
wpY = []

# Si el segmento anterior ya tuvo fuego
segment_seen_prev = False

# Límite vertical para evitar repetición
HEIGHT_LIMIT_RATIO = 0.1  # 40% desde arriba de la imagen


# ============================================================
# DETECCIÓN Y CÁLCULO DEL CENTRO DE FUEGO
# ============================================================
def get_fire_centroid(frame):
    results = model(frame, conf=0.3, verbose=False)
    if not results:
        return None

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    best_conf = 0
    best_center = None

    for box, cls_id, cf in zip(boxes, classes, confs):
        if names[cls_id].lower() == TARGET_CLASS.lower():
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cf > best_conf:
                best_conf = cf
                best_center = (cx, cy)

    return best_center


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
if __name__ == "__main__":

    i = 0            # Segmento actual
    segment_seen = False  # Si este segmento ya registró fuego

    while i < 6:

        frame = drone.get_frame_read().frame
        if frame is None:
            continue

        # Procesamiento inicial
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)

        # Detección
        fire_centroid = get_fire_centroid(frame)
        fireDetected = fire_centroid is not None

        # Dibujar línea límite
        limit_line = int(frame.shape[0] * HEIGHT_LIMIT_RATIO)
        cv2.line(frame, (0, limit_line), (frame.shape[1], limit_line), (0, 255, 0), 2)

        # Dibujar centro si hay fuego
        if fireDetected:
            cx, cy = fire_centroid
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # ===============================
        # LÓGICA DE REGISTRO
        # ===============================
        if fireDetected and not segment_seen:

            if segment_seen_prev:
                # Validar que esté suficientemente arriba para no ser el mismo fuego
                _, cy = fire_centroid
                if cy < limit_line:
                    wpX.append(i)
                    wpY.append(0)
                    segment_seen = True
                    print(f"Fuego registrado en segmento {i}")
                else:
                    print(f"Fuego ignorado en segmento {i} (misma fuente visual)")
            else:
                # El segmento anterior no tenía fuego → registrar normal
                wpX.append(i)
                wpY.append(0)
                segment_seen = True
                print(f"Fuego registrado en segmento {i}")

        # Mostrar video
        cv2.imshow("CAMARA", frame)

        # ===============================
        # DETECTAR ENTER SIN BLOQUEAR
        # ===============================
        key = cv2.waitKey(1)

        if key == 13:     # ENTER → pasar al siguiente segmento
            i += 1
            segment_seen_prev = segment_seen
            segment_seen = False
            print(f"➡ Segmento cambiado a i = {i}")

        elif key == ord('q'):
            break

    print("\n===== RUTA REGISTRADA =====")
    print("X:", wpX)
    print("Y:", wpY)

    cv2.destroyAllWindows()