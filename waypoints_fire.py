from djitellopy import Tello
from ultralytics import YOLO
import time
import numpy as np
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

print("BATERÍA:", drone.get_battery(), "%")

# ============================================================
# CONFIGURACIÓN CONTROL
# ============================================================
TARGET_CLASS = "Fire"

KP = 0.25
KD = 0.2
DEADZONE = 40
MAX_SPEED = 40
CORRECTION_TIME = 8.0

# ============================================================
# WAYPOINTS
# ============================================================
wpX = [3, 6, 6, 0]
wpY = [0, 1, 4, 0]

positionX = 0
positionY = 0
heading = 0


# ============================================================
# DETECCIÓN DEL CENTRO DE "FIRE"
# ============================================================
def get_fire_centroid(frame):

    results = model(frame, conf=0.3, verbose=False)
    if not results:
        return None

    result = results[0]
    h, w = frame.shape[:2]

    if result.boxes is None:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    best = None
    best_conf = 0

    for box, cls_id, cf in zip(boxes, classes, confs):
        cls_name = names[cls_id]
        if cls_name.lower() == TARGET_CLASS.lower() and cf > best_conf:
            best = box
            best_conf = cf

    if best is None:
        return None

    x1, y1, x2, y2 = best.astype(int)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return cx, cy


# ============================================================
# CENTRAR Y BAJAR-SUBIR
# ============================================================
def center_fire_and_go_down_up():

    print("=== BUSCANDO Y CENTRANDO FIRE ===")

    start = time.time()
    prev_time = start
    prev_error = 0
    last_cmd = 0
    error_hist = []
    centered = False

    while time.time() - start < CORRECTION_TIME:

        frame = drone.get_frame_read().frame
        if frame is None:
            continue

        h, w = frame.shape[:2]
        center_x = w // 2

        centroid = get_fire_centroid(frame)
        now = time.time()
        dt = now - prev_time
        if dt < 0.001:
            dt = 0.001

        if centroid is None:
            drone.send_rc_control(0, 0, 0, 0)
            print("   FIRE no detectado…")
        else:
            cx, cy = centroid
            error = cx - center_x

            error_hist.append(error)
            if len(error_hist) > 4:
                error_hist.pop(0)
            error_f = int(np.mean(error_hist))

            print(f"   FIRE detectado en x={cx}, error={error_f}")

            if abs(error_f) <= DEADZONE:
                print("   FIRE centrado.")
                centered = True
                break

            # Control PD
            p = KP * error_f
            d = KD * (error_f - prev_error) / dt
            cmd = p + d

            if abs(cmd - last_cmd) > 10:
                cmd = last_cmd + np.sign(cmd - last_cmd) * 10

            cmd = int(max(min(cmd, MAX_SPEED), -MAX_SPEED))

            # Tello: left/right es el primer parámetro
            # FIRE a derecha → mover a derecha → cmd positivo
            drone.send_rc_control(cmd, 0, 0, 0)

            prev_error = error_f
            last_cmd = cmd

        prev_time = now
        time.sleep(0.05)

    drone.send_rc_control(0, 0, 0, 0)

    if centered:
        print(">>> FIRE centrado: BAJANDO 20 cm")
        drone.move_down(20)
        time.sleep(5)
        print(">>> SUBIENDO 20 cm")
        drone.move_up(20)
        print(">>> MANIOBRA FIRE COMPLETADA")
    else:
        print("No se centró dentro del tiempo límite.")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
drone.takeoff()
time.sleep(2)

for i in range(len(wpX)):

    dx = wpX[i] - positionX
    dy = wpY[i] - positionY

    target_angle = np.degrees(np.arctan2(dy, dx))
    turn_angle = target_angle - heading
    turn_angle = -((turn_angle + 180) % 360 - 180)
    turn_angle = int(turn_angle)

    distance = int(np.sqrt(dx ** 2 + dy ** 2) * 60)

    print(f"\n=== WAYPOINT {i+1} ===")
    print(f"Target: ({wpX[i]},{wpY[i]})")
    print(f"Giro:   {turn_angle}°")
    print(f"Avance: {distance} cm")

    # Girar
    if abs(turn_angle) > 5:
        if turn_angle > 0:
            drone.rotate_clockwise(turn_angle)
        else:
            drone.rotate_counter_clockwise(-turn_angle)

    # Avanzar
    drone.move_forward(distance)
    print("Waypoint alcanzado")

    # Corrección FIRE
    try:
        center_fire_and_go_down_up()
    except Exception as e:
        print("Error durante corrección:", e)

    positionX = wpX[i]
    positionY = wpY[i]
    heading = target_angle

# Aterrizar
drone.land()
drone.streamoff()
