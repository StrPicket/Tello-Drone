import cv2
import time
import numpy as np
from robomaster import robot
from ultralytics import YOLO

# === Inicializar dron ===
tl_drone = robot.Drone()
tl_drone.initialize()
tl_flight = tl_drone.flight

# === Inicializar cámara ===
tl_camera = tl_drone.camera
tl_camera.start_video_stream(display=False)

# === Cargar modelo YOLO ===
model = YOLO("best.pt")

# ============================================================
# CONFIGURACIÓN DE CORRECCIÓN (adaptado de Drone1_routine.py)
# ============================================================
TARGET_CLASS = "FIRE"  # ← nombre EXACTO de la clase en tu modelo

KP = 0.25
KD = 0.2
DEADZONE = 40          # margen en píxeles para considerar centrado
MAX_SPEED = 30         # velocidad máx. en rc()
CORRECTION_TIME = 8.0  # tiempo máx. intentando centrar FIRE (s)

# ============================================================
# WAYPOINTS fire 3,0 ; 6,1; 6,4; 6,7; 0,7
# ============================================================
wpX = [3, 6, 6, 0]
wpY = [0, 1, 4, 0]

positionX = 0
positionY = 0

# Ángulo inicial del dron (en grados)
heading = 0


# ============================================================
# FUNCIONES DE VISIÓN Y CORRECCIÓN
# ============================================================
def get_fire_centroid(frame_bgr):
    """
    Ejecuta YOLO sobre el frame y devuelve el centro (cx, cy) de la mejor
    detección de la clase TARGET_CLASS. Si el modelo tiene máscaras,
    se usa el centroide de la máscara; si no, el centro del bounding box.
    """          
    results = model(frame_bgr, conf=0.3, verbose=False)
    if not results:
        return None

    result = results[0]
    h, w = frame_bgr.shape[:2]

    if result.boxes is None:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    class_names = result.names

    best_idx = None
    best_conf = 0.0

    # Buscar la detección FIRE con mayor confianza
    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
        class_name = class_names[cls_id]
        if class_name.lower() == TARGET_CLASS.lower() and conf > best_conf:
            best_idx = i
            best_conf = conf

    if best_idx is None:
        return None

    x1, y1, x2, y2 = boxes[best_idx].astype(int)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Si hay máscaras, refinar con el centroide de la máscara
    if result.masks is not None and len(result.masks.data) > best_idx:
        mask = result.masks.data[best_idx].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_resized > 0.5).astype(np.uint8)
        M = cv2.moments(mask_bin)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

    return cx, cy


def center_fire_and_go_down_up():
    """
    1. Busca FIRE en la imagen.
    2. Usa control PD sobre rc() para centrar FIRE horizontalmente.
    3. Cuando está centrado, baja 20 cm, espera 5 s, y sube 20 cm.
    4. Si no logra centrar FIRE en CORRECTION_TIME, sale sin mover altura.
    """
    print("=== BUSCANDO Y CENTRANDO FIRE ===")

    start_time = time.time()
    prev_time = start_time
    prev_error = 0
    last_control = 0
    error_history = []
    centered = False
    last_centroid = None

    while time.time() - start_time < CORRECTION_TIME:
        # Leer último frame de la cámara
        # La mayoría de ejemplos del SDK usan strategy="newest"
        frame = tl_camera.read_cv2_image(strategy="newest")
        if frame is None:
            time.sleep(0.02)
            continue

        # RoboMaster suele devolver RGB; convertimos a BGR para OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        center_x = w // 2

        centroid = get_fire_centroid(frame_bgr)
        current_time = time.time()
        dt = current_time - prev_time
        if dt < 0.001:
            dt = 0.001

        if centroid is None:
            # No se ve FIRE, parar movimiento lateral
            tl_flight.rc(a=0, b=0, c=0, d=0)
            error_history.clear()
            print("   FIRE no detectado...")
        else:
            cx, cy = centroid
            last_centroid = centroid
            error = cx - center_x   # error horizontal (px)

            error_history.append(error)
            if len(error_history) > 4:
                error_history.pop(0)
            error_filtered = int(np.mean(error_history))

            print(f"   FIRE detectado en x={cx}, error={error_filtered} px")

            # ¿Ya está suficientemente centrado?
            if abs(error_filtered) <= DEADZONE:
                tl_flight.rc(a=0, b=0, c=0, d=0)
                centered = True
                print("   FIRE centrado.")
                break

            # Control PD
            p_term = KP * error_filtered
            d_term = KD * (error_filtered - prev_error) / dt
            control = p_term + d_term

            # Limitar cambios bruscos respecto al último control
            if abs(control - last_control) > 10:
                control = last_control + np.sign(control - last_control) * 10

            control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))

            # En RoboMaster, rc(a=20) mueve a la IZQUIERDA.
            # Queremos que si FIRE está a la derecha (error>0) el dron
            # se mueva a la derecha, por eso usamos el signo invertido.
            tl_flight.rc(a=-control, b=0, c=0, d=0)

            prev_error = error_filtered
            last_control = control

        prev_time = current_time
        time.sleep(0.05)

    # Parar siempre el rc al salir
    tl_flight.rc(a=0, b=0, c=0, d=0)

    # Si se centró FIRE, ejecutar la maniobra de bajar-guardar-subir
    if centered and last_centroid is not None:
        print(">>> FIRE centrado: bajando 20 cm...")
        tl_flight.down(20).wait_for_completed()

        print(">>> Manteniendo posición 5 segundos...")
        time.sleep(5)

        print(">>> Subiendo 20 cm...")
        tl_flight.up(20).wait_for_completed()
        print(">>> Maniobra FIRE completada.")
    else:
        print("No se logró centrar FIRE dentro del tiempo límite; se continúa sin bajar/subir.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    tl_flight.takeoff().wait_for_completed()
    time.sleep(2)

    for i in range(len(wpX)):

        dx = wpX[i] - positionX
        dy = wpY[i] - positionY

        # Ángulo objetivo respecto al mundo
        target_angle = np.rad2deg(np.arctan2(dy, dx))

        # Angulo respecto al heading actual
        turn_angle = target_angle - heading

        # Normalizar a -180..180
        turn_angle = -((turn_angle + 180) % 360 - 180)

        turn_angle = int(turn_angle)

        # Distancia a recorrer (cada unidad de grid → 45 cm)
        distance = int(np.sqrt(dx ** 2 + dy ** 2) * 45)

        print(f"Coordenada objetivo: ({wpX[i]}),({wpY[i]})")
        print(f"Ángulo a girar:         {turn_angle}°")
        print(f"Distancia a recorrer:   {distance} cm")

        if abs(turn_angle) > 5:
            tl_flight.rotate(angle=turn_angle).wait_for_completed()

        tl_flight.forward(distance).wait_for_completed()
        print("Waypoint alcanzado")

        # === NUEVO: fase de corrección para FIRE en cada waypoint ===
        try:
            center_fire_and_go_down_up()
        except Exception as e:
            print(f"Error en fase FIRE: {e}")

        print("-----")

        # Actualizar posición y orientación
        positionX = wpX[i]
        positionY = wpY[i]
        heading = target_angle

    # === Aterrizar ===
    tl_flight.land().wait_for_completed()

    # === Cerrar recursos ===
    tl_drone.camera.stop_video_stream()
    tl_drone.close()
