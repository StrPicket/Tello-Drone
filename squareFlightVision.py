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
# FUNCIÓN DE CORRECCIÓN POST-AVANCE
# ============================================================
def correction_controller_loop(model, target_class="Pipes", timeout_seconds=2.5):
    """
    Corrección continua usando un mini-controlador PD + RC.
    Se llama después de cada forward().
    """

    kp = 0.35
    kd = 0.15
    deadzone = 30

    prev_error = 0
    last_time = time.time()
    start_time = time.time()

    while True:

        # Si se pasó del tiempo máximo → salir
        if time.time() - start_time > timeout_seconds:
            print("Tiempo límite de corrección alcanzado")
            tl_flight.rc(0, 0, 0, 0)
            return

        # Leer frame
        frame = tl_drone.camera.read_cv2_image(strategy="newest")
        if frame is None:
            continue

        h, w = frame.shape[:2]
        center_x = w // 2

        results = model(frame, conf=0.3, verbose=False)

        if len(results) == 0 or results[0].masks is None:
            tl_flight.rc(0, 0, 0, 0)
            return

        result = results[0]
        boxes_classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        class_names = result.names

        valid_idxs = [
            i for i, c in enumerate(boxes_classes)
            if class_names[c].lower() == target_class.lower()
        ]

        if len(valid_idxs) == 0:
            print(f"Tubo '{target_class}' no detectado")
            tl_flight.rc(0, 0, 0, 0)
            return

        # Tomar la detección con mayor confianza
        best_idx = valid_idxs[np.argmax([confidences[i] for i in valid_idxs])]

        mask = result.masks.data[best_idx].cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask_bin = (mask > 0.5).astype("uint8")

        M = cv2.moments(mask_bin)
        if M["m00"] == 0:
            print("Máscara sin centroide válido")
            tl_flight.rc(0, 0, 0, 0)
            return

        cx = int(M["m10"] / M["m00"])
        error = cx - center_x

        print(f"Error lateral: {error:.1f}")

        # Zona muerta
        if abs(error) < deadzone:
            print("Centrado")
            tl_flight.rc(0, 0, 0, 0)  # Stop movement
            return

        # PD ---------------------------------------------------
        now = time.time()
        dt = now - last_time
        if dt < 0.001:
            dt = 0.001

        d_error = (error - prev_error) / dt

        u = kp * error + kd * d_error

        # Saturar velocidad (50 cm/s)
        u = np.clip(u, -50, 50)

        # Izquierda positiva o derecha positiva depende de orientación
        left_right = int(u)

        print(f"rc LR = {left_right}")

        # Mandar comando RC (mientras el loop sigue vivo)
        tl_flight.rc(left_right, 0, 0, 0)

        prev_error = error
        last_time = now

        # Pequeño delay de control (10 Hz)
        time.sleep(0.1)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # === Despegar ===
    tl_flight.takeoff().wait_for_completed()
    time.sleep(3)

    tl_flight.forward(distance=30).wait_for_completed()

    # === Bajar un poco ===
    tl_flight.down(distance=30).wait_for_completed()
    time.sleep(1)

    # ============================================================
    # RECORRIDO 6x7 CON CORRECCIÓN TRAS CADA AVANCE
    # ============================================================
    for i in range(2):

        for j in range(6):
            print(f"\nSegmento {j+1}/6")
            tl_flight.forward(distance=60).wait_for_completed()
            correction_controller_loop(model)

        tl_flight.rotate(angle=-90).wait_for_completed()
        time.sleep(1)

        for k in range(7):
            print(f"\nSegmento {k+1}/7")
            tl_flight.forward(distance=60).wait_for_completed()
            correction_controller_loop(model)

        tl_flight.rotate(angle=-90).wait_for_completed()
        time.sleep(1)

    # === Aterrizar ===
    tl_flight.land().wait_for_completed()

    # === Cerrar recursos ===
    tl_drone.camera.stop_video_stream()
    tl_drone.close()