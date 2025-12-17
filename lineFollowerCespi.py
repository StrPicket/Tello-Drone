import robomaster
from robomaster import robot
import time
import cv2
from cv2 import aruco
import numpy as np

ROI_HEIGHT_RATIO = 0.4
ROI_WIDTH_RATIO = 0.5

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
parameters = aruco.DetectorParameters()

# Control PD
Kp_yaw = 1.12
Kd_yaw = 0.18

ARUCO_CENTER_THRESHOLD = 30  # píxeles aceptados para estar centrado

STATE = "line_follow"  # otros: aruco_centering , aruco_action
current_action = None

def process_line(frame, center_x_ref):
    h, w, _ = frame.shape
    y0 = int(h * 0.05)
    y1 = int(h * ROI_HEIGHT_RATIO)

    roi_w_half = int(w * ROI_WIDTH_RATIO / 2)
    x_center = w // 2
    x1 = x_center - roi_w_half
    x2 = x_center + roi_w_half
    roi = frame[:y1, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 120, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    error = None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            error = float((roi.shape[1] / 2) - cx)

    cv2.imshow("ROI Azul", mask)
    cv2.waitKey(1)

    return error

def process_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    cx, cy = None, None
    found_id = None

    if ids is not None and len(ids) > 0:
        # corners es lista: one entry per marker; cada entry puede ser (1,4,2) o (4,2) según versión
        # Normalizamos la forma para trabajar cómodamente
        for m_idx in range(len(corners)):
            pts = np.squeeze(corners[m_idx])  # queda (4,2)
            if pts.shape != (4,2):
                # segurar, si algo raro continúe
                continue

            # dibuja las 4 esquinas
            for i, p in enumerate(pts):
                px, py = int(p[0]), int(p[1])
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                # opcional: etiqueta la esquina
                cv2.putText(frame, str(i), (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            # calcula y dibuja el centro del marcador
            cx_m = int(np.mean(pts[:, 0]))
            cy_m = int(np.mean(pts[:, 1]))
            cv2.circle(frame, (cx_m, cy_m), 6, (0, 255, 0), -1)  # centro en verde
            # dibuja contorno y id
            aruco.drawDetectedMarkers(frame, [corners[m_idx]], ids[m_idx:m_idx+1])
            cv2.putText(frame, f"ID:{int(ids[m_idx][0])}", (cx_m+8, cy_m+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # si aún no hemos seleccionado un marcador para retornar, toma el primero válido
            if found_id is None:
                found_id = int(ids[m_idx][0])
                cx, cy = cx_m, cy_m
                # si quieres devolver las esquinas del primer marcador como el resto del código, 
                # devolvemos corners[m_idx] abajo
                chosen_corners = corners[m_idx]

        # muestra el frame con marcadores
        cv2.imshow("Marker Detection", frame)
        cv2.waitKey(1)

        return found_id, cx, cy, chosen_corners

    # si no hay ids
    cv2.imshow("Marker Detection", frame)
    cv2.waitKey(1)
    return None, None, None, None


if __name__ == "__main__":
    tl_drone = robot.Drone()
    tl_drone.initialize()

    tl_flight = tl_drone.flight
    tl_camera = tl_drone.camera
    tl_camera.start_video_stream(display=False)

    tl_flight.takeoff().wait_for_completed()
    tl_flight.up(15).wait_for_completed

    prev_error = 0

    try:
        while True:
            frame = tl_camera.read_cv2_image(strategy="newest", timeout=5)
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 0)

            h, w, _ = frame.shape
            center_x_ref = w / 2
            center_y_ref = h / 2 

            # Procesos visuales
            line_error = process_line(frame, center_x_ref)
            aruco_id, ax, ay, cornersAruco = process_aruco(frame)

            # === ESTADO: Seguir línea ===
            if STATE == "line_follow":
                if line_error is not None:

                    derivative = line_error - prev_error
                    yaw = int(Kp_yaw * line_error + Kd_yaw * derivative)
                    yaw = np.clip(yaw, -75, 75)

                    print(f"Error yaw: {line_error} , Yaw cmd: {yaw}")

                    if abs(line_error) < 20:
                        forward = 25
                    else:
                        forward = 0

                    tl_flight.rc(0, forward, 0, -yaw)
                    prev_error = line_error

                else:
                    if aruco_id is not None:
                        STATE = "aruco_centering"
                        current_action = None
                        print("Cambiando a estado: CENTRADO ARUCO")
                    else:
                        tl_flight.rc(0, 0, 10, -60)


            # === ESTADO: Centrar ArUco ===
            elif STATE == "aruco_centering":
                if aruco_id is None:
                    aruco_id, ax, ay, cornersAruco = process_aruco(frame)
                    continue

                # Errores horizontal y vertical
                error_x = center_x_ref - ax
                error_y = center_y_ref - ay  # centro vertical

                # Control
                # === Calcular error de orientación (yaw) con esquinas ===
                pts = cornersAruco[0][0]
                pts = np.squeeze(cornersAruco)  # queda (4,2)
                pt0 = pts[0]  # esquina inferior izquierda
                pt1 = pts[1]  # esquina inferior derecha

                yaw_error = pt0[1] - pt1[1]
                print(f"Error yaw: {yaw_error}")

                # Control de orientación (positivo o negativo)
                yaw_cmd = np.clip(Kp_yaw * yaw_error, -60, 60)

                # Control de posición
                yaw_pos_cmd = np.clip(0.3 * error_x, -10, 10)
                updown_cmd   = np.clip(1.2 * (-error_y), -10, 10)

                # Movimiento combinado
                tl_flight.rc(
                    int(-yaw_pos_cmd),    # corrección horizontal
                    int(-updown_cmd),     # corrección vertical
                    0,
                    int(yaw_cmd)          # corrección de orientación
                )

                # Checar si ya está centrado en ambos ejes
                if (
                    abs(error_x) < ARUCO_CENTER_THRESHOLD and
                    abs(error_y) < ARUCO_CENTER_THRESHOLD and
                    abs(yaw_error) < 25    # umbral angular
                    ):
                    tl_flight.rc(0, 0, 0, 0)
                    time.sleep(1)
                    
                    STATE = "aruco_action"
                    current_action = aruco_id
                    print(f"Objetivo centrado — Ejecutando acción ID {aruco_id}")

            # === ESTADO: Ejecutar acción continua según ID ===
            elif STATE == "aruco_action":

                # Si aparece una nueva línea → volver
                if line_error is not None:
                    STATE = "line_follow"
                    current_action = None
                    continue

                # Si aparece otro ArUco → cambiar a ese
                if aruco_id is not None and aruco_id != current_action:
                    print("Nuevo ArUco detectado, cambiando acción...")
                    STATE = "aruco_centering"
                    current_action = None
                    continue

                # Si detecta el MISMO ArUco → ignorarlo y seguir la acción

                if current_action == 1:
                    tl_flight.rc(0, 20, 0, 0)  # Adelante
                    time.sleep(1)
                elif current_action == 2:
                    tl_flight.rc(-20, 0, 0, 0)  # Izquierda
                elif current_action == 3:
                    tl_flight.rc(20, 0, 0, 0)  # Derecha
                    time.sleep(1)
                elif current_action == 4:
                    tl_flight.rc(0, 0, 0, 0)
                    print("Fin detectado — Aterrizando")
                    break

    except KeyboardInterrupt:
        print("Interrupción del usuario.")

    finally:
        tl_flight.land().wait_for_completed()
        tl_camera.stop_video_stream()
        cv2.destroyAllWindows()
        tl_drone.close()
        print("Drone cerrado.")
