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

wpX = [2,6,6,3,0,0]
wpY = [0,1,5,6,4,0]

positionX = 0
positionY = 0

# Ángulo inicial del dron (en grados)
heading = 0

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

        # Distancia a recorrer
        distance = int(np.sqrt(dx**2 + dy**2) * 45)


        print(f"Coordenada objetivo: ({wpX[i]}),({wpY[i]})")
        print(f"Ángulo a girar:         {turn_angle}°")
        print(f"Distancia a recorrer:   {distance}")

        if abs(turn_angle) > 5:
            tl_flight.rotate(angle=turn_angle).wait_for_completed()

        tl_flight.forward(distance).wait_for_completed()
        print("Waypoint alcanzado")
        time.sleep(5)
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