import robomaster
import time 
from robomaster import robot

# Inicializar dron
tl_drone = robot.Drone()
tl_drone.initialize()


if __name__ == '__main__':

    tl_flight = tl_drone.flight

    # Set the QUAV to takeoff
    tl_flight.takeoff().wait_for_completed()

    # Add a delay to remain in hover
    print("Remaning in hover")
    time.sleep(5)

    #tl_flight.forward(distance=60).wait_for_completed()

    tl_flight.down(distance=30).wait_for_completed()

    for i in range(2):
        for j in range(7):
            print(j)
            tl_flight.forward(distance=60).wait_for_completed()
            if j % 2 == 1:
                tl_flight.rotate(angle=5).wait_for_completed()
            time.sleep(1)
        tl_flight.rotate(angle=75).wait_for_completed()
        time.sleep(1)
        for k in range(6):
            print(k)
            tl_flight.forward(distance=60).wait_for_completed()
            if k % 2 == 1:
                tl_flight.rotate(angle=5).wait_for_completed()
            time.sleep(1)
        tl_flight.rotate(angle=75).wait_for_completed()
        time.sleep(1)

    # Set the QUAV to land
    tl_flight.land().wait_for_completed()

    # Close resources
    tl_drone.close()
