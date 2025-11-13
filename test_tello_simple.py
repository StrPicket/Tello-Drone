# safe_flight.py
from djitellopy import Tello
import time

def safe_takeoff():
    tello = Tello()
    tello.connect()
    
    # Verificaciones de seguridad
    battery = tello.get_battery()
    temp = tello.get_temperature()
    
    print(f"Batería: {battery}%")
    print(f"Temperatura: {temp}°C")
    
    # Verificar temperatura
    if temp > 65:
        print("❌ TEMPERATURA MUY ALTA. Deja que se enfríe.")
        return False
    
    # Verificar batería
    if battery < 20:
        print("❌ Batería muy baja.")
        return False
    
    print("✅ Condiciones OK para volar\n")
    
    # Esperar antes de despegar
    print("Esperando 5 segundos...")
    time.sleep(5)
    
    try:
        print("🚁 Despegando...")
        tello.takeoff()
        
        print("✅ En el aire! Mantiendo posición...")
        time.sleep(3)
        
        print("🛬 Aterrizando...")
        tello.land()
        
        print("✅ Vuelo completado!")
        return True
        
    except Exception as e:
        print(f"❌ Error durante el vuelo: {e}")
        try:
            tello.land()
        except:
            pass
        return False

if __name__ == '__main__':
    safe_takeoff()