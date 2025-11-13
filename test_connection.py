# test_dual_connection.py
import socket
import time

def test_tello(name, local_port, tello_ip='192.168.10.1'):
    """Prueba conexión a un Tello en un puerto local específico"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', local_port))
        sock.settimeout(5)
        
        print(f"\n{name} (puerto {local_port}):")
        print("  Enviando 'command'...")
        
        sock.sendto('command'.encode('utf-8'), (tello_ip, 8889))
        response, _ = sock.recvfrom(1024)
        print(f"  ✅ Respuesta: {response.decode('utf-8')}")
        
        # Obtener batería
        sock.sendto('battery?'.encode('utf-8'), (tello_ip, 8889))
        battery, _ = sock.recvfrom(1024)
        print(f"  🔋 Batería: {battery.decode('utf-8')}%")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

if __name__ == '__main__':
    print("=== Test de Conexión Dual Tello ===")
    
    print("\n📡 IMPORTANTE: Asegúrate de que ambos adaptadores estén conectados:")
    print("   - wlo1 → TELLO-XXXXXX")
    print("   - wlx8c902d8e3f0b → TELLO-FE1947")
    
    input("\nPresiona Enter cuando ambos estén conectados...")
    
    # Probar ambos Tello en puertos diferentes
    result1 = test_tello("Tello-1 (wlo1)", 8889)
    time.sleep(1)
    result2 = test_tello("Tello-2 (wlx8c902d8e3f0b)", 8890)
    
    print("\n" + "="*50)
    if result1 and result2:
        print("✅ ¡Ambos Tello conectados exitosamente!")
        print("Ya puedes ejecutar el nodo ROS2")
    else:
        print("⚠️ Hay problemas con la conexión")
        print("Verifica que ambos adaptadores estén conectados")