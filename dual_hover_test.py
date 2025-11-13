#!/usr/bin/env python3
import socket
import time
import threading
import re
import struct

class TelloBindToDevice:
    def __init__(self, name, interface_name, local_port):
        self.name = name
        self.interface_name = interface_name
        self.local_port = local_port
        self.tello_ip = '192.168.10.1'
        self.tello_port = 8889
        self.sock = None
        
    def create_socket(self):
        """Crear socket vinculado FÍSICAMENTE a la interfaz"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # CRÍTICO: Vincular a la interfaz FÍSICA, no solo a la IP
            # Esto requiere SO_BINDTODEVICE
            self.sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BINDTODEVICE,
                self.interface_name.encode()
            )
            
            # Ahora bind al puerto
            self.sock.bind(('', self.local_port))
            self.sock.settimeout(7)
            
            print(f'{self.name} - Socket vinculado a interfaz física: {self.interface_name}')
            return True
            
        except PermissionError:
            print(f'{self.name} - ❌ ERROR: Requiere permisos de root')
            print('   Ejecuta con: sudo python dual_hover_bindtodevice.py')
            return False
        except Exception as e:
            print(f'{self.name} - Error: {e}')
            return False
    
    def send_cmd(self, cmd):
        try:
            self.sock.sendto(cmd.encode('utf-8'), (self.tello_ip, self.tello_port))
            response, _ = self.sock.recvfrom(1518)
            return response.decode('utf-8').strip()
        except socket.timeout:
            return 'timeout'
        except Exception as e:
            return f'error: {e}'
    
    def test_connection(self):
        if not self.create_socket():
            return False, -1, -1
        
        time.sleep(0.3)
        
        resp = self.send_cmd('command')
        if not resp or 'error' in resp or 'timeout' in resp:
            return False, -1, -1
        
        time.sleep(0.5)
        
        resp_bat = self.send_cmd('battery?')
        battery = -1
        
        try:
            battery = int(resp_bat)
        except:
            match = re.search(r'bat:(\d+)', resp_bat)
            if match:
                battery = int(match.group(1))
        
        time.sleep(0.5)
        
        resp_temp = self.send_cmd('temp?')
        temp = -1
        
        try:
            temp = int(resp_temp)
        except:
            match_l = re.search(r'templ:(\d+)', resp_temp)
            match_h = re.search(r'temph:(\d+)', resp_temp)
            if match_l and match_h:
                temp = (int(match_l.group(1)) + int(match_h.group(1))) // 2
        
        return True, battery, temp
    
    def takeoff(self, debug=True):
        resp = self.send_cmd('takeoff')
        if debug:
            print(f'{self.name} - Respuesta takeoff: "{resp}"')
        return 'ok' in resp.lower()
    
    def land(self):
        resp = self.send_cmd('land')
        return 'ok' in resp.lower()

def fly(tello):
    try:
        print(f'\n{tello.name} - 🚀 Despegando...')
        
        if not tello.takeoff(debug=True):
            print(f'{tello.name} - ❌ Error despegue')
            return
        
        print(f'{tello.name} - ✅ En el aire!')
        time.sleep(3)
        
        print(f'{tello.name} - ✈️  Hover 5s...')
        time.sleep(5)
        
        print(f'{tello.name} - 🛬 Aterrizando...')
        tello.land()
        time.sleep(2)
        
        print(f'{tello.name} - ✅ En tierra')
        
    except Exception as e:
        print(f'{tello.name} - ❌ Error: {e}')
        try:
            tello.land()
        except:
            pass

if __name__ == '__main__':
    print('='*70)
    print('🚁🚁 VUELO DUAL TELLO - BIND TO DEVICE 🚁🚁')
    print('='*70)
    
    print('\n📡 Configuración:')
    print('  • wlx8c902d8e3f0b → TELLO-FE18FE (ambos usan 192.168.10.2)')
    print('  • wlx8c902daca34a → TELLO-9A57E0 (ambos usan 192.168.10.2)')
    print('  ⚠️  Usando SO_BINDTODEVICE para diferenciarlos')
    print()
    
    # Crear instancias
    tello1 = TelloBindToDevice('Tello-1 (FE18FE)', 'wlx8c902d8e3f0b', 8889)
    tello2 = TelloBindToDevice('Tello-2 (9A57E0)', 'wlx8c902daca34a', 8890)
    
    print('🔌 Probando conexiones...\n')
    
    ok1, bat1, temp1 = tello1.test_connection()
    
    if ok1:
        print(f'✅ Tello-1 (FE18FE): OK - Batería: {bat1}%')
        if temp1 > 0:
            print(f'   Temp: {temp1}°C')
    else:
        print('❌ Tello-1: Fallo')
        exit(1)
    
    time.sleep(2)
    
    ok2, bat2, temp2 = tello2.test_connection()
    
    if ok2:
        print(f'✅ Tello-2 (9A57E0): OK - Batería: {bat2}%')
        if temp2 > 0:
            print(f'   Temp: {temp2}°C')
    else:
        print('❌ Tello-2: Fallo')
        exit(1)
    
    # Modo
    print('\n' + '='*70)
    print('MODO: 1=Secuencial, 2=Paralelo')
    modo = input('Elige (1/2) [1]: ').strip() or '1'
    
    print('\n' + '='*70)
    print('⚠️  CHECKLIST: Espacio libre, sin personas, buena luz')
    print('='*70)
    
    input('\n🚨 ENTER para iniciar...')
    
    for i in range(5, 0, -1):
        print(f'🚀 {i}...')
        time.sleep(1)
    
    print('\n🚀 INICIANDO\n')
    
    if modo == '1':
        print('📋 SECUENCIAL\n')
        print('--- TELLO-1 ---')
        fly(tello1)
        print('\n⏸️  Pausa 5s...\n')
        time.sleep(5)
        print('--- TELLO-2 ---')
        fly(tello2)
    else:
        print('📋 PARALELO\n')
        t1 = threading.Thread(target=fly, args=(tello1,))
        t2 = threading.Thread(target=fly, args=(tello2,))
        t1.start()
        time.sleep(3)
        t2.start()
        t1.join()
        t2.join()
    
    print('\n' + '='*70)
    print('✅ MISIÓN COMPLETADA')
    print('='*70)