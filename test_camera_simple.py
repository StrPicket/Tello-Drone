#!/usr/bin/env python3
"""
Script simple para probar la cámara del Tello
Solo muestra el video en tiempo real
"""

import cv2
from djitellopy import Tello
import time

class TelloCameraTest:
    def __init__(self):
        self.tello = Tello()
    
    def connect(self):
        """Conectar al Tello"""
        print('🔌 Conectando al Tello...')
        try:
            self.tello.connect()
            battery = self.tello.get_battery()
            temp = self.tello.get_temperature()
            
            print(f'✅ Conectado exitosamente')
            print(f'   🔋 Batería: {battery}%')
            print(f'   🌡️  Temperatura: {temp}°C')
            
            if battery < 10:
                print('⚠️  Batería crítica!')
                return False
            
            return True
            
        except Exception as e:
            print(f'❌ Error de conexión: {e}')
            return False
    
    def start_stream(self):
        """Iniciar stream de video"""
        print('\n📹 Iniciando stream...')
        try:
            self.tello.streamon()
            time.sleep(2)  # Esperar a que se estabilice
            print('✅ Stream activo')
            return True
        except Exception as e:
            print(f'❌ Error iniciando stream: {e}')
            return False
    
    def show_video(self):
        """Mostrar video en tiempo real"""
        print('\n' + '='*60)
        print('📺 VISUALIZACIÓN DE CÁMARA TELLO')
        print('='*60)
        print('Controles:')
        print('  [Q] - Salir')
        print('  [S] - Capturar foto (guarda en captures/)')
        print('  [ESC] - Salir')
        print('='*60 + '\n')
        
        input('Presiona ENTER para iniciar...')
        
        frame_read = self.tello.get_frame_read()
        
        capture_count = 0
        
        print('📹 Mostrando video... (Presiona Q para salir)\n')
        
        while True:
            # Obtener frame
            frame = frame_read.frame
            
            if frame is None:
                print('⚠️  Esperando video...')
                time.sleep(0.1)
                continue
            
            # Convertir de RGB a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Agregar info básica
            h, w, _ = frame_bgr.shape
            
            # Obtener info del Tello
            try:
                battery = self.tello.get_battery()
                cv2.putText(frame_bgr, f'Bateria: {battery}%', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except:
                pass
            
            cv2.putText(frame_bgr, 'Q=Salir | S=Capturar', 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostrar frame
            cv2.imshow('Tello Camera Test', frame_bgr)
            
            # Capturar tecla
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q o ESC
                print('\n👋 Cerrando...')
                break
            
            elif key == ord('s'):  # Capturar foto
                import os
                from datetime import datetime
                
                if not os.path.exists('captures'):
                    os.makedirs('captures')
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'captures/tello_{timestamp}.jpg'
                cv2.imwrite(filename, frame_bgr)
                capture_count += 1
                
                print(f'📸 Foto guardada: {filename}')
        
        cv2.destroyAllWindows()
        
        if capture_count > 0:
            print(f'\n📊 Total de fotos capturadas: {capture_count}')
    
    def stop_stream(self):
        """Detener stream"""
        print('\n📹 Deteniendo stream...')
        try:
            self.tello.streamoff()
        except:
            pass
    
    def disconnect(self):
        """Desconectar"""
        print('🔌 Desconectando...')
        try:
            self.tello.end()
        except:
            pass


if __name__ == '__main__':
    print('='*60)
    print('🎥 TEST DE CÁMARA TELLO')
    print('='*60)
    
    camera_test = TelloCameraTest()
    
    try:
        # Conectar
        if not camera_test.connect():
            print('\n❌ No se pudo conectar al Tello')
            exit(1)
        
        # Iniciar stream
        if not camera_test.start_stream():
            print('\n❌ No se pudo iniciar el stream')
            exit(1)
        
        # Mostrar video
        camera_test.show_video()
        
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrumpido por el usuario')
    
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        camera_test.stop_stream()
        camera_test.disconnect()
        
        print('\n' + '='*60)
        print('✅ Test de cámara finalizado')
        print('='*60)
