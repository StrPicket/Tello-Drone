#!/usr/bin/env python3
import cv2
from djitellopy import Tello
import time
import os
from datetime import datetime

class TelloAutoCapture:
    def __init__(self, save_dir='dataset', total_photos=200, interval=1):
        self.tello = Tello()
        self.save_dir = save_dir
        self.total_photos = total_photos
        self.interval = interval
        self.image_count = 0
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f'📁 Directorio creado: {save_dir}/')
    
    def connect(self):
        print('🔌 Conectando al Tello FE18FE...')
        self.tello.connect()
        
        battery = self.tello.get_battery()
        print(f'✅ Conectado - Batería: {battery}%')
        
        if battery < 15:
            print('⚠️  Batería baja!')
            return False
        
        return True
    
    def start_stream(self):
        print('📹 Iniciando stream de video...')
        self.tello.streamon()
        time.sleep(3)
        print('✅ Stream activo')
    
    def auto_capture(self):
        print('\n' + '='*70)
        print('📸 CAPTURA AUTOMÁTICA')
        print('='*70)
        print(f'📊 Total de fotos: {self.total_photos}')
        print(f'⏱️  Intervalo: {self.interval} segundo(s)')
        print(f'📁 Guardando en: {self.save_dir}/')
        print('\nControles:')
        print('  [Q] - Detener captura')
        print('  [P] - Pausar/Reanudar')
        print('='*70 + '\n')
        
        input('Presiona ENTER para comenzar captura...')
        
        frame_read = self.tello.get_frame_read()
        
        paused = False
        last_capture_time = 0
        
        print('\n🚀 Captura iniciada...\n')
        
        while self.image_count < self.total_photos:
            # Obtener frame ORIGINAL
            frame = frame_read.frame
            
            if frame is None:
                print('⚠️  Esperando video...')
                time.sleep(0.1)
                continue
            
            # Convertir a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # CRÍTICO: Crear una COPIA para el display con overlays
            # La imagen original NO tendrá overlays
            frame_display = frame_bgr.copy()
            
            current_time = time.time()
            
            # Info overlay SOLO en frame_display (no en frame_bgr)
            h, w, _ = frame_display.shape
            progress = (self.image_count / self.total_photos) * 100
            
            # Barra de progreso
            bar_width = w - 40
            bar_filled = int((progress / 100) * bar_width)
            cv2.rectangle(frame_display, (20, h-60), (20 + bar_width, h-40), (50, 50, 50), -1)
            cv2.rectangle(frame_display, (20, h-60), (20 + bar_filled, h-40), (0, 255, 0), -1)
            
            # Texto
            status = "PAUSADO" if paused else "CAPTURANDO"
            color = (0, 165, 255) if paused else (0, 255, 0)
            
            cv2.putText(frame_display, f'{status} - {self.image_count}/{self.total_photos} ({progress:.1f}%)', 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            time_until_next = self.interval - (current_time - last_capture_time)
            if not paused and time_until_next > 0:
                cv2.putText(frame_display, f'Siguiente en: {time_until_next:.1f}s', 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame_display, 'Q=Salir | P=Pausar', 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mostrar frame CON overlays
            cv2.imshow('Tello Auto Capture', frame_display)
            
            # Captura automática - Guardar frame ORIGINAL sin overlays
            if not paused and (current_time - last_capture_time >= self.interval):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f'{self.save_dir}/img_{self.image_count:04d}_{timestamp}.jpg'
                
                # ✅ GUARDAR FRAME ORIGINAL SIN OVERLAYS
                cv2.imwrite(filename, frame_bgr)
                self.image_count += 1
                last_capture_time = current_time
                
                print(f'📸 [{self.image_count}/{self.total_photos}] {filename}')
            
            # Capturar tecla
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print('\n⚠️  Captura detenida por el usuario')
                break
            elif key == ord('p'):
                paused = not paused
                status_text = "PAUSADO" if paused else "REANUDADO"
                print(f'\n⏸️  {status_text}')
        
        cv2.destroyAllWindows()
    
    def stop_stream(self):
        print('\n📹 Deteniendo stream...')
        self.tello.streamoff()
    
    def disconnect(self):
        print('🔌 Desconectando...')

if __name__ == '__main__':
    print('='*70)
    print('📸 TELLO AUTO CAPTURE - DATASET LIMPIO')
    print('='*70)
    
    # Configuración
    dataset_name = input('\n📁 Nombre del dataset [datasetDron]: ').strip() or 'datasetDron'
    
    num_photos = input('📸 Número de fotos [200]: ').strip()
    num_photos = int(num_photos) if num_photos else 200
    
    interval = input('⏱️  Intervalo en segundos [1]: ').strip()
    interval = float(interval) if interval else 1
    
    print('\n⚙️  Configuración:')
    print(f'   📁 Carpeta: {dataset_name}/')
    print(f'   📸 Fotos: {num_photos}')
    print(f'   ⏱️  Intervalo: {interval}s')
    
    camera = TelloAutoCapture(
        save_dir=dataset_name,
        total_photos=num_photos,
        interval=interval
    )
    
    try:
        if not camera.connect():
            exit(1)
        
        camera.start_stream()
        camera.auto_capture()
        
    except KeyboardInterrupt:
        print('\n⚠️  Interrupción del usuario')
    
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        camera.stop_stream()
        camera.disconnect()
        
        print('\n' + '='*70)
        print('✅ CAPTURA FINALIZADA')
        print('='*70)
        print(f'📊 Fotos capturadas: {camera.image_count}/{num_photos}')
        print(f'📁 Guardadas en: {dataset_name}/')
        print('='*70)