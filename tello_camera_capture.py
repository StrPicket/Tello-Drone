#!/usr/bin/env python3
import cv2
from djitellopy import Tello
import time
import os
from datetime import datetime

class TelloAutoCapture:
    def __init__(self, save_dir='dataset', total_photos=180, interval=1, cooldown_every=30, cooldown_time=10):
        self.tello = Tello()
        self.save_dir = save_dir
        self.total_photos = total_photos
        self.interval = interval
        self.cooldown_every = cooldown_every
        self.cooldown_time = cooldown_time
        self.image_count = 0
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f'📁 Directorio creado: {save_dir}/')
    
    def connect(self):
        print('🔌 Conectando al Tello...')
        self.tello.connect()
        
        battery = self.tello.get_battery()
        print(f'✅ Conectado - Batería: {battery}%')
        
        if battery < 15:
            print('⚠️  Batería baja!')
            return False
        
        return True
    
    def start_stream(self):
        print('📹 Iniciando stream...')
        self.tello.streamon()
        time.sleep(2)
        print('✅ Stream activo')
    
    def auto_capture(self):
        print('\n' + '='*70)
        print('📸 CAPTURA AUTOMÁTICA CON PAUSAS')
        print('='*70)
        print(f'📊 Total: 100 fotos')
        print(f'⏱️  Intervalo: 1 segundo')
        print(f'⏸️  Pausa cada 20 fotos por 3 segundos')
        print(f'📁 Guardando en: {self.save_dir}/')
        print('\nControles: [Q]=Salir | [P]=Pausar')
        print('='*70 + '\n')
        
        input('Presiona ENTER para comenzar...')
        
        frame_read = self.tello.get_frame_read()
        
        paused = False
        last_capture_time = time.time()
        last_cooldown_at = 0  # ✅ Rastrear cuándo fue el último cooldown
        
        print('\n🚀 Captura iniciada...\n')
        
        while self.image_count < self.total_photos:
            # Frame
            frame = frame_read.frame
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Convertir
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_display = frame_bgr.copy()
            
            current_time = time.time()
            
            # ✅ Verificar cooldown (solo si NO acabamos de hacer uno)
            if (self.image_count > 0 and 
                self.image_count % self.cooldown_every == 0 and 
                self.image_count != last_cooldown_at):  # CRÍTICO: evitar loop
                
                last_cooldown_at = self.image_count  # Marcar que hicimos cooldown
                
                print(f'\n⏸️  COOLDOWN ({self.cooldown_time}s) - Cambia de posición')
                print(f'   Progreso: {self.image_count}/{self.total_photos}\n')
                
                # Countdown visual
                for remaining in range(self.cooldown_time, 0, -1):
                    frame = frame_read.frame
                    if frame is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_display = frame_bgr.copy()
                        
                        h, w, _ = frame_display.shape
                        
                        # Countdown grande
                        text = f'COOLDOWN: {remaining}s'
                        cv2.putText(frame_display, text, 
                                   (w//2 - 150, h//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                        
                        cv2.putText(frame_display, 'Cambia de posicion', 
                                   (w//2 - 120, h//2 + 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        cv2.imshow('Tello Auto Capture', frame_display)
                        cv2.waitKey(1)
                    
                    time.sleep(1)
                
                print(f'▶️  Continuando...\n')
                last_capture_time = time.time()  # Reset timer
                continue
            
            # Overlay simplificado
            h, w, _ = frame_display.shape
            progress = (self.image_count / self.total_photos) * 100
            
            # Estado
            if paused:
                status = "PAUSADO"
                color = (0, 165, 255)
            else:
                status = "CAPTURANDO"
                color = (0, 255, 0)
            
            cv2.putText(frame_display, f'{status} - {self.image_count}/100 ({progress:.0f}%)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Próxima pausa
            photos_until_cooldown = self.cooldown_every - (self.image_count % self.cooldown_every)
            if photos_until_cooldown > 0 and self.image_count < self.total_photos:
                cv2.putText(frame_display, f'Pausa en: {photos_until_cooldown} fotos', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Tello Auto Capture', frame_display)
            
            # Captura
            if not paused and (current_time - last_capture_time >= self.interval):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{self.save_dir}/img_{self.image_count:04d}_{timestamp}.jpg'
                
                cv2.imwrite(filename, frame_bgr)
                self.image_count += 1
                last_capture_time = current_time
                
                print(f'📸 [{self.image_count:3d}/100] {filename}')
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print('\n⚠️  Detenido por el usuario')
                break
            elif key == ord('p'):
                paused = not paused
                print(f'\n{"⏸️  PAUSADO" if paused else "▶️  REANUDADO"}\n')
        
        cv2.destroyAllWindows()
        
        if self.image_count >= self.total_photos:
            print('\n✅ ¡100 fotos capturadas!')
    
    def stop_stream(self):
        print('\n📹 Deteniendo stream...')
        try:
            self.tello.streamoff()
        except:
            pass
    
    def disconnect(self):
        print('🔌 Desconectando...')

if __name__ == '__main__':
    print('='*70)
    print('📸 TELLO AUTO CAPTURE - 100 FOTOS CON PAUSAS')
    print('='*70)
    
    dataset_name = input('\n📁 Nombre del dataset [datasetDron]: ').strip() or 'datasetDron'
    
    print(f'\n⚙️  Configuración: 100 fotos, pausa cada 20 fotos')
    
    camera = TelloAutoCapture(
        save_dir=dataset_name,
        total_photos=60,
        interval=1,
        cooldown_every=20,
        cooldown_time=3
    )
    
    try:
        if not camera.connect():
            exit(1)
        
        camera.start_stream()
        camera.auto_capture()
        
    except KeyboardInterrupt:
        print('\n⚠️  Interrumpido')
    
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
        print(f'📊 Fotos capturadas: {camera.image_count}/100')
        print(f'📁 Guardadas en: {dataset_name}/')
        print('='*70)