#!/usr/bin/env python3
"""
Script para probar modelo YOLO entrenado con Tello en tiempo real
Detecta: Cooler, Gas_station, Gas_tank, Oxxo, Pipes, Tree, Truck
"""

import cv2
from djitellopy import Tello
from ultralytics import YOLO
import time
import numpy as np

class TelloYOLODetection:
    def __init__(self, model_path='best.pt'):
        self.tello = Tello()
        self.model = None
        self.model_path = model_path
        
        # Clases del modelo
        self.class_names = [
            'Cooler',
            'Gas_station',
            'Gas_tank',
            'Oxxo',
            'Pipes',
            'Tree',
            'Truck'
        ]
        
        # Colores para cada clase (BGR)
        self.colors = [
            (255, 0, 0),     # Cooler - Azul
            (0, 255, 0),     # Gas_station - Verde
            (0, 0, 255),     # Gas_tank - Rojo
            (255, 255, 0),   # Oxxo - Cyan
            (255, 0, 255),   # Pipes - Magenta
            (0, 255, 255),   # Tree - Amarillo
            (128, 0, 128),   # Truck - Púrpura
        ]
    
    def load_model(self):
        """Cargar modelo YOLO"""
        print(f'🤖 Cargando modelo: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            print(f'✅ Modelo cargado exitosamente')
            print(f'   📊 Clases: {len(self.class_names)}')
            print(f'   🏷️  {", ".join(self.class_names)}')
            return True
        except Exception as e:
            print(f'❌ Error cargando modelo: {e}')
            return False
    
    def connect(self):
        """Conectar al Tello"""
        print('\n🔌 Conectando al Tello...')
        try:
            self.tello.connect()
            battery = self.tello.get_battery()
            temp = self.tello.get_temperature()
            
            print(f'✅ Conectado')
            print(f'   🔋 Batería: {battery}%')
            print(f'   🌡️  Temperatura: {temp}°C')
            
            if battery < 15:
                print('⚠️  Batería baja!')
                return False
            
            return True
            
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def start_stream(self):
        """Iniciar stream"""
        print('\n📹 Iniciando stream...')
        try:
            self.tello.streamon()
            time.sleep(2)
            print('✅ Stream activo')
            return True
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def draw_detections(self, frame, results):
        """Dibujar detecciones en el frame"""
        detections_info = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Obtener coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Obtener clase y confianza
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Nombre de la clase
                class_name = self.class_names[cls] if cls < len(self.class_names) else f'Class {cls}'
                
                # Color para esta clase
                color = self.colors[cls] if cls < len(self.colors) else (255, 255, 255)
                
                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Preparar label
                label = f'{class_name} {conf:.2f}'
                
                # Fondo para el texto
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # Texto
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                
                # Guardar info para estadísticas
                detections_info.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return frame, detections_info
    
    def run_detection(self, conf_threshold=0.5):
        """Ejecutar detección en tiempo real"""
        print('\n' + '='*70)
        print('🎯 DETECCIÓN YOLO EN TIEMPO REAL - TELLO')
        print('='*70)
        print(f'Umbral de confianza: {conf_threshold}')
        print('\nControles:')
        print('  [Q] - Salir')
        print('  [S] - Capturar foto con detecciones')
        print('  [+] - Aumentar confianza (+0.05)')
        print('  [-] - Disminuir confianza (-0.05)')
        print('  [ESC] - Salir')
        print('='*70 + '\n')
        
        input('Presiona ENTER para comenzar...')
        
        frame_read = self.tello.get_frame_read()
        
        capture_count = 0
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        print('🚀 Detección iniciada...\n')
        
        while True:
            # Obtener frame
            frame = frame_read.frame
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Convertir RGB a BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Hacer detección con YOLO
            try:
                results = self.model(frame_bgr, conf=conf_threshold, verbose=False)
                
                # Dibujar detecciones
                frame_bgr, detections = self.draw_detections(frame_bgr, results)
                
            except Exception as e:
                print(f'❌ Error en detección: {e}')
                detections = []
            
            # Calcular FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
            
            # Info overlay
            h, w, _ = frame_bgr.shape
            
            # FPS y batería
            try:
                battery = self.tello.get_battery()
                info_text = f'FPS: {fps:.1f} | Bateria: {battery}% | Conf: {conf_threshold:.2f}'
            except:
                info_text = f'FPS: {fps:.1f} | Conf: {conf_threshold:.2f}'
            
            cv2.putText(frame_bgr, info_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Número de detecciones
            det_text = f'Detecciones: {len(detections)}'
            cv2.putText(frame_bgr, det_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Lista de detecciones
            y_offset = 90
            for det in detections[:5]:  # Mostrar máximo 5
                det_info = f"{det['class']}: {det['confidence']:.2f}"
                cv2.putText(frame_bgr, det_info, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 25
            
            # Controles
            cv2.putText(frame_bgr, 'Q=Salir | S=Capturar | +/- Conf', 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mostrar
            cv2.imshow('Tello YOLO Detection', frame_bgr)
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q o ESC
                print('\n👋 Cerrando...')
                break
            
            elif key == ord('s'):  # Capturar
                import os
                from datetime import datetime
                
                if not os.path.exists('detections'):
                    os.makedirs('detections')
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'detections/tello_det_{timestamp}.jpg'
                cv2.imwrite(filename, frame_bgr)
                capture_count += 1
                
                print(f'📸 Guardado: {filename} ({len(detections)} detecciones)')
            
            elif key == ord('+') or key == ord('='):  # Aumentar confianza
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f'📊 Confianza: {conf_threshold:.2f}')
            
            elif key == ord('-') or key == ord('_'):  # Disminuir confianza
                conf_threshold = max(0.1, conf_threshold - 0.05)
                print(f'📊 Confianza: {conf_threshold:.2f}')
        
        cv2.destroyAllWindows()
        
        if capture_count > 0:
            print(f'\n📊 Capturas guardadas: {capture_count}')
    
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
    print('='*70)
    print('🎯 TEST DE MODELO YOLO CON TELLO')
    print('='*70)
    
    # Configuración
    MODEL_PATH = 'best.pt'
    CONF_THRESHOLD = 0.5  # Umbral de confianza inicial
    
    detector = TelloYOLODetection(model_path=MODEL_PATH)
    
    try:
        # 1. Cargar modelo
        if not detector.load_model():
            print('\n❌ No se pudo cargar el modelo')
            exit(1)
        
        # 2. Conectar Tello
        if not detector.connect():
            print('\n❌ No se pudo conectar al Tello')
            exit(1)
        
        # 3. Iniciar stream
        if not detector.start_stream():
            print('\n❌ No se pudo iniciar el stream')
            exit(1)
        
        # 4. Ejecutar detección
        detector.run_detection(conf_threshold=CONF_THRESHOLD)
        
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrumpido')
    
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        detector.stop_stream()
        detector.disconnect()
        
        print('\n' + '='*70)
        print('✅ Test finalizado')
        print('='*70)
