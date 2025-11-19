#!/usr/bin/env python3
"""
TEST VISUAL - LINE FOLLOWER (SIN VOLAR)
Prueba la detección de pipe blanco y cálculo de centroide sin despegar
Útil para calibración y debugging
"""

import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time


class LineFollowerTest:
    def __init__(self, model_path='best.pt', target_class='Pipes'):
        self.tello = Tello()
        self.model = None
        self.model_path = model_path
        self.target_class = target_class
        self.target_class_id = None
        self.conf_threshold = 0.5
        
        # Estadísticas
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
    def load_model(self):
        """Cargar modelo YOLO"""
        print(f'🤖 Cargando modelo: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            
            # Obtener nombres de clases
            class_names = list(self.model.names.values())
            print(f'✅ Modelo cargado')
            print(f'   📊 Clases disponibles: {class_names}')
            
            # Buscar ID de la clase objetivo
            self.target_class_id = None
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == self.target_class.lower():
                    self.target_class_id = class_id
                    print(f'   🎯 Clase objetivo: "{self.target_class}" (ID: {self.target_class_id})')
                    break
            
            if self.target_class_id is None:
                print(f'   ⚠️  ADVERTENCIA: Clase "{self.target_class}" no encontrada')
                print(f'   📝 Clases disponibles: {class_names}')
                print(f'   ℹ️  El sistema mostrará cualquier detección (sin filtrar)')
            
            return True
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def connect(self):
        """Conectar al Tello"""
        print('🔌 Conectando...')
        try:
            self.tello.connect()
            battery = self.tello.get_battery()
            print(f'✅ Conectado (Batería: {battery}%)')
            return True
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def start_stream(self):
        """Iniciar stream"""
        print('📹 Iniciando stream...')
        try:
            self.tello.streamon()
            time.sleep(2)
            print('✅ Stream activo')
            return True
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def calculate_centroid(self, mask):
        """Calcular centroide de máscara"""
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        return None
    
    def process_frame(self, frame):
        """Procesar frame: detectar, calcular centroide y visualizar"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # YOLO detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Crear frame de salida
        output_frame = frame.copy()
        
        # Verificar detecciones
        if not results or len(results) == 0:
            return output_frame, None, None
        
        result = results[0]
        
        if result.masks is None or len(result.masks) == 0:
            return output_frame, None, None
        
        # FILTRAR SOLO LA CLASE OBJETIVO
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Encontrar detecciones de la clase objetivo
        valid_detections = []
        if self.target_class_id is not None:
            # Filtrar por clase específica
            for idx, cls in enumerate(classes):
                if cls == self.target_class_id:
                    valid_detections.append(idx)
        else:
            # Si no se especificó clase, usar todas
            valid_detections = list(range(len(classes)))
        
        # Verificar si hay detecciones válidas
        if len(valid_detections) == 0:
            return output_frame, None, None
        
        # Tomar la detección con mayor confianza de la clase objetivo
        valid_confidences = [confidences[idx] for idx in valid_detections]
        best_valid_idx = valid_detections[np.argmax(valid_confidences)]
        best_conf = confidences[best_valid_idx]
        
        # Obtener máscara
        mask = result.masks.data[best_valid_idx].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Calcular centroide
        centroid = self.calculate_centroid(mask_binary)
        
        # Calcular error si hay centroide
        error_x = error_y = 0
        if centroid:
            error_x = centroid[0] - center_x
            error_y = center_y - centroid[1]  # Invertido en imagen
        
        # === VISUALIZACIÓN ===
        
        # 1. Dibujar máscara con transparencia
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_binary > 0] = (0, 255, 0)  # Verde
        output_frame = cv2.addWeighted(output_frame, 0.7, colored_mask, 0.3, 0)
        
        # 2. Contorno de la máscara
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 3)
        
        # 3. Centro del frame (cruz amarilla)
        cv2.drawMarker(output_frame, (center_x, center_y), 
                      (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(output_frame, 'CENTRO', (center_x + 15, center_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 4. Centroide (círculo rojo)
        if centroid:
            cv2.circle(output_frame, centroid, 12, (0, 0, 255), -1)
            cv2.circle(output_frame, centroid, 18, (255, 255, 255), 3)
            cv2.putText(output_frame, f'CENTROIDE', 
                       (centroid[0] + 20, centroid[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 5. Línea de error (magenta)
            cv2.line(output_frame, (center_x, center_y), centroid, 
                    (255, 0, 255), 3)
            
            # 6. Vectores de error
            # Horizontal
            cv2.arrowedLine(output_frame, 
                          (center_x, center_y), 
                          (centroid[0], center_y),
                          (255, 128, 0), 2, tipLength=0.3)
            
            # Vertical
            cv2.arrowedLine(output_frame, 
                          (center_x, center_y), 
                          (center_x, centroid[1]),
                          (0, 128, 255), 2, tipLength=0.3)
        
        # === INFORMACIÓN EN PANTALLA ===
        
        # Panel de info superior
        info_y = 30
        line_height = 35
        
        # Batería y FPS
        try:
            battery = self.tello.get_battery()
            info = f'FPS: {self.fps:.1f} | Bateria: {battery}% | Conf: {self.conf_threshold:.2f}'
        except:
            info = f'FPS: {self.fps:.1f} | Conf: {self.conf_threshold:.2f}'
        
        cv2.putText(output_frame, info, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += line_height
        
        # Estado de detección con nombre de clase
        class_name = self.model.names[classes[best_valid_idx]]
        if centroid:
            status = f'{class_name.upper()} DETECTADO - Confianza: {best_conf:.2f}'
            color = (0, 255, 0)
        else:
            status = 'NO DETECTADO'
            color = (0, 0, 255)
        
        cv2.putText(output_frame, status, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        info_y += line_height
        
        # Mostrar cuántas detecciones fueron filtradas
        total_detections = len(classes)
        filtered_out = total_detections - len(valid_detections)
        if filtered_out > 0:
            filter_info = f'Ignoradas: {filtered_out} otras detecciones'
            cv2.putText(output_frame, filter_info, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            info_y += line_height
        
        # Coordenadas del centroide
        if centroid:
            coords = f'Centroide: ({centroid[0]}, {centroid[1]})'
            cv2.putText(output_frame, coords, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += line_height
        
        # Errores
        if centroid:
            errors = f'Error X: {error_x:+4d} px | Error Y: {error_y:+4d} px'
            cv2.putText(output_frame, errors, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            info_y += line_height
            
            # Comandos simulados (sin volar)
            kp_x = 0.5
            kp_y = 0.5
            left_right = int(kp_x * error_x)
            up_down = int(kp_y * error_y)
            forward = 20
            
            cmd = f'Comandos simulados: LR:{left_right:+3d} UD:{up_down:+3d} FW:{forward:+3d}'
            cv2.putText(output_frame, cmd, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Panel de ayuda inferior
        help_y = h - 80
        cv2.putText(output_frame, 'MODO PRUEBA - NO VOLARA', 
                   (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        help_y += 30
        cv2.putText(output_frame, 'Q=Salir | S=Capturar | +/-=Confianza', 
                   (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame, centroid, (error_x, error_y)
    
    def run(self):
        """Loop principal de prueba"""
        print('\n' + '='*70)
        print('🧪 TEST VISUAL - LINE FOLLOWER (SIN VOLAR)')
        print('='*70)
        print('\n✅ Este modo NO despegará el dron')
        print('   Solo mostrará detección y centroide\n')
        print('Controles:')
        print('  [Q] o [ESC] - Salir')
        print('  [S] - Capturar pantalla')
        print('  [+/-] - Ajustar confianza')
        print('='*70 + '\n')
        
        input('Presiona ENTER para comenzar...')
        
        frame_read = self.tello.get_frame_read()
        
        # Estadísticas
        total_frames = 0
        detected_frames = 0
        capture_count = 0
        
        print('📹 Mostrando stream...\n')
        
        while True:
            # Obtener frame
            frame = frame_read.frame
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Convertir RGB a BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Procesar
            output_frame, centroid, error = self.process_frame(frame_bgr)
            
            # Actualizar estadísticas
            total_frames += 1
            if centroid:
                detected_frames += 1
            
            # Calcular FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.fps = 30 / (time.time() - self.fps_time)
                self.fps_time = time.time()
            
            # Mostrar
            cv2.imshow('Line Follower - TEST (No volara)', output_frame)
            
            # Teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q o ESC
                print('\n👋 Cerrando...')
                break
            
            elif key == ord('s'):  # Capturar
                import os
                from datetime import datetime
                
                if not os.path.exists('test_captures'):
                    os.makedirs('test_captures')
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'test_captures/test_{timestamp}.jpg'
                cv2.imwrite(filename, output_frame)
                capture_count += 1
                
                status = "CON centroide" if centroid else "SIN centroide"
                print(f'📸 Guardado: {filename} ({status})')
            
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                print(f'📊 Confianza: {self.conf_threshold:.2f}')
            
            elif key == ord('-') or key == ord('_'):
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                print(f'📊 Confianza: {self.conf_threshold:.2f}')
        
        # Estadísticas finales
        cv2.destroyAllWindows()
        
        print('\n' + '='*70)
        print('📊 ESTADÍSTICAS')
        print('='*70)
        print(f'Total de frames: {total_frames}')
        print(f'Frames con detección: {detected_frames}')
        if total_frames > 0:
            detection_rate = (detected_frames / total_frames) * 100
            print(f'Tasa de detección: {detection_rate:.1f}%')
        print(f'Capturas guardadas: {capture_count}')
        print('='*70)
        
        # Recomendaciones
        if total_frames > 0:
            detection_rate = (detected_frames / total_frames) * 100
            print('\n💡 RECOMENDACIONES:')
            
            if detection_rate > 90:
                print('✅ Excelente detección! Listo para volar.')
            elif detection_rate > 70:
                print('⚠️  Buena detección, pero puede mejorar.')
                print('   - Mejora iluminación')
                print('   - Acerca el dron al pipe')
            elif detection_rate > 50:
                print('⚠️  Detección inconsistente.')
                print('   - Verifica que el modelo detecte pipes blancos')
                print('   - Mejora iluminación')
                print('   - Baja conf_threshold')
            else:
                print('❌ Detección muy baja!')
                print('   - Verifica el modelo')
                print('   - Mejora iluminación significativamente')
                print('   - NO vueles hasta mejorar detección')
    
    def stop_stream(self):
        """Detener stream"""
        try:
            self.tello.streamoff()
        except:
            pass
    
    def disconnect(self):
        """Desconectar"""
        try:
            self.tello.end()
        except:
            pass


def main():
    print('='*70)
    print('🧪 LINE FOLLOWER - TEST VISUAL (NO VUELA)')
    print('='*70)
    
    MODEL_PATH = 'best.pt'
    TARGET_CLASS = 'Pipes'    # Cambia al nombre de la clase que quieres seguir
    
    print(f'\n⚙️  Configuración:')
    print(f'   📦 Modelo: {MODEL_PATH}')
    print(f'   🎯 Clase objetivo: "{TARGET_CLASS}"')
    print(f'   ℹ️  Solo mostrará detecciones de esta clase\n')
    
    tester = LineFollowerTest(model_path=MODEL_PATH, target_class=TARGET_CLASS)
    
    try:
        # Cargar modelo
        if not tester.load_model():
            print('\n❌ No se pudo cargar el modelo')
            return
        
        # Conectar
        if not tester.connect():
            print('\n❌ No se pudo conectar')
            return
        
        # Stream
        if not tester.start_stream():
            print('\n❌ No se pudo iniciar stream')
            return
        
        # Ejecutar test
        tester.run()
        
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrumpido')
    
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        tester.stop_stream()
        tester.disconnect()
        
        print('\n✅ Test finalizado')


if __name__ == '__main__':
    main()