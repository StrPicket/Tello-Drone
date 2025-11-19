#!/usr/bin/env python3
"""
LINE FOLLOWER con YOLO Segmentation + Control PD
Reduce oscilaciones con control derivativo
"""

import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time


class LineFollowerYOLO:
    def __init__(self, model_path='best.pt', target_class='Pipes'):
        """
        Inicializar Line Follower con control PD
        
        Args:
            model_path: Ruta al modelo YOLO entrenado
            target_class: Nombre de la clase a seguir (ej: 'Pipes', 'line', etc.)
        """
        self.tello = Tello()
        self.model = None
        self.model_path = model_path
        self.target_class = target_class
        self.target_class_id = None
        
        # ===== PARÁMETROS DE CONTROL PD =====
        # Control lateral (left/right)
        self.kp_x = 0.3        # Ganancia proporcional REDUCIDA
        self.kd_x = 0.8        # Ganancia derivativa (amortigua oscilaciones)
        
        # Control vertical (up/down)
        self.kp_y = 0.3        # Ganancia proporcional REDUCIDA
        self.kd_y = 0.8        # Ganancia derivativa
        
        # Velocidad adelante
        self.forward_speed = 10
        
        # Límites de velocidad
        self.max_speed = 25    # Reducido para mayor estabilidad
        self.min_speed = 5
        
        # Umbral de confianza REDUCIDO para mayor detección
        self.conf_threshold = 0.3  # Reducido de 0.5 a 0.3
        
        # Zona muerta más grande
        self.deadzone_x = 80   # Aumentado
        self.deadzone_y = 80   # Aumentado
        
        # ===== VARIABLES PARA CONTROL DERIVATIVO =====
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.prev_time = time.time()
        
        # Filtro de suavizado para error (reduce ruido)
        self.error_x_filtered = 0
        self.error_y_filtered = 0
        self.filter_alpha = 0.7  # Factor de suavizado (0-1, más alto = más filtrado)
        
        # Estado
        self.flying = False
        self.started_following = False
        self.line_detected = False
        self.frames_without_line = 0
        self.max_frames_without_line = 30  # Aumentado de 15 a 30 (más tolerante)
        
        # Estadísticas
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
    def load_model(self):
        """Cargar modelo YOLO de segmentación"""
        print(f'🤖 Cargando modelo: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            
            class_names = list(self.model.names.values())
            
            print(f'✅ Modelo cargado')
            print(f'   📊 Clases disponibles: {class_names}')
            
            self.target_class_id = None
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == self.target_class.lower():
                    self.target_class_id = class_id
                    print(f'   🎯 Clase objetivo: "{self.target_class}" (ID: {self.target_class_id})')
                    break
            
            if self.target_class_id is None:
                print(f'   ⚠️  ADVERTENCIA: Clase "{self.target_class}" no encontrada')
                print(f'   📝 Clases disponibles: {class_names}')
            
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
            
            print(f'✅ Conectado')
            print(f'   🔋 Batería: {battery}%')
            
            if battery < 20:
                print('⚠️  Batería baja!')
                return False
            
            return True
            
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def start_stream(self):
        """Iniciar stream de video"""
        print('\n📹 Iniciando stream...')
        try:
            self.tello.streamon()
            time.sleep(2)
            print('✅ Stream activo')
            return True
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def calculate_centroid(self, mask):
        """Calcular centroide de máscara binaria"""
        moments = cv2.moments(mask)
        
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        
        return None
    
    def detect_line_and_centroid(self, frame):
        """Detectar línea con YOLO y calcular centroide"""
        h, w = frame.shape[:2]
        
        # Verificar que el frame sea válido
        if frame is None or frame.size == 0:
            print('⚠️  Frame inválido recibido')
            return None, None, frame
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        if not results or len(results) == 0:
            return None, None, frame
        
        result = results[0]
        
        if result.masks is None or len(result.masks) == 0:
            return None, None, frame
        
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        valid_detections = []
        if self.target_class_id is not None:
            for idx, cls in enumerate(classes):
                if cls == self.target_class_id:
                    valid_detections.append(idx)
        else:
            valid_detections = list(range(len(classes)))
        
        if len(valid_detections) == 0:
            return None, None, frame
        
        valid_confidences = [confidences[idx] for idx in valid_detections]
        best_valid_idx = valid_detections[np.argmax(valid_confidences)]
        
        mask = result.masks.data[best_valid_idx].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        
        centroid = self.calculate_centroid(mask_binary)
        
        annotated_frame = frame.copy()
        
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_binary > 0] = (0, 255, 0)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, colored_mask, 0.3, 0)
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)
        
        if centroid:
            cv2.circle(annotated_frame, centroid, 10, (0, 0, 255), -1)
            cv2.circle(annotated_frame, centroid, 15, (255, 255, 255), 2)
            
            center = (w // 2, h // 2)
            cv2.line(annotated_frame, center, centroid, (255, 0, 255), 2)
        
        center = (w // 2, h // 2)
        cv2.drawMarker(annotated_frame, center, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        conf = confidences[best_valid_idx]
        class_name = self.model.names[classes[best_valid_idx]]
        cv2.putText(annotated_frame, f'{class_name}: {conf:.2f}', 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return centroid, mask_binary, annotated_frame
    
    def calculate_control_commands(self, centroid, frame_shape):
        """
        Control PD para reducir oscilaciones
        P = proporcional al error actual
        D = proporcional a la velocidad de cambio del error
        """
        h, w = frame_shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Evitar división por cero
        if dt < 0.001:
            dt = 0.001
        
        if centroid is None:
            # Sin detección: resetear errores
            self.prev_error_x = 0
            self.prev_error_y = 0
            self.error_x_filtered = 0
            self.error_y_filtered = 0
            self.prev_time = current_time
            return 0, 0, 0, 0
        
        cx, cy = centroid
        
        # Calcular error
        error_x = cx - center_x
        error_y = center_y - cy
        
        # Aplicar filtro de suavizado (reduce ruido en detección)
        self.error_x_filtered = (self.filter_alpha * self.error_x_filtered + 
                                 (1 - self.filter_alpha) * error_x)
        self.error_y_filtered = (self.filter_alpha * self.error_y_filtered + 
                                 (1 - self.filter_alpha) * error_y)
        
        # Usar errores filtrados
        error_x = self.error_x_filtered
        error_y = self.error_y_filtered
        
        # Aplicar zona muerta
        if abs(error_x) < self.deadzone_x:
            error_x = 0
        if abs(error_y) < self.deadzone_y:
            error_y = 0
        
        # Calcular derivada del error (velocidad de cambio)
        d_error_x = (error_x - self.prev_error_x) / dt
        d_error_y = (error_y - self.prev_error_y) / dt
        
        # Control PD
        left_right = int(self.kp_x * error_x + self.kd_x * d_error_x)
        up_down = int(self.kp_y * error_y + self.kd_y * d_error_y)
        
        # Limitar velocidades
        left_right = int(np.clip(left_right, -self.max_speed, self.max_speed))
        up_down = int(np.clip(up_down, -self.max_speed, self.max_speed))
        
        # Velocidad constante hacia adelante
        forward = int(self.forward_speed)
        yaw = 0
        
        # Guardar estado para siguiente iteración
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = current_time
        
        return left_right, up_down, forward, yaw
    
    def draw_overlay(self, frame, centroid, left_right, up_down, forward):
        """Dibujar información en el frame"""
        h, w = frame.shape[:2]
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = 30 / (time.time() - self.fps_time)
            self.fps_time = time.time()
        
        try:
            battery = self.tello.get_battery()
            info_text = f'FPS: {self.fps:.1f} | Bat: {battery}% | Flying: {self.flying}'
        except:
            info_text = f'FPS: {self.fps:.1f} | Flying: {self.flying}'
        
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if centroid:
            status = f'PIPE: ({centroid[0]}, {centroid[1]})'
            color = (0, 255, 0)
            self.line_detected = True
            self.frames_without_line = 0
        else:
            self.frames_without_line += 1
            status = f'NO PIPE! ({self.frames_without_line}/{self.max_frames_without_line})'
            color = (0, 0, 255)
            self.line_detected = False
        
        cv2.putText(frame, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cmd_text = f'Comandos: LR:{left_right:+3d} UD:{up_down:+3d} FW:{forward:+3d}'
        cv2.putText(frame, cmd_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Mostrar errores
        if centroid:
            center_x = w // 2
            error_x = int(self.error_x_filtered)
            error_text = f'Error X: {error_x:+4d} px | Kp:{self.kp_x} Kd:{self.kd_x}'
            cv2.putText(frame, error_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, 'T=Despegar | L=Aterrizar | Q=Salir', 
                   (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def takeoff(self):
        """Despegar el dron"""
        if not self.flying:
            print('\n🚁 Despegando...')
            try:
                self.tello.takeoff()
                time.sleep(3)
                
                print('🚀 Avanzando para posicionar...')
                self.tello.send_rc_control(0, 15, 0, 0)
                time.sleep(2)
                self.tello.send_rc_control(0, 0, 0, 0)
                
                # Resetear errores después de despegue
                self.prev_error_x = 0
                self.prev_error_y = 0
                self.error_x_filtered = 0
                self.error_y_filtered = 0
                self.prev_time = time.time()
                
                self.flying = True
                print('✅ En el aire - Control PD activo')
                return True
            except Exception as e:
                print(f'❌ Error al despegar: {e}')
                return False
        return True
    
    def land(self):
        """Aterrizar el dron"""
        if self.flying:
            print('\n🛬 Aterrizando...')
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                
                self.tello.land()
                time.sleep(2)
                self.flying = False
                print('✅ Aterrizado')
                return True
            except Exception as e:
                print(f'❌ Error al aterrizar: {e}')
                return False
        return True
    
    def run(self):
        """Loop principal"""
        print('\n' + '='*70)
        print('🎯 LINE FOLLOWER - CONTROL PD ANTI-OSCILACIÓN')
        print('='*70)
        print('\nControles:')
        print('  [T] - Despegar')
        print('  [L] - Aterrizar')
        print('  [Q] - Salir')
        print('\n⚙️  Control PD:')
        print(f'  Kp_x: {self.kp_x} | Kd_x: {self.kd_x}')
        print(f'  Kp_y: {self.kp_y} | Kd_y: {self.kd_y}')
        print(f'  Zona muerta: ±{self.deadzone_x} px')
        print('='*70 + '\n')
        
        input('Presiona ENTER para comenzar...')
        
        frame_read = self.tello.get_frame_read()
        
        while True:
            frame = frame_read.frame
            
            if frame is None:
                print('⚠️  Frame None recibido - esperando...')
                time.sleep(0.1)
                continue
            
            # Verificar si el frame es válido (no congelado)
            if hasattr(self, 'last_frame_hash'):
                current_hash = hash(frame.tobytes())
                if current_hash == self.last_frame_hash:
                    self.frozen_frame_count = getattr(self, 'frozen_frame_count', 0) + 1
                    if self.frozen_frame_count > 30:
                        print('⚠️  Stream congelado detectado!')
                else:
                    self.frozen_frame_count = 0
                self.last_frame_hash = current_hash
            else:
                self.last_frame_hash = hash(frame.tobytes())
                self.frozen_frame_count = 0
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            centroid, mask, annotated_frame = self.detect_line_and_centroid(frame_bgr)
            
            left_right, up_down, forward, yaw = self.calculate_control_commands(
                centroid, frame_bgr.shape
            )
            
            if self.flying:
                if centroid:
                    if not self.started_following:
                        print('\n🟢 PIPE DETECTADO - Iniciando seguimiento PD...')
                        self.started_following = True
                    
                    try:
                        self.tello.send_rc_control(left_right, forward, up_down, yaw)
                    except Exception as e:
                        print(f'❌ Error enviando comandos: {e}')
                    
                    self.frames_without_line = 0
                    
                else:
                    self.frames_without_line += 1
                    
                    if self.started_following:
                        # MODO DE RECUPERACIÓN: intentar seguir adelante brevemente
                        if self.frames_without_line < 10:
                            # Primeros 10 frames: continuar adelante lentamente
                            print(f'\n⚠️  PIPE PERDIDO - Continuando adelante ({self.frames_without_line}/10)')
                            try:
                                self.tello.send_rc_control(0, 8, 0, 0)  # Solo adelante lento
                            except Exception as e:
                                print(f'❌ Error: {e}')
                        elif self.frames_without_line < self.max_frames_without_line:
                            # Frames 10-30: detenerse y esperar
                            print(f'\n⚠️  PIPE PERDIDO - Esperando ({self.frames_without_line}/{self.max_frames_without_line})')
                            self.tello.send_rc_control(0, 0, 0, 0)
                        else:
                            # Después de 30 frames: aterrizar
                            print('\n🛬 ATERRIZAJE AUTOMÁTICO')
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.5)
                            self.land()
                            print('\n⚠️  Sistema detenido.')
                            while True:
                                cv2.imshow('Line Follower - DETENIDO', annotated_frame)
                                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                                    break
                            break
                    else:
                        self.tello.send_rc_control(0, 0, 0, 0)
            
            display_frame = self.draw_overlay(
                annotated_frame, centroid, left_right, up_down, forward
            )
            
            if self.flying:
                if self.started_following:
                    if centroid:
                        status = 'SIGUIENDO (PD Activo)'
                        color = (0, 255, 0)
                    else:
                        status = f'PERDIDO! Aterrizando en {self.max_frames_without_line - self.frames_without_line}'
                        color = (0, 0, 255)
                else:
                    status = 'BUSCANDO PIPE...'
                    color = (0, 255, 255)
            else:
                status = 'EN TIERRA - Presiona T'
                color = (255, 255, 255)
            
            cv2.putText(display_frame, status, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Line Follower - PD Control', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                print('\n👋 Cerrando...')
                break
            
            elif key == ord('t'):
                if not self.flying:
                    self.takeoff()
            
            elif key == ord('l'):
                if self.flying:
                    print('\n🛬 ATERRIZAJE MANUAL')
                    self.tello.send_rc_control(0, 0, 0, 0)
                    self.land()
        
        cv2.destroyAllWindows()
        
        if self.flying:
            self.land()
    
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


def main():
    print('='*70)
    print('🎯 LINE FOLLOWER CON CONTROL PD ANTI-OSCILACIÓN')
    print('='*70)
    
    MODEL_PATH = 'best.pt'
    TARGET_CLASS = 'Pipes'
    
    print(f'\n⚙️  Configuración:')
    print(f'   📦 Modelo: {MODEL_PATH}')
    print(f'   🎯 Clase: "{TARGET_CLASS}"')
    print(f'   🎛️  Control: PD (Proporcional-Derivativo)')
    print(f'   📉 Oscilaciones: REDUCIDAS\n')
    
    follower = LineFollowerYOLO(model_path=MODEL_PATH, target_class=TARGET_CLASS)
    
    try:
        if not follower.load_model():
            print('\n❌ Error cargando modelo')
            return
        
        if not follower.connect():
            print('\n❌ Error conectando')
            return
        
        if not follower.start_stream():
            print('\n❌ Error iniciando stream')
            return
        
        follower.run()
        
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrumpido')
    
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        if follower.flying:
            follower.land()
        
        follower.stop_stream()
        follower.disconnect()
        
        print('\n' + '='*70)
        print('✅ Line Follower finalizado')
        print('='*70)


if __name__ == '__main__':
    main()