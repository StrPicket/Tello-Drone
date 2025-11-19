#!/usr/bin/env python3
"""
LINE FOLLOWER AUTÓNOMO con Exploración Inteligente + MULTITHREADING
- Thread dedicado para captura de frames
- Thread dedicado para procesamiento YOLO
- Thread principal para máquina de estados y control
- Buffer de frames para evitar pérdidas
"""

import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time
from enum import Enum
from threading import Thread, Lock, Event
from collections import deque
import queue


class DroneState(Enum):
    """Estados del dron"""
    GROUNDED = "EN TIERRA"
    INITIAL_ADVANCE = "AVANCE INICIAL"
    SEARCHING = "BUSCANDO PIPE"
    FOLLOWING = "SIGUIENDO PIPE"
    EXPLORING = "EXPLORANDO"
    MANEUVER = "EJECUTANDO MANIOBRA"
    LANDING = "ATERRIZANDO"


class ExplorationDirection(Enum):
    """Direcciones de exploración"""
    FORWARD = "ADELANTE"
    LEFT = "IZQUIERDA"
    RIGHT = "DERECHA"
    NONE = "NO ENCONTRADO"


class FrameBuffer:
    """Buffer thread-safe para frames"""
    def __init__(self, maxsize=2):
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = Lock()
        
    def put(self, frame):
        """Agregar frame al buffer"""
        with self.lock:
            self.latest_frame = frame
        # Si la cola está llena, descarta el más viejo
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get(self):
        """Obtener último frame"""
        with self.lock:
            return self.latest_frame


class DetectionResult:
    """Resultado de detección thread-safe"""
    def __init__(self):
        self.centroid = None
        self.mask = None
        self.annotated_frame = None
        self.pipes_forward = 0
        self.pipes_left = 0
        self.pipes_right = 0
        self.lock = Lock()
        self.timestamp = time.time()
        self.has_data = False
    
    def update(self, centroid, mask, annotated_frame, pipes_fwd=0, pipes_left=0, pipes_right=0):
        with self.lock:
            self.centroid = centroid
            self.mask = mask
            self.annotated_frame = annotated_frame
            self.pipes_forward = pipes_fwd
            self.pipes_left = pipes_left
            self.pipes_right = pipes_right
            self.timestamp = time.time()
            self.has_data = True
    
    def get(self):
        with self.lock:
            return (self.centroid, self.mask, self.annotated_frame, 
                   self.pipes_forward, self.pipes_left, self.pipes_right)

    def has_valid_data(self):
        with self.lock:
            return self.has_data and self.annotated_frame is not None

class LineFollowerYOLO:
    def __init__(self, model_path='best.pt', target_class='Pipes'):
        self.tello = Tello()
        self.model = None
        self.model_path = model_path
        self.target_class = target_class
        self.target_class_id = None

        self.frame_read  = None

        # === WATCHDOG PARA STREAM ===
        self.last_frame_time = time.time()
        self.stream_timeout = 5.0  # Segundos sin frames antes de considerar muerto
        self.stream_frozen = False
        
        # Control PD
        self.kp_x = 0.3
        self.kd_x = 0.8
        self.kp_y = 0.3
        self.kd_y = 0.8
        self.forward_speed = 10
        self.max_speed = 25
        self.min_speed = 5
        self.conf_threshold = 0.4
        self.deadzone_x = 80
        self.deadzone_y = 80
        
        # Variables PD
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.prev_time = time.time()
        self.error_x_filtered = 0
        self.error_y_filtered = 0
        self.filter_alpha = 0.7
        
        # ===== ROI DINÁMICO =====
        # ROI AMPLIO (para búsqueda/exploración)
        self.roi_wide = {
            'top': 0.1,
            'bottom': 0.7,
            'left': 0.2,
            'right': 0.8
        }
        
        # ROI ESTRECHO (para seguimiento - elimina ruido lateral)
        self.roi_narrow = {
            'top': 0.15,      # Un poco más arriba
            'bottom': 0.65,   # Un poco más abajo
            'left': 0.35,     # MÁS ESTRECHO - elimina lados
            'right': 0.65     # MÁS ESTRECHO - elimina lados
        }
        
        # ROI actual (se cambia según el estado)
        self.current_roi = self.roi_wide.copy()
            
        # EXPLORACIÓN
        self.exploration_step = 0
        self.exploration_direction = ExplorationDirection.NONE
        self.exploration_start_time = None
        
        self.exploration_sequence = [
            ("elevate", 0, 0, 15, 0, 1.5),
            ("look_left", 0, 0, 0, -45, 1.5),
            ("scan_left", 0, 0, 0, 0, 1.0),
            ("center", 0, 0, 0, 45, 1.5),
            ("scan_forward", 0, 0, 0, 0, 1.0),
            ("look_right", 0, 0, 0, 45, 1.5),
            ("scan_right", 0, 0, 0, 0, 1.0),
            ("back_center", 0, 0, 0, -45, 1.5),
        ]
        
        self.detections_left = []
        self.detections_forward = []
        self.detections_right = []
        
        # MANIOBRAS
        self.maneuver_sequences = {
            ExplorationDirection.LEFT: [
                (0, 0, 0, -90, 2.5),
                (0, 15, 0, 0, 2.0),
                (0, 0, 0, 0, 0.5),
            ],
            ExplorationDirection.RIGHT: [
                (0, 0, 0, 90, 2.5),
                (0, 15, 0, 0, 2.0),
                (0, 0, 0, 0, 0.5),
            ],
            ExplorationDirection.FORWARD: [
                (0, 15, 0, 0, 2.0),
                (0, 0, 0, 0, 0.5),
            ],
        }
        
        self.current_maneuver = []
        self.maneuver_step = 0
        self.maneuver_start_time = None
        
        # Estado
        self.state = DroneState.GROUNDED
        self.state_lock = Lock()
        self.flying = False
        self.frames_without_line = 0
        self.max_frames_without_line = 30
        
        # AVANCE INICIAL
        self.initial_advance_time = None
        self.initial_advance_duration = 3.0
        
        # Contador de pipes
        self.pipes_completed = 0
        self.max_pipes = 5
        
        # Estadísticas
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        self.cached_battery = 0
        
        # === THREADING ===
        self.frame_buffer = FrameBuffer(maxsize=2)
        self.detection_result = DetectionResult()
        
        self.stop_threads = Event()
        self.capture_thread = None
        self.detection_thread = None
        self.rc_command_queue = queue.Queue(maxsize=10)
        self.rc_thread = None
        
        # Último comando RC enviado (para evitar spam)
        self.last_rc_command = (0, 0, 0, 0)
        self.last_rc_time = 0
        self.rc_min_interval = 0.05  # 50ms entre comandos
    
    def load_model(self):
        """Cargar modelo YOLO"""
        print(f'🤖 Cargando modelo: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            class_names = list(self.model.names.values())
            
            print(f'✅ Modelo cargado')
            print(f'   📊 Clases: {class_names}')
            
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == self.target_class.lower():
                    self.target_class_id = class_id
                    print(f'   🎯 Clase objetivo: "{self.target_class}" (ID: {self.target_class_id})')
                    break
            
            return True
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def connect(self):
        """Conectar al Tello"""
        print('\n🔌 Conectando al Tello...')
        try:
            self.tello.connect()
            battery = self.tello.get_battery()
            print(f'✅ Conectado | 🔋 Batería: {battery}%')
            return battery >= 20
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def start_stream(self):
        """Iniciar stream"""
        print('\n📹 Iniciando stream...')
        try:
            self.tello.streamon()
            time.sleep(3)  # Aumentado de 2 a 3 segundos
            
            # Verificar que el stream funciona
            print('   🔄 Verificando stream...')
            test_read = self.tello.get_frame_read()
            time.sleep(1)
            
            # Intentar obtener un frame de prueba
            for i in range(10):
                if test_read.frame is not None and test_read.frame.size > 0:
                    print('   ✅ Stream verificado y funcionando')
                    # Guardar el frame_read para uso futuro
                    self.frame_read = test_read
                    return True
                time.sleep(0.5)
            
            print('   ⚠️  Stream iniciado pero sin frames aún')
            self.frame_read = test_read
            return True
            
        except Exception as e:
            print(f'❌ Error: {e}')
            return False

    def attempt_stream_recovery(self):
        """Intentar recuperar el stream cuando se congela"""
        print('\n🔄 Intentando recuperar stream...')
        
        try:
            # Detener frame read actual
            if self.frame_read is not None:
                try:
                    self.frame_read.stop()
                except:
                    pass
            
            # Esperar un poco
            time.sleep(1)
            
            # Reiniciar stream
            self.tello.streamoff()
            time.sleep(1)
            self.tello.streamon()
            time.sleep(2)
            
            # Nuevo frame read
            self.frame_read = self.tello.get_frame_read()
            time.sleep(1)
            
            # Verificar que funciona
            for i in range(5):
                if self.frame_read.frame is not None and self.frame_read.frame.size > 0:
                    print('✅ Stream recuperado exitosamente')
                    self.stream_frozen = False
                    return True
                time.sleep(0.5)
            
            print('⚠️  Recuperación parcial - continuando...')
            self.stream_frozen = False
            return False
            
        except Exception as e:
            print(f'❌ Error en recuperación: {e}')
            return False

    
    # === THREAD: CAPTURA DE FRAMES ===
    def frame_capture_worker(self):
        """Thread dedicado para captura continua de frames con recuperación"""
        print('🎥 Thread de captura iniciado')
        
        # Usar el frame_read ya inicializado
        if self.frame_read is None:
            print('❌ Error: frame_read no inicializado')
            return
        
        consecutive_errors = 0
        max_errors = 30
        frozen_count = 0
        last_frame_hash = None
        frames_received = 0
        last_successful_frame = time.time()
        
        while not self.stop_threads.is_set():
            try:
                frame = self.frame_read.frame
                
                if frame is not None and frame.size > 0:
                    frames_received += 1
                    last_successful_frame = time.time()
                    self.last_frame_time = time.time()  # Actualizar watchdog
                    self.stream_frozen = False
                    
                    # Log cada 200 frames para reducir spam
                    if frames_received % 200 == 0:
                        print(f'🎥 Captura: {frames_received} frames OK')
                    
                    # Detectar frames congelados
                    current_hash = hash(frame.tobytes())
                    if current_hash == last_frame_hash:
                        frozen_count += 1
                        if frozen_count > 50:  # Reducido de 100 a 50
                            print('⚠️  Stream congelado detectado!')
                            self.stream_frozen = True
                            frozen_count = 0
                    else:
                        frozen_count = 0
                    last_frame_hash = current_hash
                    
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.frame_buffer.put(frame_bgr.copy())
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    
                    # Verificar timeout
                    if time.time() - last_successful_frame > self.stream_timeout:
                        print('❌ Stream timeout - sin frames por 5 segundos')
                        self.stream_frozen = True
                        last_successful_frame = time.time()  # Reset para evitar spam
                    
                    if consecutive_errors > max_errors:
                        print('⚠️  Demasiados frames None consecutivos')
                        consecutive_errors = 0
                
                time.sleep(0.025)  # Ligeramente más lento: ~66 FPS
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors % 10 == 0:
                    print(f'❌ Error en captura ({consecutive_errors}): {e}')
                if consecutive_errors > max_errors:
                    print('❌ Demasiados errores en captura')
                    self.stream_frozen = True
                    consecutive_errors = 0
                time.sleep(0.1)
        
        print('🎥 Thread de captura finalizado')
        
    
    # === THREAD: DETECCIÓN YOLO ===
    def detection_worker(self):
        """Thread dedicado para procesamiento YOLO"""
        print('🧠 Thread de detección iniciado')
        
        frame_count = 0
        last_report = time.time()
        skip_counter = 0
        
        while not self.stop_threads.is_set():
            try:
                frame = self.frame_buffer.get()
                
                if frame is None:
                    time.sleep(0.05)
                    continue
                
                # OPTIMIZACIÓN: Procesar solo 1 de cada 4 frames (antes era 3)
                skip_counter += 1
                if skip_counter % 4 != 0:
                    time.sleep(0.01)
                    continue
                
                # NUEVA OPTIMIZACIÓN: Reducir resolución antes de procesar
                frame_small = self._resize_for_processing(frame, target_width=320)
                
                # Realizar detección con frame reducido
                centroid, mask, annotated, pipes_fwd, pipes_left, pipes_right = \
                    self._process_detection(frame_small)
                
                # Escalar centroide de vuelta a tamaño original
                if centroid:
                    h_orig, w_orig = frame.shape[:2]
                    h_small, w_small = frame_small.shape[:2]
                    scale_x = w_orig / w_small
                    scale_y = h_orig / h_small
                    centroid = (int(centroid[0] * scale_x), int(centroid[1] * scale_y))
                
                # Actualizar resultado (usar frame original con anotaciones escaladas)
                self.detection_result.update(
                    centroid, mask, frame,  # Usar frame original para display
                    pipes_fwd, pipes_left, pipes_right
                )
                
                frame_count += 1
                
                # Reportar cada 3 segundos
                if time.time() - last_report > 5.0:
                    print(f'🧠 Detección: {frame_count} frames procesados')
                    frame_count = 0
                    last_report = time.time()
                
                time.sleep(0.08)  # Más lento: ~12 FPS procesamiento
                
            except Exception as e:
                print(f'❌ Error en detección: {e}')
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print('🧠 Thread de detección finalizado')

    def _resize_for_processing(self, frame, target_width=416):
        """Reducir resolución del frame para procesamiento más rápido"""
        h, w = frame.shape[:2]
        aspect_ratio = h / w
        new_width = target_width
        new_height = int(target_width * aspect_ratio)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _process_detection(self, frame):
        """Procesar frame con YOLO (llamado desde thread)"""
        h, w = frame.shape[:2]
        
        # Detección con YOLO
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        if not results or len(results) == 0:
            return None, None, frame, 0, 0, 0
        
        result = results[0]
        
        if result.masks is None or len(result.masks) == 0:
            return None, None, frame, 0, 0, 0
        
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        
        annotated_frame = frame.copy()
        
        # Calcular pipes por región
        pipes_forward, pipes_left, pipes_right = self._count_pipes_by_region(
            result, classes, h, w
        )
        
        # Dibujar ROI (sin extraer región)
        annotated_frame = self._draw_roi(annotated_frame, h, w)
        
        # Dibujar regiones de exploración si está explorando
        with self.state_lock:
            current_state = self.state
        
        if current_state == DroneState.EXPLORING:
            annotated_frame = self._draw_exploration_regions(annotated_frame, h, w)
        
        # Procesar detecciones y encontrar mejor pipe
        pipe_centroid, pipe_mask, annotated_frame = self._process_detections(
            result, classes, confidences, boxes, annotated_frame, h, w
        )
        
        return pipe_centroid, pipe_mask, annotated_frame, pipes_forward, pipes_left, pipes_right

    def _count_pipes_by_region(self, result, classes, h, w):
        """Contar pipes en cada región"""
        center_x = w // 2
        third_h = h // 3
        
        pipes_forward = 0
        pipes_left = 0
        pipes_right = 0
        
        for idx, cls in enumerate(classes):
            if cls != self.target_class_id:
                continue
            
            mask = result.masks.data[idx].cpu().numpy()
            mask_resized = cv2.resize(mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            centroid = self.calculate_centroid(mask_binary)
            if not centroid:
                continue
            
            cx, cy = centroid
            
            if cy < third_h:
                pipes_forward += 1
            elif cx < center_x * 0.6:
                pipes_left += 1
            elif cx > center_x * 1.4:
                pipes_right += 1
        
        return pipes_forward, pipes_left, pipes_right
    
    def _draw_roi(self, frame, h, w):
        """Dibujar ROI en el frame (SIMPLIFICADO)"""
        roi_top_px = int(h * self.current_roi['top'])
        roi_bottom_px = int(h * self.current_roi['bottom'])
        roi_left_px = int(w * self.current_roi['left'])
        roi_right_px = int(w * self.current_roi['right'])
        
        # Determinar color del ROI según el tipo
        with self.state_lock:
            current_state = self.state
        
        if current_state == DroneState.FOLLOWING:
            roi_color = (0, 255, 0)
            thickness = 2
        else:
            roi_color = (0, 255, 255)
            thickness = 2
        
        # Solo rectángulo, sin overlay semitransparente
        cv2.rectangle(frame,
                    (roi_left_px, roi_top_px),
                    (roi_right_px, roi_bottom_px),
                    roi_color, thickness)
        
        return frame
    
    def _draw_exploration_regions(self, frame, h, w):
        """Dibujar regiones de exploración"""
        third_h = h // 3
        center_x = w // 2
        
        cv2.rectangle(frame, (0, 0), (w, third_h), (0, 255, 0), 2)
        cv2.putText(frame, 'FORWARD', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (0, third_h), (int(center_x*0.6), h), (255, 0, 255), 2)
        cv2.putText(frame, 'LEFT', (10, third_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        cv2.rectangle(frame, (int(center_x*1.4), third_h), (w, h), (0, 165, 255), 2)
        cv2.putText(frame, 'RIGHT', (int(center_x*1.4) + 10, third_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        return frame
    
    def _process_detections(self, result, classes, confidences, boxes, frame, h, w):
        """Procesar todas las detecciones y encontrar mejor pipe (OPTIMIZADO)"""
        pipe_centroid = None
        pipe_mask = None
        best_pipe_conf = 0
        
        for idx in range(len(classes)):
            class_id = classes[idx]
            conf = confidences[idx]
            
            # Solo procesar pipes, ignorar otras clases
            is_pipe = (self.target_class_id is not None and class_id == self.target_class_id)
            if not is_pipe:
                continue
            
            mask = result.masks.data[idx].cpu().numpy()
            mask_resized = cv2.resize(mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # SIMPLIFICADO: Solo dibujar contornos y bbox, sin máscaras coloreadas
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # Bbox simple
            x1, y1, x2, y2 = boxes[idx].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label pequeño
            label = f'{conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Calcular centroide
            centroid = self.calculate_centroid(mask_binary)
            if centroid and conf > best_pipe_conf:
                pipe_centroid = centroid
                pipe_mask = mask_binary
                best_pipe_conf = conf
        
        # Dibujar solo centroide principal (SIMPLIFICADO)
        if pipe_centroid:
            cv2.circle(frame, pipe_centroid, 8, (0, 0, 255), -1)
            
            center = (w // 2, h // 2)
            cv2.line(frame, center, pipe_centroid, (255, 0, 255), 2)
            
            # Status simple
            in_roi = self.is_centroid_in_roi(pipe_centroid, (h, w, 3))
            roi_color = (0, 255, 0) if in_roi else (0, 0, 255)
            cv2.circle(frame, pipe_centroid, 12, roi_color, 2)
        
        # Centro del frame
        center = (w // 2, h // 2)
        cv2.drawMarker(frame, center, (255, 255, 0), cv2.MARKER_CROSS, 15, 2)
        
        return pipe_centroid, pipe_mask, frame

    # === THREAD: COMANDOS RC ===
    def rc_command_worker(self):
        """Thread dedicado para enviar comandos RC con rate limiting"""
        print('🎮 Thread de comandos RC iniciado')
        
        while not self.stop_threads.is_set():
            try:
                # Obtener comando con timeout
                try:
                    lr, fw, ud, yaw = self.rc_command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_rc_time < self.rc_min_interval:
                    time.sleep(self.rc_min_interval)
                
                # Enviar solo si cambió
                if (lr, fw, ud, yaw) != self.last_rc_command:
                    self.tello.send_rc_control(lr, fw, ud, yaw)
                    self.last_rc_command = (lr, fw, ud, yaw)
                    self.last_rc_time = time.time()
                
            except Exception as e:
                print(f'❌ Error enviando comando RC: {e}')
                time.sleep(0.1)
        
        # Detener dron al finalizar
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except:
            pass
        
        print('🎮 Thread de comandos RC finalizado')
    
    def send_rc_command(self, lr, fw, ud, yaw):
        """Encolar comando RC (thread-safe)"""
        try:
            # Si la cola está llena, sacar el comando más viejo
            if self.rc_command_queue.full():
                try:
                    self.rc_command_queue.get_nowait()
                except queue.Empty:
                    pass
            self.rc_command_queue.put_nowait((lr, fw, ud, yaw))
        except queue.Full:
            pass

    def update_roi_for_state(self, state):
        """Actualizar ROI según el estado del dron"""
        if state == DroneState.FOLLOWING:
            # ROI estrecho para seguimiento preciso sin ruido
            self.current_roi = self.roi_narrow.copy()
        else:
            # ROI amplio para búsqueda y exploración
            self.current_roi = self.roi_wide.copy()
    
    # === UTILIDADES ===
    def calculate_centroid(self, mask):
        """Calcular centroide"""
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        return None
    
    def is_centroid_in_roi(self, centroid, frame_shape):
        """Verificar si centroide está en ROI"""
        if centroid is None:
            return False
        
        h, w = frame_shape[:2]
        cx, cy = centroid
        
        roi_top_px = int(h * self.current_roi['top'])
        roi_bottom_px = int(h * self.current_roi['bottom'])
        roi_left_px = int(w * self.current_roi['left'])
        roi_right_px = int(w * self.current_roi['right'])
    
        
        return (roi_left_px <= cx <= roi_right_px and
                roi_top_px <= cy <= roi_bottom_px)
    
    def calculate_control_commands(self, centroid, frame_shape):
        """Control PD"""
        h, w = frame_shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt < 0.001:
            dt = 0.001
        
        if centroid is None:
            self.prev_error_x = 0
            self.prev_error_y = 0
            self.error_x_filtered = 0
            self.error_y_filtered = 0
            self.prev_time = current_time
            return 0, 0, 0, 0
        
        cx, cy = centroid
        error_x = cx - center_x
        error_y = center_y - cy
        
        self.error_x_filtered = (self.filter_alpha * self.error_x_filtered +
                                 (1 - self.filter_alpha) * error_x)
        self.error_y_filtered = (self.filter_alpha * self.error_y_filtered +
                                 (1 - self.filter_alpha) * error_y)
        
        error_x = self.error_x_filtered
        error_y = self.error_y_filtered
        
        if abs(error_x) < self.deadzone_x:
            error_x = 0
        if abs(error_y) < self.deadzone_y:
            error_y = 0
        
        d_error_x = (error_x - self.prev_error_x) / dt
        d_error_y = (error_y - self.prev_error_y) / dt
        
        left_right = int(self.kp_x * error_x + self.kd_x * d_error_x)
        up_down = int(self.kp_y * error_y + self.kd_y * d_error_y)
        
        left_right = int(np.clip(left_right, -self.max_speed, self.max_speed))
        up_down = int(np.clip(up_down, -self.max_speed, self.max_speed))
        
        forward = int(self.forward_speed)
        yaw = 0
        
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = current_time
        
        return left_right, up_down, forward, yaw
    
    def explore_for_next_pipe(self):
        """Ejecutar secuencia de exploración"""
        if self.exploration_step >= len(self.exploration_sequence):
            return True
        
        step_name, lr, fw, ud, yaw, duration = self.exploration_sequence[self.exploration_step]
        
        if self.exploration_start_time is None:
            self.exploration_start_time = time.time()
            print(f'   🔍 Explorando: {step_name} ({duration}s)')
        
        elapsed = time.time() - self.exploration_start_time
        
        # Durante pasos de escaneo, detectar pipes
        if "scan" in step_name:
            _, _, _, fwd, left, right = self.detection_result.get()
            
            if "left" in step_name:
                self.detections_left.append(left)
            elif "forward" in step_name:
                self.detections_forward.append(fwd)
            elif "right" in step_name:
                self.detections_right.append(right)
        
        if elapsed < duration:
            self.send_rc_command(lr, fw, ud, yaw)
        else:
            self.exploration_step += 1
            self.exploration_start_time = None
            
            if self.exploration_step >= len(self.exploration_sequence):
                return True
        
        return False
    
    def decide_exploration_direction(self):
        """Decidir dirección basada en detecciones"""
        avg_left = np.mean(self.detections_left) if self.detections_left else 0
        avg_forward = np.mean(self.detections_forward) if self.detections_forward else 0
        avg_right = np.mean(self.detections_right) if self.detections_right else 0
        
        print(f'\n📊 Detecciones promedio:')
        print(f'   ⬅️  Izquierda: {avg_left:.1f}')
        print(f'   ⬆️  Adelante: {avg_forward:.1f}')
        print(f'   ➡️  Derecha: {avg_right:.1f}')
        
        threshold = 0.4
        
        if avg_left >= threshold and avg_left >= avg_forward and avg_left >= avg_right:
            return ExplorationDirection.LEFT
        elif avg_forward >= threshold and avg_forward >= avg_left and avg_forward >= avg_right:
            return ExplorationDirection.FORWARD
        elif avg_right >= threshold:
            return ExplorationDirection.RIGHT
        else:
            print('   ⚠️  No hay detección clara, intentando adelante')
            return ExplorationDirection.FORWARD
    
    def execute_maneuver(self):
        """Ejecutar maniobra según dirección"""
        if self.maneuver_step >= len(self.current_maneuver):
            return True
        
        lr, fw, ud, yaw, duration = self.current_maneuver[self.maneuver_step]
        
        if self.maneuver_start_time is None:
            self.maneuver_start_time = time.time()
            print(f'   📍 Paso {self.maneuver_step + 1}/{len(self.current_maneuver)}: LR={lr} FW={fw} UD={ud} YAW={yaw}')
        
        elapsed = time.time() - self.maneuver_start_time
        
        if elapsed >= duration:
            self.maneuver_step += 1
            self.maneuver_start_time = None
            
            if self.maneuver_step >= len(self.current_maneuver):
                print('   ✅ Maniobra completada')
                return True
        else:
            self.send_rc_command(lr, fw, ud, yaw)
        
        return False
    
    def draw_overlay(self, frame, centroid, left_right, up_down, forward):
        """Dibujar overlay con información (OPTIMIZADO)"""
        if frame is None:
            return frame
        
        h, w = frame.shape[:2]
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = 30 / (time.time() - self.fps_time)
            self.fps_time = time.time()
        
        # Consultar batería solo cada 60 frames en lugar de cada frame
        if self.frame_count % 60 == 0:
            try:
                self.cached_battery = self.tello.get_battery()
            except:
                if not hasattr(self, 'cached_battery'):
                    self.cached_battery = 0
        
        if hasattr(self, 'cached_battery'):
            info_text = f'FPS: {self.fps:.1f} | Bat: {self.cached_battery}%'
        else:
            info_text = f'FPS: {self.fps:.1f}'
        
        cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        state_colors = {
            DroneState.GROUNDED: (255, 255, 255),
            DroneState.INITIAL_ADVANCE: (0, 255, 255),
            DroneState.SEARCHING: (0, 255, 255),
            DroneState.FOLLOWING: (0, 255, 0),
            DroneState.EXPLORING: (255, 0, 255),
            DroneState.MANEUVER: (255, 165, 0),
            DroneState.LANDING: (0, 0, 255),
        }
        
        with self.state_lock:
            current_state = self.state
        
        state_text = f'ESTADO: {current_state.value}'
        cv2.putText(frame, state_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colors.get(current_state, (255, 255, 255)), 2)
        
        pipes_text = f'Pipes: {self.pipes_completed}/{self.max_pipes}'
        cv2.putText(frame, pipes_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if current_state == DroneState.EXPLORING:
            direction_text = f'Dir: {self.exploration_direction.value}'
            cv2.putText(frame, direction_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        elif current_state == DroneState.MANEUVER:
            step_text = f'Paso {self.maneuver_step + 1}/{len(self.current_maneuver)}'
            cv2.putText(frame, step_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        cv2.putText(frame, 'T=Despegar | L=Aterrizar | Q=Salir',
                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def takeoff(self):
        """Despegar"""
        if not self.flying:
            print('\n🚁 Despegando...')
            try:
                self.tello.takeoff()
                time.sleep(3)
                
                self.flying = True
                with self.state_lock:
                    self.state = DroneState.INITIAL_ADVANCE
                self.initial_advance_time = time.time()
                print('✅ En el aire - Iniciando avance inicial...')
                return True
            except Exception as e:
                print(f'❌ Error: {e}')
                return False
        return True
    
    def land(self):
        """Aterrizar"""
        if self.flying:
            print('\n🛬 Aterrizando...')
            try:
                self.send_rc_command(0, 0, 0, 0)
                time.sleep(0.5)
                self.tello.land()
                time.sleep(2)
                self.flying = False
                with self.state_lock:
                    self.state = DroneState.GROUNDED
                print('✅ Aterrizado')
                return True
            except Exception as e:
                print(f'❌ Error: {e}')
                return False
        return True
    
    def start_threads(self):
        """Iniciar todos los threads"""
        print('\n🔧 Iniciando threads...')
        
        if self.frame_read is None:
            print('❌ Error: Stream no inicializado correctamente')
            return False
        
        self.stop_threads.clear()
        
        # Thread de captura
        self.capture_thread = Thread(target=self.frame_capture_worker, daemon=True)
        self.capture_thread.start()
        
        # Esperar a que haya frames
        print('   ⏳ Esperando frames...')
        timeout = time.time() + 10  # Aumentado a 10 segundos
        frames_ok = False
        
        while time.time() < timeout:
            frame = self.frame_buffer.get()
            if frame is not None:
                print(f'   ✅ Frames disponibles ({frame.shape})')
                frames_ok = True
                break
            time.sleep(0.2)
        
        if not frames_ok:
            print('   ⚠️  Timeout esperando frames, pero continuando...')
        
        # Thread de detección
        self.detection_thread = Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        
        # Thread de comandos RC
        self.rc_thread = Thread(target=self.rc_command_worker, daemon=True)
        self.rc_thread.start()
        
        time.sleep(1)
        print('✅ Threads iniciados')
        return True
    
    def stop_all_threads(self):
        """Detener todos los threads"""
        print('\n🛑 Deteniendo threads...')
        self.stop_threads.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        
        if self.rc_thread and self.rc_thread.is_alive():
            self.rc_thread.join(timeout=2)
        
        print('✅ Threads detenidos')
    
    def run(self):
        """Loop principal - Máquina de estados"""
        print('\n' + '='*70)
        print('🎯 LINE FOLLOWER AUTÓNOMO CON EXPLORACIÓN + MULTITHREADING')
        print('='*70)
        print('\n🔄 Funcionamiento:')
        print('  1. Sigue pipe dentro del ROI')
        print('  2. Pipe sale del ROI → EXPLORA')
        print('  3. Detecta dirección del siguiente pipe')
        print('  4. Ejecuta maniobra según dirección')
        print(f'\n🎯 Objetivo: Completar {self.max_pipes} pipes')
        print('\n🔧 Modo optimizado - procesamiento reducido')
        print('='*70 + '\n')
        
        input('Presiona ENTER para comenzar...')
        
        # Iniciar threads
        if not self.start_threads():
            print("❌ Error iniciando threads")
            return
        
        # Esperar a que haya datos válidos de detección
        print('⏳ Esperando frames y detecciones...')
        timeout = time.time() + 15

        while not self.detection_result.has_valid_data() and time.time() < timeout:
            time.sleep(0.3)
        
        if not self.detection_result.has_valid_data():
            print('❌ No se recibieron detecciones válidas')
            return

        print('✅ Sistema listo, iniciando control...')
        
        # Contador para verificación de stream
        stream_check_counter = 0
        stream_check_interval = 150
        last_recovery_attempt = 0
        
        while True:
            try:
                # ===== VERIFICAR SALUD DEL STREAM =====
                stream_check_counter += 1
                if stream_check_counter >= stream_check_interval:
                    stream_check_counter = 0
                    
                    time_since_frame = time.time() - self.last_frame_time
                    if time_since_frame > self.stream_timeout:
                        self.stream_frozen = True
                    
                    if self.stream_frozen and not self.flying:
                        if time.time() - last_recovery_attempt > 15:
                            last_recovery_attempt = time.time()
                            self.attempt_stream_recovery()
                
                # Obtener último resultado de detección
                centroid, mask, annotated_frame, pipes_fwd, pipes_left, pipes_right = \
                    self.detection_result.get()
                
                # Si no hay frame aún, esperar
                if annotated_frame is None:
                    time.sleep(0.05)
                    continue
                
                h, w = annotated_frame.shape[:2]
                
                # Calcular comandos de control
                left_right, up_down, forward, yaw = self.calculate_control_commands(
                    centroid, (h, w, 3)
                )
                
                # === MÁQUINA DE ESTADOS ===
                with self.state_lock:
                    current_state = self.state
                
                # ===== ACTUALIZAR ROI SEGÚN ESTADO =====
                self.update_roi_for_state(current_state)
                
                if self.flying:
                    
                    if current_state == DroneState.INITIAL_ADVANCE:
                        elapsed = time.time() - self.initial_advance_time
                        
                        if elapsed < self.initial_advance_duration:
                            remaining = self.initial_advance_duration - elapsed
                            if int(elapsed * 2) != int((elapsed - 0.05) * 2):
                                print(f'   🚀 Avanzando... ({remaining:.1f}s restantes)')
                            self.send_rc_command(0, 15, 0, 0)
                        else:
                            print('✅ Avance inicial completado - Buscando pipe en ROI...')
                            self.send_rc_command(0, 0, 0, 0)
                            time.sleep(0.5)
                            with self.state_lock:
                                self.state = DroneState.SEARCHING
                    
                    elif current_state == DroneState.SEARCHING:
                        if centroid and self.is_centroid_in_roi(centroid, (h, w, 3)):
                            print(f'\n🟢 PIPE #{self.pipes_completed + 1} DETECTADO - Iniciando seguimiento')
                            with self.state_lock:
                                self.state = DroneState.FOLLOWING
                            self.frames_without_line = 0
                        else:
                            self.send_rc_command(0, 0, 0, 0)
                    
                    elif current_state == DroneState.FOLLOWING:
                        if centroid:
                            if self.is_centroid_in_roi(centroid, (h, w, 3)):
                                self.send_rc_command(left_right, forward, up_down, yaw)
                                self.frames_without_line = 0
                            else:
                                print(f'\n🏁 PIPE #{self.pipes_completed + 1} COMPLETADO!')
                                self.pipes_completed += 1
                                
                                if self.pipes_completed >= self.max_pipes:
                                    print(f'\n🎉 ¡MISIÓN COMPLETA! {self.pipes_completed} pipes completados')
                                    with self.state_lock:
                                        self.state = DroneState.LANDING
                                else:
                                    print(f'\n🔍 Iniciando EXPLORACIÓN para encontrar pipe #{self.pipes_completed + 1}...')
                                    with self.state_lock:
                                        self.state = DroneState.EXPLORING
                                    self.exploration_step = 0
                                    self.exploration_start_time = None
                                    self.exploration_direction = ExplorationDirection.NONE
                                    self.detections_left = []
                                    self.detections_forward = []
                                    self.detections_right = []
                        else:
                            self.frames_without_line += 1
                            
                            if self.frames_without_line < 10:
                                if self.frames_without_line % 5 == 0:
                                    print(f'\n⚠️  Pipe perdido - Continuando ({self.frames_without_line}/10)')
                                self.send_rc_command(0, 8, 0, 0)
                            elif self.frames_without_line < self.max_frames_without_line:
                                if self.frames_without_line % 10 == 0:
                                    print(f'\n⚠️  Esperando ({self.frames_without_line}/{self.max_frames_without_line})')
                                self.send_rc_command(0, 0, 0, 0)
                            else:
                                print('\n🛬 ATERRIZAJE - Pipe perdido')
                                with self.state_lock:
                                    self.state = DroneState.LANDING
                    
                    elif current_state == DroneState.EXPLORING:
                        exploration_complete = self.explore_for_next_pipe()
                        
                        if exploration_complete:
                            self.exploration_direction = self.decide_exploration_direction()
                            
                            print(f'\n✅ Exploración completada')
                            print(f'   🎯 Dirección elegida: {self.exploration_direction.value}')
                            
                            if self.exploration_direction == ExplorationDirection.NONE:
                                print('\n⚠️  No se encontró pipe, aterrizando')
                                with self.state_lock:
                                    self.state = DroneState.LANDING
                            else:
                                self.current_maneuver = self.maneuver_sequences[self.exploration_direction]
                                self.maneuver_step = 0
                                self.maneuver_start_time = None
                                
                                print(f'\n🔄 Ejecutando maniobra hacia {self.exploration_direction.value}...')
                                with self.state_lock:
                                    self.state = DroneState.MANEUVER
                    
                    elif current_state == DroneState.MANEUVER:
                        maneuver_complete = self.execute_maneuver()
                        
                        if maneuver_complete:
                            print(f'\n🔍 Buscando pipe #{self.pipes_completed + 1}...')
                            with self.state_lock:
                                self.state = DroneState.SEARCHING
                            self.maneuver_step = 0
                            self.maneuver_start_time = None
                    
                    elif current_state == DroneState.LANDING:
                        self.send_rc_command(0, 0, 0, 0)
                        time.sleep(0.5)
                        self.land()
                        print('\n✅ Sistema detenido. Presiona Q para salir.')
                        with self.state_lock:
                            self.state = DroneState.GROUNDED
                
                # Dibujar overlay
                display_frame = self.draw_overlay(
                    annotated_frame, centroid, left_right, up_down, forward
                )
                
                # Mostrar frame principal (ÚNICA VENTANA)
                if display_frame is not None:
                    cv2.imshow('Line Follower', display_frame)
                
                # Teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print('\n👋 Cerrando...')
                    break
                elif key == ord('t'):
                    if not self.flying and current_state == DroneState.GROUNDED:
                        self.takeoff()
                elif key == ord('l'):
                    if self.flying:
                        print('\n🛬 ATERRIZAJE MANUAL')
                        self.send_rc_command(0, 0, 0, 0)
                        self.land()
                
                # Mostrar mensaje final si completó la misión
                if current_state == DroneState.GROUNDED and not self.flying and self.pipes_completed > 0:
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
                    display_frame = cv2.addWeighted(display_frame, 0.3, overlay, 0.7, 0)
                    
                    cv2.putText(display_frame, 'MISION COMPLETADA!',
                            (w//4 + 20, h//2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.putText(display_frame, f'{self.pipes_completed} pipes completados',
                            (w//4 + 40, h//2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display_frame, 'Presiona Q para salir',
                            (w//4 + 60, h//2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow('Line Follower', display_frame)
                
                time.sleep(0.03)  # 50 FPS max en el loop principal
                
            except Exception as e:
                print(f'❌ Error en loop principal: {e}')
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        
        if self.flying:
            self.land()
    
    def stop_stream(self):
        """Detener stream"""
        print('\n📹 Deteniendo stream...')
        try:
            if self.frame_read is not None:
                self.frame_read.stop()
            self.tello.streamoff()
            time.sleep(1)
        except Exception as e:
            print(f'⚠️  Error deteniendo stream: {e}')

    
    def disconnect(self):
        """Desconectar"""
        print('🔌 Desconectando...')
        try:
            self.tello.end()
        except:
            pass


def main():
    print('='*70)
    print('🤖 LINE FOLLOWER AUTÓNOMO + MULTITHREADING')
    print('='*70)
    
    MODEL_PATH = 'best.pt'
    TARGET_CLASS = 'Pipes'
    
    print(f'\n⚙️  Configuración:')
    print(f'   📦 Modelo: {MODEL_PATH}')
    print(f'   🎯 Clase: "{TARGET_CLASS}"')
    print(f'   🧵 Threads: Captura | Detección | RC Control')
    print(f'   🔍 Sistema: Exploración automática\n')
    
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
        # Detener threads
        follower.stop_all_threads()
        
        if follower.flying:
            follower.land()
        
        follower.stop_stream()
        follower.disconnect()
        
        print('\n' + '='*70)
        print('✅ Line Follower finalizado')
        print('='*70)


if __name__ == '__main__':
    main()