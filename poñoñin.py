import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time
import threading
from collections import deque, Counter

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODEL_PATH = 'best.pt'
KP = 0.25
KD = 0.2
DEADZONE = 60
MAX_SPEED = 18
CORRECTION_TIME = 10.0
INFERENCE_SIZE = 224
INFERENCE_SIZE_LIGHT = 160

# ROI
ROI_LEFT = 80
ROI_RIGHT = 900
ROI_TOP = 0
ROI_BOTTOM = 480

# Debug (opcional)
DEBUG_DETECTIONS = False  # Cambiar a True para ver más info

# ============================================================
# SETUP
# ============================================================
print("Conectando...")
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")

tello.streamon()
time.sleep(2)

model = YOLO(MODEL_PATH)
frame_read = tello.get_frame_read()

# Variables globales
latest_frame = None
cached_frame = None
cached_frame_time = 0
CACHE_DURATION = 0.05  # Cache de 50ms para evitar lecturas redundantes

display_running = True
in_correction_mode = False
segment_counter = 0
wpX = []
wpY = []

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def get_frame(use_cache=True):
    """Obtiene y transforma el frame actual con cache opcional"""
    global cached_frame, cached_frame_time
    
    current_time = time.time()
    
    # Usar cache si es reciente
    if use_cache and cached_frame is not None and (current_time - cached_frame_time) < CACHE_DURATION:
        return cached_frame
    
    # Leer nuevo frame
    frame = frame_read.frame
    if frame is None:
        return cached_frame if cached_frame is not None else None
    
    # Transformar y cachear
    processed_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0)
    cached_frame = processed_frame
    cached_frame_time = current_time
    
    return processed_frame

def in_roi(cx, cy):
    """Verifica si un punto está dentro del ROI"""
    return (ROI_LEFT <= cx <= ROI_RIGHT and ROI_TOP <= cy <= ROI_BOTTOM)

def get_mask_centroid(mask, w, h):
    """Calcula el centroide de una máscara"""
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 0.5).astype(np.uint8)
    M = cv2.moments(mask_bin)
    if M["m00"] > 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return None, None

# ============================================================
# THREAD DE VISUALIZACIÓN (OPTIMIZADO)
# ============================================================
def display_thread():
    global latest_frame, display_running, segment_counter
    
    skip_counter = 0
    
    while display_running:
        frame = get_frame(use_cache=True)
        if frame is None:
            time.sleep(0.02)
            continue
        
        skip_counter += 1
        # Más agresivo en saltar frames cuando no está en modo corrección
        skip_frames = 2 if in_correction_mode else 6
        
        if skip_counter % skip_frames != 0:
            time.sleep(0.01)
            continue
        
        h, w = frame.shape[:2]
        center_x = w // 2
        
        # Inferencia con tamaño adaptativo
        inference_size = INFERENCE_SIZE if in_correction_mode else INFERENCE_SIZE_LIGHT
        results = model(frame, conf=0.3, verbose=False, imgsz=inference_size)
        
        # Dibujar ROI
        cv2.line(frame, (ROI_LEFT, 0), (ROI_LEFT, h), (0, 0, 255), 2)
        cv2.line(frame, (ROI_RIGHT, 0), (ROI_RIGHT, h), (0, 0, 255), 2)
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 0), 1)
        
        # Procesar detecciones
        if results and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                class_name = result.names[cls_id]
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2
                
                color = (0, 255, 0) if in_roi(cx_box, cy_box) else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name}:{conf:.2f}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Procesar máscaras solo para pipes (no fire)
            if result.masks is not None:
                for i, cls_id in enumerate(classes):
                    class_name = result.names[cls_id].lower()
                    if class_name in ['pipes', 'pipe', 'fire']:
                        mask = result.masks.data[i].cpu().numpy()
                        cx, cy = get_mask_centroid(mask, w, h)
                        
                        if cx is not None:
                            # Color diferente para fuego
                            if class_name == 'fire':
                                color = (0, 0, 255) if in_roi(cx, cy) else (100, 0, 0)  # Rojo para fuego
                            else:
                                color = (0, 255, 255) if in_roi(cx, cy) else (100, 100, 100)
                            
                            cv2.circle(frame, (cx, cy), 8, color, -1)
                            
                            # Solo mostrar error para pipes
                            if class_name in ['pipes', 'pipe']:
                                error = cx - center_x
                                cv2.line(frame, (center_x, cy), (cx, cy), color, 2)
                                cv2.putText(frame, f"e:{error}", (cx+15, cy), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info en pantalla
        cv2.putText(frame, f"Seg:{segment_counter}/26", (10, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        mode_text = "CORRIGIENDO" if in_correction_mode else "NAVEGANDO"
        cv2.putText(frame, mode_text, (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        latest_frame = frame.copy()
        cv2.imshow("Tello", frame)
        cv2.waitKey(1)
        time.sleep(0.015)

threading.Thread(target=display_thread, daemon=True).start()

# ============================================================
# NAVEGACIÓN
# ============================================================
def move_and_stabilize(distance=60):
    """Mueve el drone y estabiliza"""
    global in_correction_mode
    in_correction_mode = False
    tello.move_forward(distance)
    time.sleep(1.0)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1.0)

def rotate_left():
    """Rota 90° a la izquierda"""
    global in_correction_mode
    in_correction_mode = False
    tello.rotate_counter_clockwise(90)
    time.sleep(1.0)

# ============================================================
# CONTEO DE PIPES
# ============================================================
def count_pipes(samples=5):
    """Cuenta pipes en ROI usando múltiples muestras"""
    counts = []
    
    for _ in range(samples):
        frame = get_frame(use_cache=False)  # No cache para muestras independientes
        if frame is None:
            time.sleep(0.1)
            continue
        
        h, w = frame.shape[:2]
        results = model(frame, conf=0.3, verbose=False, imgsz=INFERENCE_SIZE)
        pipes_in_roi = 0
        
        if results and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, cls_id in enumerate(classes):
                class_name = result.names[cls_id].lower()
                # SOLO contar pipes, NO fuego
                if class_name in ['pipes', 'pipe']:
                    mask = result.masks.data[i].cpu().numpy()
                    cx, cy = get_mask_centroid(mask, w, h)
                    if cx is not None and in_roi(cx, cy):
                        pipes_in_roi += 1
        
        counts.append(pipes_in_roi)
        time.sleep(0.1)
    
    return Counter(counts).most_common(1)[0][0] if counts else 0

# ============================================================
# CORRECCIÓN PD (SOLO PARA PIPES)
# ============================================================
def correction_phase():
    """Control PD para centrar SOLO en pipes (NO en fuego)"""
    global in_correction_mode
    in_correction_mode = True
    
    print(f"    → Control PD en PIPES ({CORRECTION_TIME}s)")
    
    prev_error = 0
    prev_time = time.time()
    start_time = time.time()
    error_buffer = deque(maxlen=4)
    last_control = 0
    corrections_made = 0
    no_detection_count = 0
    
    while time.time() - start_time < CORRECTION_TIME:
        frame = get_frame(use_cache=False)  # Frame fresco para control
        if frame is None:
            time.sleep(0.02)
            continue
        
        h, w = frame.shape[:2]
        center_x = w // 2
        current_time = time.time()
        dt = max(current_time - prev_time, 0.001)
        
        results = model(frame, conf=0.3, verbose=False, imgsz=INFERENCE_SIZE)
        
        pipe_found = False
        
        # Debug: mostrar clases detectadas
        if DEBUG_DETECTIONS and results and results[0].boxes is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            detected_classes = [result.names[c] for c in classes]
            if detected_classes:
                print(f"      DEBUG: Clases detectadas: {set(detected_classes)}")
        
        if results and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            best_idx = None
            best_conf = 0
            candidates_info = []
            
            for i, cls_id in enumerate(classes):
                class_name = result.names[cls_id].lower()
                
                # ⚠️ SOLO buscar pipes, NO fire
                if class_name in ['pipes', 'pipe']:
                    mask = result.masks.data[i].cpu().numpy()
                    cx, cy = get_mask_centroid(mask, w, h)
                    
                    if cx is not None:
                        conf = confidences[i]
                        in_roi_flag = in_roi(cx, cy)
                        
                        if DEBUG_DETECTIONS:
                            candidates_info.append(f"{class_name}@({cx},{cy}) ROI:{in_roi_flag} conf:{conf:.2f}")
                        
                        if in_roi_flag and conf > best_conf:
                            best_idx = i
                            best_conf = conf
            
            # Debug: mostrar candidatos
            if DEBUG_DETECTIONS and candidates_info:
                print(f"      Candidatos: {' | '.join(candidates_info)}")
            
            if best_idx is not None:
                pipe_found = True
                no_detection_count = 0
                
                mask = result.masks.data[best_idx].cpu().numpy()
                cx, cy = get_mask_centroid(mask, w, h)
                
                if cx is not None:
                    error = cx - center_x
                    
                    # Filtrar error con buffer
                    error_buffer.append(error)
                    error_filtered = int(np.mean(error_buffer))
                    
                    if abs(error_filtered) < DEADZONE:
                        control = int(np.clip(error_filtered * 0.08, -8, 8))
                    else:
                        p_term = KP * error_filtered
                        d_term = KD * (error_filtered - prev_error) / dt
                        control = p_term + d_term
                        
                        # Limitar cambios bruscos
                        control_change = abs(control - last_control)
                        if control_change > 10:
                            control = last_control + np.sign(control - last_control) * 10
                        
                        control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))
                        corrections_made += 1
                    
                    tello.send_rc_control(control, 0, 0, 0)
                    prev_error = error_filtered
                    last_control = control
                    
                    # Feedback cada 2 segundos
                    elapsed = int(time.time() - start_time)
                    if elapsed > 0 and elapsed % 2 == 0 and corrections_made > 0:
                        print(f"      ✓ error:{error_filtered:4d} | ctrl:{control:3d}")
                        corrections_made = 0
        
        if not pipe_found:
            no_detection_count += 1
            if DEBUG_DETECTIONS and no_detection_count % 20 == 0:
                print(f"      ⚠ Sin pipe en ROI ({no_detection_count} frames)")
            tello.send_rc_control(0, 0, 0, 0)
            error_buffer.clear()
        
        prev_time = current_time
        time.sleep(0.05)
    
    tello.send_rc_control(0, 0, 0, 0)
    print("    ✓ Corrección en pipes completa")
    in_correction_mode = False

# ============================================================
# DETECCIÓN DE FUEGO
# ============================================================
def detect_fire(segment_idx, prev_had_fire):
    """Detecta fuego y registra waypoint (SIN centrado)"""
    frame = get_frame(use_cache=True)
    if frame is None:
        return False, False
    
    results = model(frame, conf=0.4, verbose=False, imgsz=160)
    
    fire_detected = False
    if results and results[0].boxes is not None:
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        fire_detected = any(results[0].names[c].lower() == 'fire' for c in classes)
    
    if fire_detected and not prev_had_fire:
        wpX.append(segment_idx)
        wpY.append(0)
        print(f"    🔥 Fuego detectado en seg {segment_idx} (sin centrado)")
    
    return fire_detected, fire_detected

# ============================================================
# RUTA PRINCIPAL
# ============================================================
def run_route():
    global segment_counter
    
    segment_seen = False
    
    for vuelta in range(2):
        print(f"\n{'='*50}\nVUELTA {vuelta+1}/2\n{'='*50}")
        
        # Lado largo (6 segmentos)
        for j in range(6):
            segment_counter += 1
            print(f"\nSeg {segment_counter}")
            
            move_and_stabilize()
            _, segment_seen = detect_fire(segment_counter, segment_seen)
            
            if j < 5:  # Corrección excepto último segmento
                pipes = count_pipes(samples=5)
                print(f"    📊 Pipes detectados: {pipes}")
                if 0 < pipes <= 2:
                    correction_phase()
        
        rotate_left()
        
        # Lado corto (7 segmentos)
        for j in range(7):
            segment_counter += 1
            print(f"\nSeg {segment_counter}")
            
            move_and_stabilize()
            _, segment_seen = detect_fire(segment_counter, segment_seen)
            
            if j < 6:  # Corrección excepto último segmento
                pipes = count_pipes(samples=5)
                print(f"    📊 Pipes detectados: {pipes}")
                if 0 < pipes <= 2:
                    correction_phase()
        
        rotate_left()
    
    print("\n✅ RUTA COMPLETADA")
    print(f"Fuegos detectados: {len(wpX)} en segmentos {wpX}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        input("Presiona ENTER para despegar...")
        
        tello.takeoff()
        time.sleep(3)
        
        run_route()

    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        display_running = False
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()