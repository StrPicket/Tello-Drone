# ============================================================
# P O Ñ O Ñ I N   V O L A D  O R
# ============================================================
import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time
import threading
from collections import deque, Counter
import csv

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODEL_PATH = 'best.pt'
KP = 0.25
KD = 0.2
DEADZONE = 60
MAX_SPEED = 18
CORRECTION_TIME = 5.0
INFERENCE_SIZE = 224

# Control de YAW
KP_YAW = 1.12
YAW_DEADZONE = 25
MAX_YAW_SPEED = 30

# ROI para pipes
ROI_LEFT = 10
ROI_RIGHT = 1000
ROI_TOP = 0
ROI_BOTTOM = 480

# ROI exclusivo para fuegos
FIRE_ROI_LEFT = 300
FIRE_ROI_RIGHT = 680

# Límite vertical
HEIGHT_LIMIT_RATIO = 0.15

# Colores BGR
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
]

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
display_running = True
in_correction_mode = False
segment_counter = 0
wp = []
wpX = []
wpY = []

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def get_frame():
    """Obtiene frame con flip vertical"""
    frame = frame_read.frame
    if frame is None:
        return None
    return cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0)

def in_roi(cx, cy):
    return (ROI_LEFT <= cx <= ROI_RIGHT and ROI_TOP <= cy <= ROI_BOTTOM)

def in_fire_roi(cx):
    """ROI exclusivo para detección de fuegos"""
    return (FIRE_ROI_LEFT <= cx <= FIRE_ROI_RIGHT)

def get_mask_centroid(mask, w, h):
    mask_resized = cv2.resize(mask, (w, h))
    mask_bin = (mask_resized > 0.5).astype(np.uint8)
    M = cv2.moments(mask_bin)
    if M["m00"] > 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return None, None

def get_pipe_alignment_points(mask, w, h):
    """Extrae punto superior e inferior de la pipe para corrección de yaw"""
    mask_resized = cv2.resize(mask, (w, h))
    mask_bin = (mask_resized > 0.5).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    h_third = h // 3
    
    top_points = largest_contour[largest_contour[:, 0, 1] < h_third]
    if len(top_points) > 0:
        top_x = int(np.mean(top_points[:, 0, 0]))
        top_y = int(np.mean(top_points[:, 0, 1]))
    else:
        return None, None
    
    bottom_points = largest_contour[largest_contour[:, 0, 1] > 2*h_third]
    if len(bottom_points) > 0:
        bottom_x = int(np.mean(bottom_points[:, 0, 0]))
        bottom_y = int(np.mean(bottom_points[:, 0, 1]))
    else:
        return None, None
    
    return (top_x, top_y), (bottom_x, bottom_y)

def draw_segmentation(frame, results):
    """Dibuja las máscaras de segmentación"""
    if not results or len(results) == 0:
        return frame
    
    result = results[0]
    
    if result.masks is None or len(result.masks) == 0:
        return frame
    
    # Crear overlay
    overlay = frame.copy()
    
    boxes = result.boxes.xyxy.cpu().numpy()
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    
    h, w = frame.shape[:2]
    center_x = w // 2
    
    for i, (box, mask, cls, conf) in enumerate(zip(boxes, masks, classes, confidences)):
        class_name = result.names[cls]
        color = COLORS[cls % len(COLORS)]
        
        # Redimensionar máscara al tamaño del frame
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Crear máscara coloreada
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_binary == 1] = color
        
        # Aplicar overlay con transparencia
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
        
        # Dibujar contorno de la máscara
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f'{class_name} {conf:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        top_pt, bottom_pt = get_pipe_alignment_points(mask, w, h)
        if top_pt is not None and bottom_pt is not None:
            cv2.circle(overlay, top_pt, 6, (0, 255, 255), -1)
            cv2.circle(overlay, bottom_pt, 6, (0, 255, 255), -1)
            cv2.line(overlay, top_pt, bottom_pt, (0, 255, 255), 2)
            yaw_error = top_pt[0] - bottom_pt[0]
            cv2.putText(overlay, f"yaw:{yaw_error}", (cx+15, cy+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Centroide para pipes
        if class_name.lower() in ['pipes', 'pipe']:
            cx, cy = get_mask_centroid(mask, w, h)
            if cx is not None:
                cv2.circle(overlay, (cx, cy), 8, color, -1)
                error = cx - center_x
                cv2.line(overlay, (center_x, cy), (cx, cy), color, 2)
                cv2.putText(overlay, f"e:{error}", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay

# ============================================================
# THREAD DE VISUALIZACIÓN
# ============================================================
def display_thread():
    global display_running, segment_counter
    
    while display_running:
        frame = get_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        
        h, w = frame.shape[:2]
        center_x = w // 2
        
        # Inferencia
        results = model(frame, conf=0.5, verbose=False, imgsz=INFERENCE_SIZE)
        
        # Dibujar segmentación
        frame = draw_segmentation(frame, results)
        
        # ROI pipes
        cv2.line(frame, (ROI_LEFT, 0), (ROI_LEFT, h), (0, 0, 255), 2)
        cv2.line(frame, (ROI_RIGHT, 0), (ROI_RIGHT, h), (0, 0, 255), 2)
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 0), 1)
        
        # ROI fuegos (líneas naranjas)
        cv2.line(frame, (FIRE_ROI_LEFT, 0), (FIRE_ROI_LEFT, h), (0, 165, 255), 2)
        cv2.line(frame, (FIRE_ROI_RIGHT, 0), (FIRE_ROI_RIGHT, h), (0, 165, 255), 2)
        
        # Info
        cv2.putText(frame, f"Seg:{segment_counter}/26", (10, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        mode = "CORRIGIENDO" if in_correction_mode else "NAVEGANDO"
        cv2.putText(frame, mode, (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Tello", frame)
        cv2.waitKey(1)
        time.sleep(0.03)

threading.Thread(target=display_thread, daemon=True).start()

# ============================================================
# NAVEGACIÓN
# ============================================================
def move_and_stabilize(distance=60):
    global in_correction_mode
    in_correction_mode = False
    tello.move_forward(distance)
    time.sleep(1.0)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1.0)

def rotate_left():
    global in_correction_mode
    in_correction_mode = False
    tello.rotate_counter_clockwise(90)
    time.sleep(1.0)

# ============================================================
# CONTEO DE PIPES
# ============================================================
def count_pipes(samples=5):
    counts = []
    
    for _ in range(samples):
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        h, w = frame.shape[:2]
        results = model(frame, conf=0.5, verbose=False, imgsz=INFERENCE_SIZE)
        pipes_in_roi = 0
        
        if results and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, cls_id in enumerate(classes):
                class_name = result.names[cls_id].lower()
                if class_name in ['pipes', 'pipe']:
                    mask = result.masks.data[i].cpu().numpy()
                    cx, cy = get_mask_centroid(mask, w, h)
                    if cx is not None and in_roi(cx, cy):
                        pipes_in_roi += 1
        
        counts.append(pipes_in_roi)
        time.sleep(0.1)
    
    return Counter(counts).most_common(1)[0][0] if counts else 0

# ============================================================
# CORRECCIÓN PD
# ============================================================
def correction_phase():
    global in_correction_mode
    in_correction_mode = True
    
    print(f"    → Control PD ({CORRECTION_TIME}s)")
    
    prev_error = 0
    prev_time = time.time()
    start_time = time.time()
    error_buffer = deque(maxlen=4)
    yaw_buffer = deque(maxlen=3)
    last_control = 0
    
    while time.time() - start_time < CORRECTION_TIME:
        frame = get_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        
        h, w = frame.shape[:2]
        center_x = w // 2
        current_time = time.time()
        dt = max(current_time - prev_time, 0.001)
        
        results = model(frame, conf=0.5, verbose=False, imgsz=INFERENCE_SIZE)
        
        pipe_found = False
        
        if results and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            best_idx = None
            best_conf = 0
            
            for i, cls_id in enumerate(classes):
                class_name = result.names[cls_id].lower()
                
                if class_name in ['pipes', 'pipe']:
                    mask = result.masks.data[i].cpu().numpy()
                    cx, cy = get_mask_centroid(mask, w, h)
                    
                    if cx is not None:
                        conf = confidences[i]
                        
                        if in_roi(cx, cy) and conf > best_conf:
                            best_idx = i
                            best_conf = conf
            
            if best_idx is not None:
                pipe_found = True
                
                mask = result.masks.data[best_idx].cpu().numpy()
                cx, cy = get_mask_centroid(mask, w, h)
                # Control de yaw (rotación)
                top_pt, bottom_pt = get_pipe_alignment_points(mask, w, h)
                
                if cx is not None:
                    if top_pt is not None and bottom_pt is not None:
                        yaw_error = top_pt[0] - bottom_pt[0]
                        
                        yaw_buffer.append(yaw_error)
                        yaw_filtered = int(np.mean(yaw_buffer))
                                                  
                    error = cx - center_x
                    
                    error_buffer.append(error)
                    error_filtered = int(np.mean(error_buffer))
                    
                    if abs(error_filtered) < DEADZONE and abs(yaw_filtered) < YAW_DEADZONE:
                        control = int(np.clip(error_filtered * 0.08, -8, 8))

                    else:
                        p_term = KP * error_filtered
                        d_term = KD * (error_filtered - prev_error) / dt
                        control = p_term + d_term
                        
                        control_change = abs(control - last_control)
                        if control_change > 10:
                            control = last_control + np.sign(control - last_control) * 10
                        
                        yaw_control = -int(KP_YAW * yaw_filtered)
                        yaw_control = int(np.clip(yaw_control, -MAX_YAW_SPEED, MAX_YAW_SPEED))
                        control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))
                    
                    tello.send_rc_control(control, 0, 0, -yaw_control)
                    prev_error = error_filtered
                    last_control = control
        
        if not pipe_found:
            tello.send_rc_control(0, 0, 0, 0)
            error_buffer.clear()
            yaw_buffer.clear()
        
        prev_time = current_time
        time.sleep(0.05)
    
    tello.send_rc_control(0, 0, 0, 0)
    print("    ✓ Corrección completa")
    in_correction_mode = False

# ============================================================
# DETECCIÓN DE FUEGO
# ============================================================
def detect_fire(segment_idx, segment_seen_prev):
    frame = get_frame()
    if frame is None:
        return False
    
    h, w = frame.shape[:2]
    limit_line = int(h * HEIGHT_LIMIT_RATIO)
    
    results = model(frame, conf=0.6, verbose=False)
    
    if results and results[0].boxes is not None:
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        
        best_conf = 0
        best_center = None
        
        for box, cls_id, cf in zip(boxes, classes, confs):
            if result.names[cls_id].lower() == 'fire':
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Verificar que esté dentro del ROI de fuegos
                if not in_fire_roi(cx):
                    continue
                
                if cf > best_conf:
                    best_conf = cf
                    best_center = (cx, cy)
        
        if best_center is not None:
            cx, cy = best_center
            
            if segment_seen_prev:
                print(f"Detectado en segmento previo, recortando ROI, line: {limit_line}, fireY: {cy}")
                if cy < limit_line:
                    wp.append(segment_idx)
                    print(f"    🔥 Fuego seg {segment_idx}")
                    return True
                else:
                    return False
            else:
                wp.append(segment_idx)
                print(f"    🔥 Fuego seg {segment_idx}")
                return True
    
    return False

# ============================================================
# RUTA PRINCIPAL
# ============================================================
def run_route():
    global segment_counter
    segment_seen_prev = False 
    
    for vuelta in range(2):
        print(f"\n{'='*50}\nVUELTA {vuelta+1}/2\n{'='*50}")
        
        for j in range(6):
            segment_counter += 1
            print(f"\nSeg {segment_counter}")
            
            move_and_stabilize()
            start = time.time()

            segment_seen = False  # Inicializar antes del while
            while (time.time() - start) < 2.5:
                if detect_fire(segment_counter, segment_seen_prev):
                    segment_seen = True  # Acumular detecciones
            
            if j < 5:
                pipes = count_pipes(samples=5)
                print(f"    📊 Pipes: {pipes}")
                if 0 < pipes <= 2:
                    correction_phase()

            segment_seen_prev = segment_seen
        
        rotate_left()
        
        for j in range(7):
            segment_counter += 1
            print(f"\nSeg {segment_counter}")
            
            move_and_stabilize()
            start = time.time()
            
            segment_seen = False  # Inicializar antes del while
            while (time.time() - start) < 2.5:
                if detect_fire(segment_counter, segment_seen_prev):
                    segment_seen = True  # Acumular detecciones
            
            if j < 6:
                pipes = count_pipes(samples=5)
                print(f"    📊 Pipes: {pipes}")
                if 0 < pipes <= 2:
                    correction_phase()

            segment_seen_prev = segment_seen
        
        rotate_left()
        
    wp_unique = list(dict.fromkeys(wp))
    for i in wp_unique:
        if i <= 6:
            wpX.append(i)
            wpY.append(0)
        elif i <= 13:
            wpX.append(6)
            wpY.append(i-6)
        elif i <= 19:
            wpX.append(19-i)
            wpY.append(7)
        else:
            wpX.append(0)
            wpY.append(27-i)
    
    print("\n✅ COMPLETADO")
    print(f"Fuegos: {len(wpX)}")
    for i in range(len(wpX)):
        print(f"({wpX[i]},{wpY[i]})")

    with open('waypoints.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wpX","wpY"])
        for x,y in zip(wpX, wpY):
            writer.writerow([x,y])
    
    print("waypoints.csv guardado")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        input("ENTER para despegar...")
        
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
        print("poñoñin completed") 
