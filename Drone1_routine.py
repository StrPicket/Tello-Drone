import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time
import threading

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODEL_PATH = 'best.pt'
TARGET_CLASS = 'Pipes'

KP = 0.25
KD = 0.2
DEADZONE = 60      
MAX_SPEED = 18

CORRECTION_TIME = 8.0
STABILIZATION_TIME = 1.0
INFERENCE_SIZE = 224
FRAME_SKIP = 2

# ============================================================
# SETUP
# ============================================================
print("Conectando...")
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")

tello.streamon()
time.sleep(2)

print("Cargando modelo...")
model = YOLO(MODEL_PATH)

frame_read = tello.get_frame_read()

segment_counter = 0
total_segments = 2 * (7 + 6)

latest_frame = None
display_running = True
frame_counter = 0

# Variable para controlar si estamos en modo corrección
in_correction_mode = False

# ============================================================
# THREAD DE DETECCIÓN Y DISPLAY CONTINUO
# ============================================================
def continuous_detection_thread():
    """
    Thread que SIEMPRE está detectando y mostrando TODAS las clases.
    Corre en paralelo a todo el código principal.
    """
    global latest_frame, display_running, in_correction_mode, segment_counter
    
    detection_frame_counter = 0
    
    while display_running:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.02)
            continue
        
        # Procesar 1 de cada 2 frames para optimizar
        detection_frame_counter += 1
        if detection_frame_counter % 2 != 0:
            time.sleep(0.01)
            continue
        
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            
            # DETECTAR TODAS LAS CLASES
            results = model(frame_bgr, conf=0.3, verbose=False, imgsz=INFERENCE_SIZE)
            
            vis_frame = frame_bgr.copy()
            
            # Dibujar detecciones de TODAS las clases
            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                class_names = result.names
                
                # Dibujar cada detección
                for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_names[cls_id]
                    
                    # Color según clase
                    if class_name.lower() == TARGET_CLASS.lower():
                        color = (0, 255, 0)  # Verde para Pipes
                        thickness = 3
                    else:
                        color = (255, 165, 0)  # Naranja para otras clases
                        thickness = 2
                    
                    # Dibujar bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Dibujar etiqueta
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 4), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(vis_frame, label, (x1, y1 - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Mostrar máscaras si existen
                if results[0].masks is not None:
                    for i, mask_data in enumerate(results[0].masks.data):
                        mask = mask_data.cpu().numpy()
                        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bin = (mask_resized > 0.5).astype(np.uint8)
                        
                        # Color de máscara según clase
                        cls_id = classes[i]
                        class_name = class_names[cls_id]
                        if class_name.lower() == TARGET_CLASS.lower():
                            mask_color = [0, 255, 255]  # Amarillo para Pipes
                        else:
                            mask_color = [255, 165, 0]  # Naranja para otras
                        
                        colored_mask = np.zeros_like(frame_bgr)
                        colored_mask[mask_bin > 0] = mask_color
                        vis_frame = cv2.addWeighted(vis_frame, 0.7, colored_mask, 0.3, 0)
            
            # Info de segmento en la esquina
            cv2.putText(vis_frame, f"Segmento: {segment_counter}/{total_segments}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Indicador de modo
            if in_correction_mode:
                cv2.putText(vis_frame, "MODO: CORRECCION", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis_frame, "MODO: NAVEGACION", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Actualizar frame global
            latest_frame = vis_frame
            
            # Mostrar
            cv2.imshow("Tello - Square Centroid", vis_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            pass
        
        time.sleep(0.01)

# Iniciar thread de detección continua
detection_thread_obj = threading.Thread(target=continuous_detection_thread, daemon=True)
detection_thread_obj.start()

# ============================================================
# AVANZAR 60CM
# ============================================================
def move_forward_safe():
    global in_correction_mode
    
    print(f"    → Avanzando 60cm...")
    in_correction_mode = False
    
    try:
        tello.move_forward(60)
        print(f"    ✓ Avance completo")
        
        print(f"    ⏳ Estabilizando {STABILIZATION_TIME}s...")
        time.sleep(STABILIZATION_TIME)
        
        print(f"    ✓ Estabilizado")
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    time.sleep(0.2)

# ============================================================
# ROTAR
# ============================================================
def rotate_left_90():
    global in_correction_mode
    
    print(f"    🔄 Rotando 90°...")
    in_correction_mode = False
    
    try:
        tello.rotate_counter_clockwise(90)
        print(f"    ✓ Rotación completa")
        time.sleep(1.0)
    except Exception as e:
        print(f"    ✗ Error: {e}")

# ============================================================
# CORRECCIÓN
# ============================================================
def correction_phase():
    global segment_counter, frame_counter, in_correction_mode
    
    in_correction_mode = True  # Activar modo corrección
    
    prev_error = 0
    prev_time = time.time()
    start_time = time.time()
    
    print(f"    → Control PD ({CORRECTION_TIME}s)")
    
    error_history = []
    correction_count = 0
    last_control = 0
    
    while time.time() - start_time < CORRECTION_TIME:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.02)
            continue
        
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            time.sleep(0.02)
            continue
        
        current_time = time.time()
        dt = current_time - prev_time
        if dt < 0.001:
            dt = 0.001
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        
        results = model(frame_bgr, conf=0.3, verbose=False, imgsz=INFERENCE_SIZE)
        
        pipe_found = False
        
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            
            pipe_idx = None
            best_conf = 0
            for i, cls_id in enumerate(classes):
                if class_names[cls_id].lower() == TARGET_CLASS.lower():
                    if result.boxes.conf[i] > best_conf:
                        pipe_idx = i
                        best_conf = result.boxes.conf[i]
            
            if pipe_idx is not None:
                pipe_found = True
                
                mask = result.masks.data[pipe_idx].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                
                M = cv2.moments(mask_bin)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    error = cx - center_x
                    
                    error_history.append(error)
                    if len(error_history) > 4:
                        error_history.pop(0)
                    error_filtered = int(np.mean(error_history))
                    
                    if abs(error_filtered) < DEADZONE:
                        control = int(np.clip(error_filtered * 0.08, -8, 8))
                        tello.send_rc_control(control, 0, 0, 0)
                    else:
                        p_term = KP * error_filtered
                        d_term = KD * (error_filtered - prev_error) / dt
                        control = p_term + d_term
                        
                        control_change = abs(control - last_control)
                        if control_change > 10:
                            control = last_control + np.sign(control - last_control) * 10
                        
                        control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))
                        
                        tello.send_rc_control(control, 0, 0, 0)
                        correction_count += 1
                        prev_error = error_filtered
                        last_control = control
        
        if not pipe_found:
            tello.send_rc_control(0, 0, 0, 0)
            error_history.clear()
        
        prev_time = current_time
        time.sleep(0.05)
    
    tello.send_rc_control(0, 0, 0, 0)
    print(f"    ✓ Control completo")
    in_correction_mode = False

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        print("\n=== PREVIEW ===")
        time.sleep(3)
        
        input("Presiona ENTER para despegar...")
        
        print("\n=== DESPEGUE ===")
        tello.takeoff()
        time.sleep(3)
        
        print("🔺 Subiendo altura (20cm)...")
        tello.move_up(20)
        time.sleep(2)
        
        print("Auto-calibración (4s)...")
        time.sleep(4)
        
        # RUTA: 2 vueltas × (7 + 6 segmentos)
        for i in range(2):
            print(f"\n{'='*50}\nVUELTA {i+1}/2\n{'='*50}")
            
            for j in range(5):
                segment_counter += 1
                print(f"\nSegmento {segment_counter}")
                move_forward_safe()
                correction_phase()

            move_forward_safe()
            segment_counter += 1
            print(f"\nSegmento {segment_counter}")
            
            print("Pipe LARGO completado")
            rotate_left_90()
            
            for k in range(2):
                segment_counter += 1
                print(f"\nSegmento {segment_counter}")
                move_forward_safe()
                correction_phase()
            
            move_forward_safe()
            segment_counter += 1
            print(f"\nSegmento {segment_counter}")
            time.sleep(3.0)  # ← Durante este tiempo SIGUE detectando y mostrando

            for k in range(2):
                segment_counter += 1
                print(f"\nSegmento {segment_counter}")
                move_forward_safe()
                correction_phase()
            
            move_forward_safe()
            segment_counter += 1
            print(f"\nSegmento {segment_counter}")

            print("Pipe CORTO completado")
            rotate_left_90()
        
        print("\n✅ COMPLETADO")
        
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