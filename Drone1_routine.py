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
TARGET_CLASS = 'Pipes'

KP = 0.25
KD = 0.2
DEADZONE = 60      
MAX_SPEED = 18

CORRECTION_TIME = 10.0
STABILIZATION_TIME = 1.0
INFERENCE_SIZE = 224
INFERENCE_SIZE_LIGHT = 160  # ← NUEVO: para navegación ligera
FRAME_SKIP = 2

CAMERA_TRANSFORM = "flip_v"

# ========== ROI AMPLIADO ==========
ROI_LEFT_LIMIT = 80
ROI_TOP_LIMIT = 0
ROI_BOTTOM_LIMIT = 480
ROI_RIGHT_LIMIT = 880

DEBUG_DETECTIONS = True
ENABLE_VISUALIZATION = True  # ← NUEVO: False para desactivar ventana

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
in_correction_mode = False

# ============================================================
# TRANSFORMACIÓN OPTIMIZADA
# ============================================================
def transform_frame(frame):
    if CAMERA_TRANSFORM == "flip_v":
        return cv2.flip(frame, 0)
    elif CAMERA_TRANSFORM == "flip_both":
        return cv2.flip(frame, -1)
    return frame

# ============================================================
# THREAD DE DETECCIÓN OPTIMIZADO
# ============================================================
def continuous_detection_thread():
    global latest_frame, display_running, in_correction_mode, segment_counter
    
    detection_counter = 0
    consecutive_failures = 0  # ← NUEVO: para reconexión automática
    
    while display_running:
        frame = frame_read.frame
        
        # ========== MANEJO DE ERRORES DE CÁMARA ==========
        if frame is None:
            consecutive_failures += 1
            if consecutive_failures > 50:  # ~1 segundo sin frames
                print("⚠️ Reconectando stream...")
                try:
                    tello.streamoff()
                    time.sleep(0.5)
                    tello.streamon()
                    time.sleep(2)
                    consecutive_failures = 0
                    print("✓ Stream reconectado")
                except Exception as e:
                    print(f"❌ Error reconexión: {e}")
            time.sleep(0.02)
            continue
        
        consecutive_failures = 0  # ← Reset si hay frame válido
        
        detection_counter += 1
        
        # ========== PROCESAMIENTO ADAPTATIVO ==========
        # Durante corrección: procesar cada 2 frames (rápido)
        # Durante navegación: procesar cada 5 frames (más lento)
        skip_rate = 2 if in_correction_mode else 5  # ← NUEVO
        
        if detection_counter % skip_rate != 0:
            time.sleep(0.01)
            continue
        
        try:
            # Transformar frame
            frame_bgr = transform_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            h, w = frame_bgr.shape[:2]
            center_x = w // 2
            
            # ========== INFERENCIA ADAPTATIVA ==========
            # Resolución más alta en corrección, más baja en navegación
            inference_size = INFERENCE_SIZE if in_correction_mode else INFERENCE_SIZE_LIGHT
            results = model(frame_bgr, conf=0.3, verbose=False, imgsz=inference_size)
            
            # Frame limpio para visualización
            vis_frame = frame_bgr.copy()
            
            # ========== DIBUJAR SOLO LO ESENCIAL ==========
            
            # Línea vertical del ROI izquierdo
            cv2.line(vis_frame, (ROI_LEFT_LIMIT, 0), (ROI_LEFT_LIMIT, h), (0, 0, 255), 3)
            cv2.putText(vis_frame, "ROI_L", (ROI_LEFT_LIMIT + 5, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Línea vertical del ROI derecho
            cv2.line(vis_frame, (ROI_RIGHT_LIMIT, 0), (ROI_RIGHT_LIMIT, h), (0, 0, 255), 3)
            cv2.putText(vis_frame, "ROI_R", (ROI_RIGHT_LIMIT - 80, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Centro de frame
            cv2.line(vis_frame, (center_x, 0), (center_x, h), (255, 255, 0), 1)
            cv2.putText(vis_frame, "Centro", (center_x + 5, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Procesar detecciones
            if len(results) > 0 and results[0].boxes is not None:
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                class_names = result.names
                
                # Dibujar solo bounding boxes y centroides
                for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_names[cls_id]
                    
                    # Calcular centroide del bbox
                    cx_box = (x1 + x2) // 2
                    cy_box = (y1 + y2) // 2
                    
                    in_roi = (cx_box >= ROI_LEFT_LIMIT and 
                             cx_box <= ROI_RIGHT_LIMIT and
                             y1 >= ROI_TOP_LIMIT and 
                             y2 <= ROI_BOTTOM_LIMIT)
                    
                    # Color según clase y ROI
                    if class_name.lower() in ['pipes', 'pipe', 'fire']:
                        color = (0, 255, 0) if in_roi else (128, 128, 128)
                    else:
                        color = (255, 165, 0)
                    
                    # Bbox simple
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label compacto
                    label = f"{class_name}:{conf:.2f}"
                    cv2.putText(vis_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Centroide del bbox
                    cv2.circle(vis_frame, (cx_box, cy_box), 4, color, -1)
                
                # Procesar máscaras
                if results[0].masks is not None:
                    for i, cls_id in enumerate(classes):
                        class_name = class_names[cls_id]
                        if class_name.lower() in ['pipes', 'pipe', 'fire']:
                            mask = result.masks.data[i].cpu().numpy()
                            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            mask_bin = (mask_resized > 0.5).astype(np.uint8)
                            
                            # Centroide de la máscara
                            M = cv2.moments(mask_bin)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                in_roi = (cx >= ROI_LEFT_LIMIT and 
                                         cx <= ROI_RIGHT_LIMIT and
                                         cy >= ROI_TOP_LIMIT and 
                                         cy <= ROI_BOTTOM_LIMIT)
                                
                                # Dibujar centroide de máscara (más grande)
                                centroid_color = (0, 255, 255) if in_roi else (100, 100, 100)
                                cv2.circle(vis_frame, (cx, cy), 8, centroid_color, -1)
                                cv2.circle(vis_frame, (cx, cy), 10, centroid_color, 2)
                                
                                # Error desde centro
                                error = cx - center_x
                                cv2.line(vis_frame, (center_x, cy), (cx, cy), centroid_color, 2)
                                cv2.putText(vis_frame, f"e:{error}", (cx + 15, cy), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, centroid_color, 2)
            
            # Info mínima en pantalla
            info_y = h - 60
            cv2.putText(vis_frame, f"Seg:{segment_counter}/{total_segments}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            mode_text = "CORRIGIENDO" if in_correction_mode else "NAVEGANDO"
            mode_color = (0, 255, 0) if in_correction_mode else (0, 255, 255)
            cv2.putText(vis_frame, mode_text, (10, info_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            
            # Mostrar resolución de inferencia actual
            inference_text = f"Infer:{inference_size}px"
            cv2.putText(vis_frame, inference_text, (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            latest_frame = vis_frame
            
            # ========== VISUALIZACIÓN CONDICIONAL ==========
            if ENABLE_VISUALIZATION:
                cv2.imshow("Tello Optimizado", vis_frame)
                cv2.waitKey(1)
            
        except Exception as e:
            print(f"Error display: {e}")
        
        time.sleep(0.01)

detection_thread_obj = threading.Thread(target=continuous_detection_thread, daemon=True)
detection_thread_obj.start()

def sensor_calibration():
    """Calibración de sensores del drone"""
    print("    → Estabilizando sensores...")
    time.sleep(2)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    print("    ✓ Calibración completa")

# ============================================================
# FUNCIONES DE NAVEGACIÓN
# ============================================================
def move_forward_safe():
    global in_correction_mode
    print(f"    → Avanzando 60cm...")
    in_correction_mode = False
    try:
        tello.move_forward(60)
        print(f"    ✓ Avance completo")
        time.sleep(STABILIZATION_TIME)
    except Exception as e:
        print(f"    ✗ Error: {e}")
    time.sleep(0.2)

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

def count_pipes_in_roi(samples=5, verbose=False):
    """
    Cuenta cuántos pipes están dentro del ROI
    Toma múltiples muestras y devuelve el valor más común
    """
    global frame_counter
    
    detections = []
    
    for sample in range(samples):
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.05)
            continue
        
        frame_bgr = transform_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        h, w = frame_bgr.shape[:2]
        
        # Usar resolución más alta para conteo preciso
        results = model(frame_bgr, conf=0.3, verbose=False, imgsz=INFERENCE_SIZE)
        
        pipes_in_roi = 0
        
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            
            for i, cls_id in enumerate(classes):
                class_name = class_names[cls_id]
                
                # Aceptar pipes y fire
                if class_name.lower() in ['pipes', 'pipe', 'fire']:
                    mask = result.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask_resized > 0.5).astype(np.uint8)
                    
                    M = cv2.moments(mask_bin)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        in_roi = (cx >= ROI_LEFT_LIMIT and 
                                 cx <= ROI_RIGHT_LIMIT and
                                 cy >= ROI_TOP_LIMIT and 
                                 cy <= ROI_BOTTOM_LIMIT)
                        
                        if in_roi:
                            pipes_in_roi += 1
        
        detections.append(pipes_in_roi)
        
        if verbose and sample < samples - 1:
            print(f"      Muestra {sample+1}/{samples}: {pipes_in_roi} pipes")
        
        time.sleep(0.1)
    
    # Devolver el valor más común
    if detections:
        counter = Counter(detections)
        most_common = counter.most_common(1)[0][0]
        if verbose:
            print(f"      → Detecciones: {detections} → Más común: {most_common}")
        return most_common
    
    return 0

# ============================================================
# CORRECCIÓN OPTIMIZADA CON FEEDBACK
# ============================================================
def correction_phase():
    global segment_counter, frame_counter, in_correction_mode
    
    in_correction_mode = True
    prev_error = 0
    prev_time = time.time()
    start_time = time.time()
    
    print(f"    → Control PD ({CORRECTION_TIME}s) [ROI: {ROI_LEFT_LIMIT}<x<{ROI_RIGHT_LIMIT}]")
    
    error_buffer = deque(maxlen=4)
    last_control = 0
    corrections_made = 0
    no_detection_count = 0
    
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
        dt = max(current_time - prev_time, 0.001)
        
        frame_bgr = transform_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        
        # Usar resolución completa en corrección
        results = model(frame_bgr, conf=0.3, verbose=False, imgsz=INFERENCE_SIZE)
        
        pipe_found = False
        
        # ========== DEBUG: MOSTRAR TODAS LAS DETECCIONES ==========
        if DEBUG_DETECTIONS and len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            detected_classes = [class_names[c] for c in classes]
            if detected_classes:
                print(f"      DEBUG: Detectadas clases: {set(detected_classes)}")
        
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            
            best_pipe_idx = None
            best_conf = 0
            candidates_info = []
            
            for i, cls_id in enumerate(classes):
                class_name = class_names[cls_id]
                
                # ========== ACEPTAR PIPES Y FIRE ==========
                if class_name.lower() in ['pipes', 'pipe', 'fire']:
                    mask = result.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask_resized > 0.5).astype(np.uint8)
                    
                    M = cv2.moments(mask_bin)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        in_roi = (cx >= ROI_LEFT_LIMIT and 
                                 cx <= ROI_RIGHT_LIMIT and
                                 cy >= ROI_TOP_LIMIT and 
                                 cy <= ROI_BOTTOM_LIMIT)
                        
                        conf = result.boxes.conf[i].item()
                        candidates_info.append(f"{class_name}@({cx},{cy}) ROI:{in_roi} conf:{conf:.2f}")
                        
                        if in_roi and conf > best_conf:
                            best_pipe_idx = i
                            best_conf = conf
            
            # ========== DEBUG: MOSTRAR CANDIDATOS ==========
            if DEBUG_DETECTIONS and candidates_info:
                print(f"      Candidatos: {' | '.join(candidates_info)}")
            
            if best_pipe_idx is not None:
                pipe_found = True
                no_detection_count = 0
                
                mask = result.masks.data[best_pipe_idx].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                
                M = cv2.moments(mask_bin)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    error = cx - center_x
                    
                    error_buffer.append(error)
                    error_filtered = int(np.mean(error_buffer))
                    
                    if abs(error_filtered) < DEADZONE:
                        control = int(np.clip(error_filtered * 0.08, -8, 8))
                    else:
                        p_term = KP * error_filtered
                        d_term = KD * (error_filtered - prev_error) / dt
                        control = p_term + d_term
                        
                        control_change = abs(control - last_control)
                        if control_change > 10:
                            control = last_control + np.sign(control - last_control) * 10
                        
                        control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))
                        corrections_made += 1
                    
                    tello.send_rc_control(control, 0, 0, 0)
                    prev_error = error_filtered
                    last_control = control
                    
                    # Feedback cada 2 segundos
                    if int(time.time() - start_time) % 2 == 0 and corrections_made > 0:
                        print(f"      ✓ error:{error_filtered:4d} | ctrl:{control:3d}")
                        corrections_made = 0
        
        if not pipe_found:
            no_detection_count += 1
            if DEBUG_DETECTIONS and no_detection_count % 20 == 0:
                print(f"      ⚠ Sin detección válida en ROI ({no_detection_count} frames)")
            tello.send_rc_control(0, 0, 0, 0)
            error_buffer.clear()
        
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
        print("\n=== CONFIGURACIÓN ===")
        print(f"Cam: {CAMERA_TRANSFORM}")
        print(f"ROI: {ROI_LEFT_LIMIT} < x < {ROI_RIGHT_LIMIT}")
        print(f"Tiempo corrección: {CORRECTION_TIME}s")
        print(f"Inferencia: {INFERENCE_SIZE}px (corrección) / {INFERENCE_SIZE_LIGHT}px (navegación)")
        print(f"Visualización: {'Activada' if ENABLE_VISUALIZATION else 'Desactivada'}")
        time.sleep(3)
        
        input("Presiona ENTER para despegar...")
        
        print("\n=== DESPEGUE ===")
        tello.takeoff()
        
        print("⏳ Esperando 3s...")
        time.sleep(3)
        print("✓ Listo para iniciar")
        
        for i in range(2):
            print(f"\n{'='*50}\nVUELTA {i+1}/2\n{'='*50}")
            
            for j in range(5):
                if j == 6:
                    move_forward_safe()
                    segment_counter += 1
                    print(f"\nSeg {segment_counter}")
                else:
                    segment_counter += 1
                    print(f"\nSeg {segment_counter}")
                    move_forward_safe()
                    
                    # Esperar y verificar detección durante 2.5 segundos
                    print(f"    🔍 Buscando pipes (2.5s)...")
                    detection_start = time.time()
                    detection_duration = 2.5
                    detection_samples = []

                    while time.time() - detection_start < detection_duration:
                        pipes_count = count_pipes_in_roi(samples=3, verbose=False)
                        detection_samples.append(pipes_count)
                        print(f"      ⏱️ {time.time() - detection_start:.1f}s: {pipes_count} pipes")
                        time.sleep(0.3)

                    # Analizar resultados: usar el valor más común
                    if detection_samples:
                        counter = Counter(detection_samples)
                        pipes_detected = counter.most_common(1)[0][0]
                        detection_rate = detection_samples.count(pipes_detected) / len(detection_samples) * 100
                        
                        print(f"    📊 Pipes detectados: {pipes_detected} (confianza: {detection_rate:.0f}%)")
                        print(f"    📈 Muestras: {detection_samples}")
                        
                        if pipes_detected == 0:
                            print(f"    ⚠️ Sin pipes - Solo avance")
                        elif pipes_detected > 2:
                            print(f"    ⚠️ Demasiados pipes - Solo avance")
                        else:
                            print(f"    ✓ Ejecutando corrección")
                            correction_phase()
                    else:
                        print(f"    ⚠️ Error en detección - Solo avance")
            
            print("Pipe LARGO OK")
            rotate_left_90()

            for j in range(6):
                if j == 6:
                    move_forward_safe()
                    segment_counter += 1
                    print(f"\nSeg {segment_counter}")
                else:
                    segment_counter += 1
                    print(f"\nSeg {segment_counter}")
                    move_forward_safe()
                    
                    # Esperar y verificar detección durante 2.5 segundos
                    print(f"    🔍 Buscando pipes (2.5s)...")
                    detection_start = time.time()
                    detection_duration = 2.5
                    detection_samples = []

                    while time.time() - detection_start < detection_duration:
                        pipes_count = count_pipes_in_roi(samples=3, verbose=False)
                        detection_samples.append(pipes_count)
                        print(f"      ⏱️ {time.time() - detection_start:.1f}s: {pipes_count} pipes")
                        time.sleep(0.3)

                    # Analizar resultados: usar el valor más común
                    if detection_samples:
                        counter = Counter(detection_samples)
                        pipes_detected = counter.most_common(1)[0][0]
                        detection_rate = detection_samples.count(pipes_detected) / len(detection_samples) * 100
                        
                        print(f"    📊 Pipes detectados: {pipes_detected} (confianza: {detection_rate:.0f}%)")
                        print(f"    📈 Muestras: {detection_samples}")
                        
                        if pipes_detected == 0:
                            print(f"    ⚠️ Sin pipes - Solo avance")
                        elif pipes_detected > 2:
                            print(f"    ⚠️ Demasiados pipes - Solo avance")
                        else:
                            print(f"    ✓ Ejecutando corrección")
                            correction_phase()
                    else:
                        print(f"    ⚠️ Error en detección - Solo avance")
            
            move_forward_safe()
            segment_counter += 1
            print(f"\nSeg {segment_counter}")
            time.sleep(3.0)

            print("Pipe CORTO OK")
            rotate_left_90()
        
        print("\n✅ COMPLETADO")
        
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