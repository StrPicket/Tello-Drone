#!/usr/bin/env python3
"""
Script para entrenar YOLOv11 SEGMENTACIÓN con dataset del Tello Drone
Optimizado para MSI Cyborg 15 - Intel Core Ultra 7 + RTX 4060 (8GB VRAM)
Dataset: Segmentación de instancias (máscaras de polígonos)
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import psutil
from datetime import datetime

# ============================================================
# CONFIGURACIÓN DEL USUARIO - MODIFICA AQUÍ
# ============================================================

# Ruta al dataset (CAMBIAR en Windows)
DATASET_PATH_LOCAL = r"C:\Users\ejohn\Documents\Concentracion Drones\Tello\Datasets\Tello_drone_S.v3i.yolov11"

# Configuración de entrenamiento
EPOCHS = 150  # Épocas de entrenamiento
IMG_SIZE = 640  # Tamaño de imagen
MODEL_SIZE = 'yolo11n-seg.pt'  # yolo11n-seg.pt (rápido), yolo11s-seg.pt, yolo11m-seg.pt
PROJECT_NAME = "tello_segmentation"
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Clases del dataset (ACTUALIZAR según tus clases finales)
EXPECTED_CLASSES = [
    'Fire',
    'Gas_station',
    'Gas_tank',
    'Obstacle_beacon',
    'Oxxo',
    'Pipes',
    'Pressure_Vessel',
    'Storage_Tank',
    'Tree',
    'Truck'
]

# ============================================================
# FUNCIONES DE SISTEMA
# ============================================================

def check_system():
    """Verificar configuración del sistema"""
    print("\n" + "="*60)
    print("VERIFICACIÓN DEL SISTEMA")
    print("="*60)
    
    # CPU
    print(f"💻 CPU:")
    print(f"   - Núcleos físicos: {psutil.cpu_count(logical=False)}")
    print(f"   - Núcleos lógicos: {psutil.cpu_count(logical=True)}")
    print(f"   - Uso actual: {psutil.cpu_percent()}%")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"\n🎯 RAM:")
    print(f"   - Total: {ram.total / (1024**3):.1f} GB")
    print(f"   - Disponible: {ram.available / (1024**3):.1f} GB")
    print(f"   - Uso: {ram.percent}%")
    
    # GPU
    print(f"\n🎮 GPU:")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA disponible")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Version: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   - Memoria GPU: {gpu_mem:.1f} GB VRAM")
        torch.cuda.empty_cache()
        print(f"   - Caché GPU limpiado")
    else:
        print(f"   ✗ CUDA no disponible - Se usará CPU")
    
    print("="*60 + "\n")
    return torch.cuda.is_available()


def verify_dataset_structure(dataset_path):
    """Verificar estructura del dataset de segmentación"""
    print("\n" + "="*60)
    print("VERIFICANDO ESTRUCTURA DEL DATASET DE SEGMENTACIÓN")
    print("="*60)
    
    dataset_path = Path(dataset_path)
    
    # Verificar data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        print(f"✗ Error: No se encontró data.yaml en {dataset_path}")
        return False, None
    
    # Leer data.yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"✓ data.yaml encontrado")
    print(f"  - Clases: {data.get('nc', 'N/A')}")
    print(f"  - Nombres: {data.get('names', {})}")
    
    # ============================================================
    # VALIDACIÓN DE CLASES - NUEVO
    # ============================================================
    dataset_classes = data.get('names', [])
    
    if len(dataset_classes) != len(EXPECTED_CLASSES):
        print(f"\n⚠️  ADVERTENCIA: Número de clases no coincide!")
        print(f"   Script espera: {len(EXPECTED_CLASSES)} clases")
        print(f"   Dataset tiene: {len(dataset_classes)} clases")
        print(f"\n   Dataset: {dataset_classes}")
        print(f"   Script:  {EXPECTED_CLASSES}")
        resp = input("\n¿Continuar de todos modos? [s/N]: ")
        if resp.lower() != 's':
            print("\n✗ Entrenamiento cancelado. Actualiza EXPECTED_CLASSES en el script.")
            return False, None
    else:
        # Verificar que los nombres coincidan
        mismatched = []
        for i, (dc, ec) in enumerate(zip(dataset_classes, EXPECTED_CLASSES)):
            if dc != ec:
                mismatched.append((i, dc, ec))
        
        if mismatched:
            print(f"\n⚠️  ADVERTENCIA: Nombres de clases no coinciden:")
            for idx, dc, ec in mismatched:
                print(f"   Clase {idx}: Dataset='{dc}' vs Script='{ec}'")
            resp = input("\n¿Continuar de todos modos? [s/N]: ")
            if resp.lower() != 's':
                print("\n✗ Entrenamiento cancelado. Actualiza EXPECTED_CLASSES en el script.")
                return False, None
        else:
            print(f"  ✓ Las clases coinciden perfectamente")
    
    # Verificar carpetas
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    valid_images = dataset_path / "valid" / "images"
    valid_labels = dataset_path / "valid" / "labels"
    
    print(f"\nVerificando carpetas:")
    
    if train_images.exists():
        num_train_imgs = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
        num_train_lbls = len(list(train_labels.glob("*.txt"))) if train_labels.exists() else 0
        print(f"  ✓ Train: {num_train_imgs} imágenes, {num_train_lbls} labels")
        
        # Verificar formato de segmentación en un label
        if num_train_lbls > 0:
            sample_label = list(train_labels.glob("*.txt"))[0]
            with open(sample_label, 'r') as f:
                first_line = f.readline().strip()
                if first_line:  # Verificar que no esté vacío
                    coords = first_line.split()
                    if len(coords) > 5:  # Segmentación tiene múltiples coordenadas
                        print(f"  ✓ Formato: Segmentación detectada (polígonos)")
                        print(f"    Ejemplo: Clase {coords[0]} con {len(coords)-1} coordenadas")
                    else:
                        print(f"  ⚠️  Formato: Parece bounding box, no segmentación")
                        print(f"     Verifica que hayas exportado en formato de segmentación")
                        print(f"     Ejemplo encontrado: {first_line}")
    else:
        print(f"  ✗ Train: No encontrado")
        return False, None
    
    if valid_images.exists():
        num_valid_imgs = len(list(valid_images.glob("*.jpg"))) + len(list(valid_images.glob("*.png")))
        num_valid_lbls = len(list(valid_labels.glob("*.txt"))) if valid_labels.exists() else 0
        print(f"  ✓ Valid: {num_valid_imgs} imágenes, {num_valid_lbls} labels")
    else:
        print(f"  ⚠️  Valid: No encontrado - Ejecuta dividir_dataset.py primero")
    
    # Advertencias
    if num_train_imgs == 0:
        print(f"\n✗ Error: No hay imágenes en train/images")
        return False, None
    
    if num_train_lbls == 0:
        print(f"\n⚠️  ADVERTENCIA: No hay labels - ¿Ya anotaste las imágenes?")
        print(f"   Usa Roboflow o CVAT para etiquetar máscaras antes de entrenar")
    
    if num_train_imgs != num_train_lbls:
        print(f"\n⚠️  ADVERTENCIA: Número de imágenes ({num_train_imgs}) != labels ({num_train_lbls})")
        print(f"   Algunas imágenes podrían no tener anotaciones")
    
    print("="*60 + "\n")
    return True, yaml_path


def train_tello_yolo_segmentation(dataset_yaml_path, epochs=150, img_size=640):
    """
    Entrenar modelo YOLOv11-seg para segmentación de instancias con Tello
    Optimizado para RTX 4060 (8GB VRAM)
    """
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO YOLO V11-SEG - SEGMENTACIÓN")
    print("="*60)
    
    # Configuración optimizada para RTX 4060 - SEGMENTACIÓN
    training_config = {
        # Dataset
        'data': str(dataset_yaml_path),
        'epochs': epochs,
        'imgsz': img_size,
        
        # Hardware - Optimizado para RTX 4060 (8GB VRAM)
        # IMPORTANTE: Segmentación usa más memoria que detección
        'batch': 8,  # Reducido de 16 a 8 para segmentación
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        
        # Eficiencia
        'cache': True,
        'amp': True,  # Mixed Precision para RTX 40 series
        
        # Guardado
        'project': PROJECT_NAME,
        'name': RUN_NAME,
        'exist_ok': True,
        'save': True,
        'save_period': 20,
        'patience': 50,  # Early stopping
        'plots': True,
        'verbose': True,
        
        # Optimizador
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights (para segmentación)
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Data Augmentation (optimizado para segmentación)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,  # Copy-paste es útil para segmentación
        
        # Parámetros específicos de segmentación
        'overlap_mask': True,  # Permitir máscaras superpuestas
        'mask_ratio': 4,  # Downsample ratio para máscaras
        
        # Otros
        'label_smoothing': 0.0,
        'nbs': 64,
        'dropout': 0.0,
        'val': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
    }
    
    print(f"Modelo: {MODEL_SIZE} (SEGMENTACIÓN)")
    print(f"Épocas: {epochs}")
    print(f"Tamaño imagen: {img_size}x{img_size}")
    print(f"Batch size: {training_config['batch']} (reducido para segmentación)")
    print(f"Dispositivo: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Workers: {training_config['workers']}")
    print(f"Clases a segmentar: {', '.join(EXPECTED_CLASSES)}")
    print("="*60)
    
    try:
        print(f"\nCargando modelo base: {MODEL_SIZE}")
        model = YOLO(MODEL_SIZE)
        
        print("Iniciando entrenamiento de segmentación...")
        print("(Presiona Ctrl+C para detener)\n")
        
        # Entrenar
        results = model.train(**training_config)
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print(f"📁 Mejores pesos: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
        print(f"📁 Últimos pesos: {PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
        print(f"📊 Resultados: {PROJECT_NAME}/{RUN_NAME}/")
        print("="*60 + "\n")
        
        return model, results
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido")
        return None, None
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        
        if "out of memory" in str(e).lower():
            print("\n💡 Solución para OOM en segmentación:")
            print("  - Reduce batch de 8 a 4")
            print("  - Reduce img_size de 640 a 512")
            print("  - Desactiva cache")
            print("  - Aumenta mask_ratio de 4 a 8")
        
        import traceback
        traceback.print_exc()
        return None, None


def evaluate_model(model, dataset_yaml_path):
    """Evaluar modelo de segmentación"""
    print("\n" + "="*60)
    print("EVALUANDO MODELO DE SEGMENTACIÓN")
    print("="*60)
    
    try:
        results = model.val(data=str(dataset_yaml_path), plots=True)
        
        print("\n📊 MÉTRICAS DE DETECCIÓN (BOXES):")
        print(f"  mAP50 (box):       {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
        print(f"  mAP50-95 (box):    {results.box.map:.4f} ({results.box.map*100:.2f}%)")
        print(f"  Precision (box):   {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
        print(f"  Recall (box):      {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
        
        # Métricas de segmentación
        if hasattr(results, 'seg') and results.seg is not None:
            print("\n📊 MÉTRICAS DE SEGMENTACIÓN (MASKS):")
            print(f"  mAP50 (mask):      {results.seg.map50:.4f} ({results.seg.map50*100:.2f}%)")
            print(f"  mAP50-95 (mask):   {results.seg.map:.4f} ({results.seg.map*100:.2f}%)")
            print(f"  Precision (mask):  {results.seg.mp:.4f} ({results.seg.mp*100:.2f}%)")
            print(f"  Recall (mask):     {results.seg.mr:.4f} ({results.seg.mr*100:.2f}%)")
            
            if results.seg.map50 > 0.7:
                print("\n  ✓ Excelente segmentación!")
            elif results.seg.map50 > 0.5:
                print("\n  ⚠️  Buena segmentación, puede mejorar")
            else:
                print("\n  ✗ Segmentación necesita mejorar, considera más épocas")
        else:
            print("\n⚠️  No se encontraron métricas de segmentación")
        
        print("="*60 + "\n")
        return results
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Función principal"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*5 + "ENTRENAMIENTO YOLO V11-SEG - TELLO DRONE DATASET" + " "*5 + "║")
    print("║" + " "*12 + "Segmentación de Instancias (Máscaras)" + " "*11 + "║")
    print("╚" + "="*58 + "╝")
    
    # 1. Verificar sistema
    has_gpu = check_system()
    
    if not has_gpu:
        print("⚠️  No se detectó GPU")
        resp = input("¿Continuar con CPU? [s/N]: ")
        if resp.lower() != 's':
            return
    
    # 2. Verificar dataset
    if not os.path.exists(DATASET_PATH_LOCAL):
        print(f"\n✗ Error: Dataset no encontrado en:")
        print(f"   {DATASET_PATH_LOCAL}")
        print(f"\n💡 Solución:")
        print(f"   1. Descarga tu dataset de segmentación desde Roboflow")
        print(f"   2. Asegúrate de exportar en formato YOLOv11 (segmentación)")
        print(f"   3. Actualiza DATASET_PATH_LOCAL en este script")
        return
    
    valid, yaml_path = verify_dataset_structure(DATASET_PATH_LOCAL)
    
    if not valid:
        print("\n✗ Estructura del dataset incorrecta o validación cancelada")
        return
    
    # 3. Confirmar
    print(f"\n{'='*60}")
    print("CONFIGURACIÓN FINAL")
    print(f"{'='*60}")
    print(f"Dataset: {DATASET_PATH_LOCAL}")
    print(f"Clases: {len(EXPECTED_CLASSES)} - {', '.join(EXPECTED_CLASSES)}")
    print(f"Modelo: {MODEL_SIZE} (SEGMENTACIÓN)")
    print(f"Épocas: {EPOCHS}")
    print(f"Imagen: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch: 8 (optimizado para segmentación)")
    print(f"{'='*60}\n")
    
    input("Presiona Enter para iniciar (Ctrl+C para cancelar)...")
    
    # 4. Entrenar
    model, results = train_tello_yolo_segmentation(yaml_path, epochs=EPOCHS, img_size=IMG_SIZE)
    
    if model is None:
        return
    
    # 5. Evaluar
    evaluate_model(model, yaml_path)
    
    # 6. Resumen
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
    print(f"📁 Mejores pesos: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"\n💡 Para usar el modelo de segmentación:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{PROJECT_NAME}/{RUN_NAME}/weights/best.pt')")
    print(f"  results = model.predict('imagen.jpg')")
    print(f"  # results[0].masks contiene las máscaras de segmentación")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()