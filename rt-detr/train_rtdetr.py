#!/usr/bin/env python3
"""
Script básico para entrenamiento de RT-DETR
Descarga automáticamente el modelo si no existe
"""

import os
from dotenv import load_dotenv
from ultralytics import RTDETR, settings
import torch
from torch.utils.tensorboard import SummaryWriter

def main():
    # Cargar variables del .env
    load_dotenv('/tmp/.env')
    
    # Limitar VRAM (0.0-1.0, donde 1.0 = 100% de la VRAM disponible)
    vram_fraction = float(os.environ.get('VRAM_FRACTION', '0.25'))  # 25% default (~2GB en 8GB GPU)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(vram_fraction, device=0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM limitada a {vram_fraction*100}% (~{vram_fraction * total_vram:.1f}GB de {total_vram:.1f}GB)")
    
    dataset_path = os.environ.get('DATASET_PATH', '/workspace/dataset')

    # Activar TensorBoard ANTES de crear el modelo
    settings.update({'tensorboard': True, 'runs_dir': '/workspace/runs'})

    # 👇 Los alumnos escriben AQUÍ sus métricas custom de TensorBoard
    tb_writer = SummaryWriter(log_dir="/workspace/runs/forced_tb")
    tb_writer.add_text("status", "tensorboard_enabled", 0)
    tb_writer.flush()
    tb_writer.close()
    
    model_name = os.environ.get('MODEL', 'rtdetr-l.pt')
    
    model = RTDETR(model_name)
    print(f"Using model: {model_name}")
    # Dataset YAML (desde variable de entorno o default)
    dataset_yaml = os.environ.get('DATASET_YAML', '')
    
    if not dataset_yaml:
        print("ERROR: No se especificó DATASET_YAML")
        print("Uso: ./train.sh train_rtdetr.py dataset config/coco8.yaml")
        exit(1)
    
    # Leer parámetros de entrenamiento
    epochs = int(os.environ.get('EPOCHS', '10'))
    imgsz = int(os.environ.get('IMG_SIZE', '640'))
    batch_size = int(os.environ.get('BATCH_SIZE', '8'))
    learning_rate = float(os.environ.get('LEARNING_RATE', '0.01'))
    optimizer = os.environ.get('OPTIMIZER', 'Adam')
    patience = int(os.environ.get('PATIENCE', '50'))
    
    # Imprimir parámetros
    print(f"\nModelo: {model_name}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, ImgSize: {imgsz}, LR: {learning_rate}, Optimizer: {optimizer}, Patience: {patience}\n")
    
    # Entrenar el modelo
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,
        patience=patience,
        project='/workspace/runs/detect',
        name='train',
        device=0,
        verbose=True,
        save=True,
        plots=True
    )
    
    print("\n" + "="*60)
    print("Entrenamiento completado!")
    print(f"Resultados guardados en: {results.save_dir}")
    print("="*60)
    
    # Validar el modelo (guardar en /workspace/runs)
    print("\nValidando modelo...")
    metrics = model.val(project="/workspace/runs/detect", name="val")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    

if __name__ == "__main__":
    main()
