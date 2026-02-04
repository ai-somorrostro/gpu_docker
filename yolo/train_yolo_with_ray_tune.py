#!/usr/bin/env python3
"""
Ray Tune Hyperparameter Search for YOLO
Runs sequential trials (one at a time) for efficient VRAM usage
All hyperparameters and search ranges configured via environment variables
"""

import os
from dotenv import load_dotenv
from ultralytics import YOLO, settings
import torch
from ray import tune

def limit_vram(vram_fraction):
    """Limit VRAM usage per trial"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(vram_fraction, device=0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM Available to allocate {vram_fraction*100}% (~{vram_fraction * total_vram:.1f}GB of {total_vram:.1f}GB)")

def get_env_float(key, default):
    """Get float from environment variable"""
    return float(os.environ.get(key, str(default)))

def get_env_int(key, default):
    """Get int from environment variable"""
    return int(os.environ.get(key, str(default)))

def main():
    # Load environment variables
    load_dotenv('/tmp/.env')
    
    # VRAM management
    vram_fraction = get_env_float('VRAM_FRACTION', 0.25)
    limit_vram(vram_fraction)
    
    # Configure datasets directory (writable location)
    os.environ['YOLO_CONFIG_DIR'] = '/workspace/.config'
    
    # Enable TensorBoard
    settings.update({'tensorboard': True, 'runs_dir': '/workspace/runs', 'datasets_dir': '/workspace/datasets'})
    
    # Get dataset configuration
    dataset_yaml = os.environ.get('DATASET_YAML', '')
    if not dataset_yaml:
        print("ERROR: No se especificó DATASET_YAML")
        print("Uso: ./train.sh train_yolo_with_ray_tune.py dataset config/coco8.yaml")
        exit(1)
    
    # Convert to absolute path for Ray Tune workers
    if not dataset_yaml.startswith('/'):
        dataset_yaml = f'/workspace/{dataset_yaml}'
    
    model_name = os.environ.get('MODEL', 'yolov8n.pt')
    
    # Basic training parameters
    epochs = get_env_int('EPOCHS', 10)
    imgsz = get_env_int('IMG_SIZE', 640)
    patience = get_env_int('PATIENCE', 50)
    
    # Ray Tune configuration
    tune_iterations = get_env_int('TUNE_ITERATIONS', 50)
    tune_grace_period = get_env_int('TUNE_GRACE_PERIOD', 10)
    tune_metric = os.environ.get('TUNE_METRIC', 'metrics/mAP50(B)')
    
    print("="*70)
    print("RAY TUNE HYPERPARAMETER SEARCH")
    print("="*70)
    print(f"Model:            {model_name}")
    print(f"Dataset:          {dataset_yaml}")
    print(f"Epochs per trial: {epochs}")
    print(f"Image size:       {imgsz}")
    print(f"Total trials:     {tune_iterations}")
    print(f"Grace period:     {tune_grace_period} epochs")
    print(f"Metric:           {tune_metric}")
    print(f"Execution:        Sequential (1 trial at a time)")
    print(f"GPU per trial:    1")
    print(f"VRAM config:      {vram_fraction*100}% (Ray Tune maneja asignación)")
    print("="*70)
    print()
    
    # Build search space from environment variables
    search_space = {}
    
    # Learning rate
    lr0_min = get_env_float('LR0_MIN', 1e-5)
    lr0_max = get_env_float('LR0_MAX', 1e-1)
    search_space['lr0'] = tune.uniform(lr0_min, lr0_max)
    
    # Final learning rate factor
    lrf_min = get_env_float('LRF_MIN', 0.01)
    lrf_max = get_env_float('LRF_MAX', 1.0)
    search_space['lrf'] = tune.uniform(lrf_min, lrf_max)
    
    # Momentum
    momentum_min = get_env_float('MOMENTUM_MIN', 0.6)
    momentum_max = get_env_float('MOMENTUM_MAX', 0.98)
    search_space['momentum'] = tune.uniform(momentum_min, momentum_max)
    
    # Weight decay
    weight_decay_min = get_env_float('WEIGHT_DECAY_MIN', 0.0)
    weight_decay_max = get_env_float('WEIGHT_DECAY_MAX', 0.001)
    search_space['weight_decay'] = tune.uniform(weight_decay_min, weight_decay_max)
    
    # Warmup epochs
    warmup_min = get_env_float('WARMUP_EPOCHS_MIN', 0.0)
    warmup_max = get_env_float('WARMUP_EPOCHS_MAX', 5.0)
    search_space['warmup_epochs'] = tune.uniform(warmup_min, warmup_max)
    
    # Loss weights
    box_min = get_env_float('BOX_LOSS_MIN', 0.02)
    box_max = get_env_float('BOX_LOSS_MAX', 0.2)
    search_space['box'] = tune.uniform(box_min, box_max)
    
    cls_min = get_env_float('CLS_LOSS_MIN', 0.2)
    cls_max = get_env_float('CLS_LOSS_MAX', 4.0)
    search_space['cls'] = tune.uniform(cls_min, cls_max)
    
    # HSV augmentation
    hsv_h_min = get_env_float('HSV_H_MIN', 0.0)
    hsv_h_max = get_env_float('HSV_H_MAX', 0.1)
    search_space['hsv_h'] = tune.uniform(hsv_h_min, hsv_h_max)
    
    hsv_s_min = get_env_float('HSV_S_MIN', 0.0)
    hsv_s_max = get_env_float('HSV_S_MAX', 0.9)
    search_space['hsv_s'] = tune.uniform(hsv_s_min, hsv_s_max)
    
    hsv_v_min = get_env_float('HSV_V_MIN', 0.0)
    hsv_v_max = get_env_float('HSV_V_MAX', 0.9)
    search_space['hsv_v'] = tune.uniform(hsv_v_min, hsv_v_max)
    
    # Geometric augmentation
    degrees_min = get_env_float('DEGREES_MIN', 0.0)
    degrees_max = get_env_float('DEGREES_MAX', 45.0)
    search_space['degrees'] = tune.uniform(degrees_min, degrees_max)
    
    translate_min = get_env_float('TRANSLATE_MIN', 0.0)
    translate_max = get_env_float('TRANSLATE_MAX', 0.9)
    search_space['translate'] = tune.uniform(translate_min, translate_max)
    
    scale_min = get_env_float('SCALE_MIN', 0.0)
    scale_max = get_env_float('SCALE_MAX', 0.9)
    search_space['scale'] = tune.uniform(scale_min, scale_max)
    
    shear_min = get_env_float('SHEAR_MIN', 0.0)
    shear_max = get_env_float('SHEAR_MAX', 10.0)
    search_space['shear'] = tune.uniform(shear_min, shear_max)
    
    perspective_min = get_env_float('PERSPECTIVE_MIN', 0.0)
    perspective_max = get_env_float('PERSPECTIVE_MAX', 0.001)
    search_space['perspective'] = tune.uniform(perspective_min, perspective_max)
    
    # Flip augmentation
    flipud_min = get_env_float('FLIPUD_MIN', 0.0)
    flipud_max = get_env_float('FLIPUD_MAX', 1.0)
    search_space['flipud'] = tune.uniform(flipud_min, flipud_max)
    
    fliplr_min = get_env_float('FLIPLR_MIN', 0.0)
    fliplr_max = get_env_float('FLIPLR_MAX', 1.0)
    search_space['fliplr'] = tune.uniform(fliplr_min, fliplr_max)
    
    # Mosaic and Mixup
    mosaic_min = get_env_float('MOSAIC_MIN', 0.0)
    mosaic_max = get_env_float('MOSAIC_MAX', 1.0)
    search_space['mosaic'] = tune.uniform(mosaic_min, mosaic_max)
    
    mixup_min = get_env_float('MIXUP_MIN', 0.0)
    mixup_max = get_env_float('MIXUP_MAX', 1.0)
    search_space['mixup'] = tune.uniform(mixup_min, mixup_max)
    
    print("Search space configured:")
    for param, space in search_space.items():
        print(f"  {param:15s}: {space}")
    print()
    
    # Load model
    model = YOLO(model_name)
    print(f"Model loaded: {model_name}")
    print()
    
    # Start Ray Tune hyperparameter search
    print("Starting hyperparameter search...")
    print("This will run ONE trial at a time (sequential execution)")
    print()
    
    result_grid = model.tune(
        data=dataset_yaml,
        space=search_space,
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,
        batch=-1,  # Auto batch size
        device=0,
        project='/workspace/runs/tune',
        use_ray=True,
        iterations=tune_iterations,
        grace_period=tune_grace_period,
        gpu_per_trial=1,  # Sequential execution: 1 GPU = 1 trial at a time
    )
    
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH COMPLETED!")
    print("="*70)
    
    # Get best trial results
    if result_grid:
        print("\nBest trial results:")
        best_result = result_grid.get_best_result(metric=tune_metric, mode="max")
        print(f"Best {tune_metric}: {best_result.metrics.get(tune_metric, 'N/A')}")
        print("\nBest hyperparameters:")
        for param, value in best_result.config.items():
            if param in search_space:
                print(f"  {param:15s}: {value:.6f}")
        print(f"\nResults saved in: /workspace/runs/tune/")
    else:
        print("\nNo results available")
    
    print("="*70)

if __name__ == "__main__":
    main()
