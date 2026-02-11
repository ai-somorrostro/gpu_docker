#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for RT-DETR
Runs sequential trials (one at a time) for efficient VRAM usage
All hyperparameters and search ranges configured via environment variables
"""

import os
from dotenv import load_dotenv
from ultralytics import RTDETR, settings
import torch
import optuna
from optuna.trial import Trial

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
    vram_fraction = get_env_float('VRAM_FRACTION', 0.90)
    limit_vram(vram_fraction)
    
    # Configure datasets directory (writable location)
    os.environ['YOLO_CONFIG_DIR'] = '/workspace/.config'
    
    # Enable TensorBoard
    settings.update({'tensorboard': True, 'runs_dir': '/workspace/runs', 'datasets_dir': '/workspace/datasets'})
    
    # Get dataset configuration
    dataset_yaml = os.environ.get('DATASET_YAML', '')
    if not dataset_yaml:
        print("ERROR: No se especificó DATASET_YAML")
        print("Uso: ./train.sh train_rtdetr_optuna.py dataset config/coco8.yaml")
        exit(1)
    
    # Convert to absolute path for Optuna workers
    if not dataset_yaml.startswith('/'):
        dataset_yaml = f'/workspace/{dataset_yaml}'
    
    model_name = os.environ.get('MODEL', 'rtdetr-l.pt')
    optimizer = os.environ.get('OPTIMIZER', 'auto')
    
    # Basic training parameters
    epochs = get_env_int('EPOCHS', 50)
    imgsz = get_env_int('IMG_SIZE', 640)
    batch_size = get_env_int('BATCH_SIZE', -1)
    patience = get_env_int('PATIENCE', 20)

    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {imgsz}")
    print(f"Patience: {patience}")
    print(f"Optimizer: {optimizer}")
    
    # Optuna configuration
    n_trials = get_env_int('TUNE_ITERATIONS', 50)
    tune_grace_period = get_env_int('TUNE_GRACE_PERIOD', 10)
    tune_metric = os.environ.get('TUNE_METRIC', 'metrics/mAP50(B)')
    
    print("="*70)
    print("OPTUNA HYPERPARAMETER SEARCH FOR RT-DETR")
    print("="*70)
    print(f"Model:            {model_name}")
    print(f"Dataset:          {dataset_yaml}")
    print(f"Epochs per trial: {epochs}")
    print(f"Image size:       {imgsz}")
    print(f"Total trials:     {n_trials}")
    print(f"Grace period:     {tune_grace_period} epochs")
    print(f"Metric:           {tune_metric}")
    print(f"VRAM per trial:   {vram_fraction*100}%")
    print("="*70)
    print()
    
    # Search space boundaries from environment variables
    lr0_min = get_env_float('LR0_MIN', 0.0001)
    lr0_max = get_env_float('LR0_MAX', 0.01)
    
    lrf_min = get_env_float('LRF_MIN', 0.01)
    lrf_max = get_env_float('LRF_MAX', 0.5)
    
    hsv_s_min = get_env_float('HSV_S_MIN', 0.0)
    hsv_s_max = get_env_float('HSV_S_MAX', 0.9)
    
    hsv_v_min = get_env_float('HSV_V_MIN', 0.0)
    hsv_v_max = get_env_float('HSV_V_MAX', 0.9)
    
    degrees_min = get_env_float('DEGREES_MIN', 0.0)
    degrees_max = get_env_float('DEGREES_MAX', 45.0)
    
    translate_min = get_env_float('TRANSLATE_MIN', 0.0)
    translate_max = get_env_float('TRANSLATE_MAX', 0.3)
    
    scale_min = get_env_float('SCALE_MIN', 0.0)
    scale_max = get_env_float('SCALE_MAX', 0.9)
    
    shear_min = get_env_float('SHEAR_MIN', 0.0)
    shear_max = get_env_float('SHEAR_MAX', 10.0)
    
    mixup_min = get_env_float('MIXUP_MIN', 0.0)
    mixup_max = get_env_float('MIXUP_MAX', 1.0)
    
    # Fixed augmentation parameters (not tuned)
    fixed_params = {
        'workers': get_env_int('WORKERS', 2),
        'perspective': get_env_float('PERSPECTIVE', 0.0),
        'flipud': get_env_float('FLIPUD', 0.0),
        'fliplr': get_env_float('FLIPLR', 0.5),
        'hsv_h': get_env_float('HSV_H', 0.015),
        'mosaic': get_env_float('MOSAIC', 1.0),
        'close_mosaic': get_env_int('CLOSE_MOSAIC', 10),
    }
    
    print("Search space configured (tunable):")
    print(f"  {'lr0':15s}: [{lr0_min}, {lr0_max}]")
    print(f"  {'lrf':15s}: [{lrf_min}, {lrf_max}]")
    print(f"  {'hsv_s':15s}: [{hsv_s_min}, {hsv_s_max}]")
    print(f"  {'hsv_v':15s}: [{hsv_v_min}, {hsv_v_max}]")
    print(f"  {'degrees':15s}: [{degrees_min}, {degrees_max}]")
    print(f"  {'translate':15s}: [{translate_min}, {translate_max}]")
    print(f"  {'scale':15s}: [{scale_min}, {scale_max}]")
    print(f"  {'shear':15s}: [{shear_min}, {shear_max}]")
    print(f"  {'mixup':15s}: [{mixup_min}, {mixup_max}]")
    print()
    print("Fixed parameters (not tuned):")
    for param, value in fixed_params.items():
        print(f"  {param:15s}: {value}")
    print()
    
    def objective(trial: Trial):
        """Optuna objective function"""
        # Sample hyperparameters
        params = {
            'lr0': trial.suggest_float('lr0', lr0_min, lr0_max),
            'lrf': trial.suggest_float('lrf', lrf_min, lrf_max),
            'hsv_s': trial.suggest_float('hsv_s', hsv_s_min, hsv_s_max),
            'hsv_v': trial.suggest_float('hsv_v', hsv_v_min, hsv_v_max),
            'degrees': trial.suggest_float('degrees', degrees_min, degrees_max),
            'translate': trial.suggest_float('translate', translate_min, translate_max),
            'scale': trial.suggest_float('scale', scale_min, scale_max),
            'shear': trial.suggest_float('shear', shear_min, shear_max),
            'mixup': trial.suggest_float('mixup', mixup_min, mixup_max),
        }
        
        # Combine with fixed params
        all_params = {**fixed_params, **params}
        
        # Load fresh model for this trial
        model = RTDETR(model_name)
        
        # Train with suggested hyperparameters
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            patience=patience,
            batch=batch_size,
            optimizer=optimizer,
            device=0,
            project=f'/workspace/runs/optuna/trial_{trial.number}',
            name='train',
            verbose=False,
            save=True,
            plots=True,
            amp=False,
            **all_params
        )
        
        # Validate and get metrics
        metrics = model.val(
            project=f'/workspace/runs/optuna/trial_{trial.number}',
            name='val'
        )
        
        # Extract the metric value (default: mAP50)
        metric_value = metrics.box.map50  # mAP50
        
        # Report metric for intermediate values
        trial.report(metric_value, step=epochs)
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()
        
        return metric_value
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name='rtdetr_hyperparameter_search',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=tune_grace_period,
            n_warmup_steps=tune_grace_period
        )
    )
    
    print("Starting hyperparameter search...")
    print("This will run ONE trial at a time (sequential execution)")
    print()
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH COMPLETED!")
    print("="*70)
    
    # Get best trial results
    print("\nBest trial results:")
    print(f"Best mAP50: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param:15s}: {value:.6f}")
    
    print(f"\nResults saved in: /workspace/runs/optuna/")
    print(f"Number of finished trials: {len(study.trials)}")
    
    # Save study to database (optional)
    try:
        import joblib
        joblib.dump(study, '/workspace/runs/optuna/study.pkl')
        print("Study object saved to /workspace/runs/optuna/study.pkl")
    except Exception as e:
        print(f"Could not save study object: {e}")
    
    print("="*70)

if __name__ == "__main__":
    main()
