#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate evaluation metrics from multiple cross-validation folds.

This script reads metrics.json from all folds and computes the mean and std.
"""

import os
import json
import numpy as np
import sys


def aggregate_metrics(model_name, city, feature, auxiliary, pred_type, seq_l, pre_len, fold_count=6):
    """
    Aggregate metrics from all folds.
    
    Args:
        model_name: Name of the model (e.g., 'gwnet')
        city: City code (e.g., 'MEL')
        feature: Feature name (e.g., 'volume')
        auxiliary: Auxiliary features (e.g., 'none')
        pred_type: Prediction type (e.g., 'site')
        seq_l: Sequence length
        pre_len: Prediction length
        fold_count: Number of folds (default: 6)
    """
    
    base_dir = f'result/{city}/{model_name}/{feature}-{auxiliary}-{pred_type}-{seq_l}-{pre_len}'
    
    all_metrics = {
        'MAE': [],
        'RMSE': [],
        'RAE': [],
        'MedAE': [],
        'R²': [],
        'EVS': []
    }
    
    # Collect metrics from all folds
    for fold in range(1, fold_count + 1):
        fold_dir = f"{base_dir}-{fold}"
        
        # Find if there's a directory with # suffix (e.g., folder#1)
        if not os.path.exists(fold_dir):
            for suffix in range(1, 100):
                alt_dir = f"{base_dir}-{fold}#{suffix}"
                if os.path.exists(alt_dir):
                    fold_dir = alt_dir
                    break
        
        metrics_file = os.path.join(fold_dir, 'metrics.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                for key in all_metrics.keys():
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
            print(f"Fold {fold}: {fold_dir}")
        else:
            print(f"Warning: Metrics file not found for fold {fold}: {metrics_file}")
    
    # Compute mean and std
    print("\n" + "="*60)
    print(f"Cross-Validation Results ({fold_count}-fold)")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"City: {city}")
    print(f"Config: seq_l={seq_l}, pre_len={pre_len}, feature={feature}")
    print("="*60)
    print(f"{'Metric':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-"*60)
    
    results = {}
    for metric_name, values in all_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            results[metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(min_val),
                'max': float(max_val),
                'values': values
            }
            
            print(f"{metric_name:<10} {mean_val:>12.6f} {std_val:>12.6f} {min_val:>12.6f} {max_val:>12.6f}")
    
    print("="*60 + "\n")
    
    # Save aggregated results
    output_file = f"{base_dir}/cv_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to: {output_file}\n")
    
    return results


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Usage: python aggregate_results.py <model> <city> <feature> <auxiliary> <pred_type> [seq_l] [pre_len] [fold_count]")
        print("\nExample:")
        print("  python aggregate_results.py gwnet MEL volume none site 12 1 6")
        sys.exit(1)
    
    model = sys.argv[1]
    city = sys.argv[2]
    feature = sys.argv[3]
    auxiliary = sys.argv[4]
    pred_type = sys.argv[5]
    seq_l = sys.argv[6] if len(sys.argv) > 6 else "12"
    pre_len = sys.argv[7] if len(sys.argv) > 7 else "1"
    fold_count = int(sys.argv[8]) if len(sys.argv) > 8 else 6
    
    aggregate_metrics(model, city, feature, auxiliary, pred_type, seq_l, pre_len, fold_count)
