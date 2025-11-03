# evaluate_model.py
import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Import Project Modules ---
try:
    # --- This must be the "fast" loader with the adapter logic inside ---
    from src.utils.data_loader import TimeSeriesDataset
    
    # --- CRITICAL FIX: Import the new threshold function ---
    from src.utils.metrics import (
        point_adjust_f1_score, 
        calculate_auprc, 
        calculate_auroc,
        find_best_point_adjust_f1_threshold  # <-- NEW
    )
    # --- END FIX ---
    
    from models.ViTAdapterModel import ViTAdapterAnomalyModel
except ImportError as e:
    print(f"Module Import Error: {e}")
    print("Please ensure your project structure is correct and added to sys.path if needed.")
    exit()

# Set this to your Kaggle environment's working directory if needed
# KAGGLE_WORKING_DIR = "/kaggle/working/" 
KAGGLE_WORKING_DIR = "" 

# --- Configuration Constants ---
BASE_PROCESSED_PATH = os.path.join(KAGGLE_WORKING_DIR, "data/processed/")
CHECKPOINT_DIR = os.path.join(KAGGLE_WORKING_DIR, "models/trained_weights/")
LOG_DIR = os.path.join(KAGGLE_WORKING_DIR, "logs/")
LOG_FILE_PATH = os.path.join(LOG_DIR, "evaluation_metrics.txt")

# Evaluation parameters
STEP_SIZE = 12 # Use a smaller step size for smoother score reconstruction
BATCH_SIZE = 128
NUM_WORKERS = 2 # Use more workers for "fast" loader

# --- FIX: Add new global constant ---
OUTPUT_SPATIAL_SIZE_DEFAULT = 224


def run_evaluation_job(test_data_path: str, model_weights_path: str, job_name: str):
    """
    Main function to run the evaluation pipeline.
    --- UPDATED FOR POINT-WISE METRIC CALCULATION (Optimized for PA-F1) ---
    """
    start_job_time = time.time()
    os.makedirs(LOG_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating dataset: {job_name}")

    # --- 1. Load Model ---
    print(f"Loading model from {model_weights_path}...")
    try:
        checkpoint = torch.load(model_weights_path, map_location=device)
        
        TIME_WINDOW_LEN = checkpoint.get('time_window_len', 96)
        MAX_VARS = checkpoint.get('max_vars', 38)

        # This must be the "fast" model (no adapter in forward)
        model = ViTAdapterAnomalyModel(
            time_window_len=TIME_WINDOW_LEN,
            max_vars=MAX_VARS
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")
        print(f"  -> Model Config: T={TIME_WINDOW_LEN}, M_max={MAX_VARS}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_weights_path}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- 2. Load Test Data ---
    print(f"Loading test data from {test_data_path}...")
    try:
        # --- CRITICAL FIX: Call "fast" loader with all args ---
        test_dataset = TimeSeriesDataset(
            data_path=test_data_path,
            window_size=TIME_WINDOW_LEN,
            step_size=STEP_SIZE,
            max_vars=MAX_VARS, # <-- This fixes the TypeError
            output_spatial_size=OUTPUT_SPATIAL_SIZE_DEFAULT,
            train_mode=False 
        )
        # --- END FIX ---
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"Test data loaded: {len(test_dataset)} evaluation windows.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    criterion = nn.MSELoss(reduction='none') # We need per-element error

    # --- 3. Evaluation Loop (Get Anomaly Scores) ---
    print("Running inference to get anomaly scores...")
    
    # --- CRITICAL FIX: Create arrays for POINT-WISE metrics ---
    # Get the full ground-truth label array from the dataset
    y_true_pointwise = test_dataset.labels.astype(int)
    # Create an array to store the reconstructed scores
    y_scores_pointwise = np.zeros_like(y_true_pointwise, dtype=float)
    # Create an array to count overlaps
    point_counts = np.zeros_like(y_true_pointwise, dtype=int)
    # --------------------------------------------------------

    inference_start_time = time.time()
    with torch.no_grad():
        # --- FIX: "fast" loader returns 4 items ---
        for i, (image_tensors, target_tensors, labels, num_vars_batch) in enumerate(tqdm(test_loader, desc=f"Evaluating {job_name}")):
            
            image_tensors = image_tensors.to(device)
            target_tensors = target_tensors.to(device)
            M_i = num_vars_batch[0].item() 
            
            # Model takes the image tensor
            reconstructed = model(image_tensors) 
            
            # Loss is against the target tensor
            # Calculate error for each window (B, T, M_i)
            error_matrix = criterion(reconstructed[..., :M_i], target_tensors[..., :M_i])
            # Average error across variables and time steps to get one score per window
            window_scores_batch = torch.mean(error_matrix, dim=(1, 2)).cpu().numpy()
            
            # --- CRITICAL FIX: Reconstruct point-wise scores ---
            # Get the start indices for each window in this batch
            batch_start_index = i * BATCH_SIZE
            
            for j in range(len(window_scores_batch)):
                window_score = window_scores_batch[j]
                
                # Get the window's (start, end) indices in the full dataset
                window_start, window_end = test_dataset.indices[batch_start_index + j]
                
                # Add this window's score to all points it covers
                y_scores_pointwise[window_start:window_end] += window_score
                # Increment the count for all points it covers
                point_counts[window_start:window_end] += 1
            # ----------------------------------------------------

    inference_end_time = time.time()
    
    # --- CRITICAL FIX: Average scores for overlapping windows ---
    # Avoid division by zero for points not covered by any window
    point_counts[point_counts == 0] = 1
    y_scores_pointwise = y_scores_pointwise / point_counts
    # ----------------------------------------------------------
    
    # Use the arrays from the start and end of the dataset, where scores are 0
    # but labels might be 1. This is the correct, full point-wise evaluation.
    y_true = y_true_pointwise
    y_scores = y_scores_pointwise
    
    print(f"Inference complete. Time taken: {inference_end_time - inference_start_time:.2f}s")
    
    if len(y_true) == 0:
        print("Error: No data was loaded or processed. Aborting evaluation.")
        return

    # --- 4. Calculate Metrics ---
    print("Calculating performance metrics (Optimizing for Point-Adjusted F1)...")
    
    # --- CRITICAL FIX: Call the new function ---
    # Call the new function to find the best threshold for PA-F1
    best_threshold, pa_f1, pa_precision, pa_recall = find_best_point_adjust_f1_threshold(y_scores, y_true)

    # (Optional) Now that we have the best threshold, we can calculate
    # the other metrics at that same threshold for comparison.
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    y_pred_at_best_pa_threshold = (y_scores > best_threshold).astype(int)
    
    best_f1 = f1_score(y_true, y_pred_at_best_pa_threshold, zero_division=0)
    precision = precision_score(y_true, y_pred_at_best_pa_threshold, zero_division=0)
    recall = recall_score(y_true, y_pred_at_best_pa_threshold, zero_division=0)
    
    auroc = calculate_auroc(y_scores, y_true)
    auprc = calculate_auprc(y_scores, y_true)
    # --- END FIX ---

    # --- 5. Log Results ---
    job_end_time = time.time()
    
    log_output = f"""
    =================================================
    Anomaly Detection Evaluation Results
    Model: {os.path.basename(model_weights_path)}
    Dataset: {job_name}
    Timestamp: {time.ctime()}
    =================================================
    
    [Execution Summary]
    Total Evaluation Time: {job_end_time - start_job_time:.2f}s
    Inference Time: {inference_end_time - inference_start_time:.2f}s
    Total Time Points: {len(y_true)}
    Total Windows: {len(test_dataset)}
    
    [Overall Performance (Point-wise)]
    Area Under ROC Curve (AUC-ROC): {auroc:.6f}
    Area Under PR Curve (AUC-PR): {auprc:.6f}
    
    [Metrics @ Optimal PA-F1 Threshold ({best_threshold:.6f})]
    Point-Adjusted F1 (PA-F1):    {pa_f1:.6f}
    Point-Adjusted Precision: {pa_precision:.6f}
    Point-Adjusted Recall:    {pa_recall:.6f}
    
    (Standard Metrics at this threshold)
    Standard F1-Score:    {best_f1:.6f}
    Standard Precision: {precision:.6f}
    Standard Recall:    {recall:.6f}
    
    =================================================
    """
    
    print(log_output)
    
    try:
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(log_output)
        print(f"Results appended to {LOG_FILE_PATH}")
    except Exception as e:
        print(f"Error writing to log file: {e}")


if __name__ == "__main__":
    # import sys
    # sys.path.append('/kaggle/working/')
    
    # --- Define All Evaluation Jobs ---
    # --- FIX: Point to the new _test files/folders ---
    evaluation_jobs = [
         {
             "name": "SMD_test",
             "path": os.path.join(BASE_PROCESSED_PATH, "SMD_test/"),
             "model": os.path.join(CHECKPOINT_DIR, "adapter_smd.pth")
         },
        {
            "name": "Financial_test", 
            "path": os.path.join(BASE_PROCESSED_PATH, "Financial", "financial_test.csv"),
            "model": os.path.join(CHECKPOINT_DIR, "adapter_financial.pth")
        },
         {
             "name": "MIT-BIH_test", 
             "path": os.path.join(BASE_PROCESSED_PATH, "MIT-BIH_test/"),
             "model": os.path.join(CHECKPOINT_DIR, "adapter_mitbih.pth")
         }
    ]
    # --- END FIX ---

    # --- Run All Jobs Sequentially ---
    for job in evaluation_jobs:
        print("\n" + "="*50)
        print(f"STARTING EVALUATION JOB: {job['name']}")
        print(f"  -> Model: {job['model']}")
        print(f"  -> Data: {job['path']}")
        print("="*50 + "\n")
             
        run_evaluation_job(
            test_data_path=job['path'],
            model_weights_path=job['model'],
            job_name=job['name']
        )
        
        print(f"\nFINISHED EVALUATION JOB: {job['name']}\n")

    print("All evaluation jobs complete.")