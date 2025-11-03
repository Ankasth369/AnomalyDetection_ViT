# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import numpy as np
import os
import time
from sklearn.model_selection import KFold

try:
    from src.utils.data_loader import TimeSeriesDataset 
    from models.ViTAdapterModel import ViTAdapterAnomalyModel
    from src.utils.metrics import point_adjust_f1_score, calculate_auprc
except ImportError:
    print("Error: Could not import project modules.")
    print("If running in Kaggle, make sure your .py files are in /kaggle/working/")
    print("You may need to add: import sys; sys.path.append('/kaggle/working/')")
    exit()

# KAGGLE_WORKING_DIR = "/kaggle/working/" 
KAGGLE_WORKING_DIR = "" 

BASE_PROCESSED_PATH = os.path.join(KAGGLE_WORKING_DIR, "data/processed/")

# Per-dataset subsampling rates (Your requested settings)
# --- NOTE: Your log shows 1.0% for MIT-BIH, so you've likely changed this ---
DATASET_SUBSET_PERCENT = {
    "SMD_train": 1.0,      # 100% of SMD
    "Financial": 0.5,      # 50% of Financial
    "MIT-BIH": 0.1         # 10% of MIT-BIH 
}

# Model & Training parameters (GLOBAL)
TIME_WINDOW_LEN = 96
STEP_SIZE = 48
BATCH_SIZE = 32 
NUM_EPOCHS = 10 
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1 
RANDOM_SEED = 42
MAX_VARS = 38 
OUTPUT_SPATIAL_SIZE_DEFAULT = 224
NUM_WORKERS = 2 

CHECKPOINT_DIR = os.path.join(KAGGLE_WORKING_DIR, "models/trained_weights/")


def run_training_job(train_data_path: str, use_data_subset_percent: float, checkpoint_name: str, job_name: str):
    """
    Main training loop for the ViT-Adapter anomaly detection model.
    Uses the "FAST" loader.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading data from {train_data_path}...")
    try:
        # --- CRITICAL FIX: Pass subset percent to loader ---
        train_dataset = TimeSeriesDataset(
            data_path=train_data_path,
            window_size=TIME_WINDOW_LEN,
            step_size=STEP_SIZE,
            max_vars=MAX_VARS, 
            output_spatial_size=OUTPUT_SPATIAL_SIZE_DEFAULT, 
            train_mode=True,
            file_subsample_percent=use_data_subset_percent # <-- FIX
        )
        # --- END FIX ---
    except Exception as e:
        print(f"Error loading dataset {job_name}: {e}")
        return 

    dataset_size = len(train_dataset)
    all_indices = list(range(dataset_size))
    
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_indices)
    
    # --- CRITICAL FIX: Remove redundant window subsampling ---
    # Subsampling is now done at the FILE level in the data loader
    # to prevent the OOM crash. We now use 100% of the windows
    # from that (already subsampled) set of files.
    
    # (Old logic commented out)
    # subset_size = int(np.floor(use_data_subset_percent * dataset_size))
    # if subset_size < 10:
    # ...
    # subset_indices = all_indices[:subset_size]
    
    # --- NEW LOGIC ---
    subset_indices = all_indices # Use all windows from the loaded files
    print(f"--- INFO: Using 100% of windows from the {use_data_subset_percent * 100:.1f}% subsampled files. ---")
    print(f"Total windows to be used: {len(subset_indices)}")
    # --- END NEW LOGIC ---
    
    split = int(np.floor(VALIDATION_SPLIT * len(subset_indices)))
    
    if split == 0 and len(subset_indices) > 1:
        split = 1
        
    if split == 0:
        print(f"--- WARNING: Not enough data for validation split. Training on all {len(subset_indices)} samples. ---")
        train_indices = subset_indices
        val_indices = []
    else:
        train_indices, val_indices = subset_indices[split:], subset_indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Data loaded: {len(train_indices)} training windows, {len(val_indices)} validation windows.")

    model = ViTAdapterAnomalyModel(
        time_window_len=TIME_WINDOW_LEN,
        max_vars=MAX_VARS
    ).to(device)
    
    model.freeze_vit_backbone()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Successfully resumed from checkpoint. Starting at Epoch {start_epoch}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")

    print(f"--- Starting Model Training for {job_name} ---")
    start_time_total = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        
        # --- Training loop (Unchanged, already correct) ---
        for i, (image_batch, original_window_batch, labels, num_vars_batch) in enumerate(train_loader):
            
            image_batch = image_batch.to(device)
            target = original_window_batch.to(device) 
            
            optimizer.zero_grad()
            reconstructed = model(image_batch) 
            
            M_i = num_vars_batch[0].item()
            loss = criterion(reconstructed[:, :, :M_i], target[:, :, :M_i])
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        if len(train_loader) > 0:
            avg_train_loss = total_train_loss / len(train_loader)
        else:
            avg_train_loss = 0.0

        avg_val_loss = 0.0
        if len(val_loader) > 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                # --- Validation loop (Unchanged, already correct) ---
                for image_batch, original_window_batch, labels, num_vars_batch in val_loader:
                    
                    image_batch = image_batch.to(device)
                    target = original_window_batch.to(device)
                    
                    reconstructed = model(image_batch)
                    
                    M_i = num_vars_batch[0].item()
                    loss = criterion(reconstructed[:, :, :M_i], target[:, :, :M_i])
                    total_val_loss += loss.item()

            if len(val_loader) > 0:
                avg_val_loss = total_val_loss / len(val_loader)
        
        epoch_end_time = time.time()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Time: {epoch_end_time - epoch_start_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            print(f"  Val loss improved ({best_val_loss:.6f} -> {avg_val_loss:.6f}). Saving model to {checkpoint_path}")
            best_val_loss = avg_val_loss
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'time_window_len': TIME_WINDOW_LEN,
                    'max_vars': MAX_VARS
                }, checkpoint_path)
            except Exception as e:
                print(f"  [ERROR] Could not save checkpoint: {e}")
        
    total_training_time = time.time() - start_time_total
    print(f"--- Training Finished for {job_name} ---")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    # import sys
    # sys.path.append('/kaggle/working/')
    
    # --- Jobs (Unchanged, your local edits will apply) ---
    training_jobs = [
         {
             "name": "SMD_train",
             "path": os.path.join(BASE_PROCESSED_PATH, "SMD_train/"),
             "percent": DATASET_SUBSET_PERCENT.get("SMD_train", 1.0), 
             "checkpoint": "adapter_smd.pth"
         },
         {
             "name": "Financial_train",
             "path": os.path.join(BASE_PROCESSED_PATH, "Financial", "financial_train.csv"),
             "percent": DATASET_SUBSET_PERCENT.get("Financial", 0.5),
             "checkpoint": "adapter_financial.pth"
         },
        {
            "name": "MIT-BIH_train",
            "path": os.path.join(BASE_PROCESSED_PATH, "MIT-BIH_train/"),
            "percent": DATASET_SUBSET_PERCENT.get("MIT-BIH", 0.1), 
            "checkpoint": "adapter_mitbih.pth"
        }
    ]

    for job in training_jobs:
        print("\n" + "="*50)
        print(f"STARTING TRAINING JOB: {job['name']}")
        print(f"  -> Path: {job['path']}")
        print(f"  -> Subsample: {job['percent']*100}%")
        print("="*50 + "\n")
        
        run_training_job(
            train_data_path=job['path'],
            use_data_subset_percent=job['percent'],
            checkpoint_name=job['checkpoint'],
            job_name=job['name']
        )
        
        print(f"\nFINISHED TRAINING JOB: {job['name']}\n")

    print("All training jobs complete.")