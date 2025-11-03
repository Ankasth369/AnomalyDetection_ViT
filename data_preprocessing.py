# data_preprocessing.py
import os
import glob
import pandas as pd
import numpy as np
import time 
from sklearn.preprocessing import MinMaxScaler
from typing import List
import wfdb # We need this for the MIT-BIH processing

# --- Configuration Constants (Must match project constants) ---
MAX_VARS = 38
OUTPUT_DIR = "data/processed/"
TRAIN_TEST_SPLIT_RATIO = 0.8 # 80% for training, 20% for testing
RANDOM_SEED = 42

# --- Dataset-Specific Configurations ---
FINANCIAL_FEATURES = [
    'amount', 'time_since_last_transaction', 
    'spending_deviation_score', 'velocity_score', 
    'geo_anomaly_score'
]
FINANCIAL_LABEL = 'is_fraud'
FINANCIAL_TRAIN_NAME = 'financial_train.csv'
FINANCIAL_TEST_NAME = 'financial_test.csv'

MITBIH_RAW_DIR_NAME = 'mit-bih-arrhythmia-database'
MITBIH_NORMAL_BEATS = ['N', 'L', 'R', 'e', 'j', '.'] 

SMD_RAW_DIR_NAME = 'ServerMachineDataset'
SMD_LABEL_COL_NAME = 'anomaly' 

def check_checkpoint(output_path: str) -> bool:
    """Checks if the final processed file already exists."""
    if os.path.exists(output_path):
        print(f"[CHECKPOINT] File already exists: {output_path}. Skipping processing.")
        return True
    return False

def _pad_dataframe(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Pads a dataframe with zero columns to match MAX_VARS."""
    # Check if label column exists and handle it
    label_col_present = SMD_LABEL_COL_NAME in df.columns
    
    padding_cols_count = MAX_VARS - len(features)
    if padding_cols_count > 0:
        padding_df = pd.DataFrame(np.zeros((df.shape[0], padding_cols_count)), 
                                  columns=[f'padding_{i+1}' for i in range(padding_cols_count)],
                                  index=df.index)
        
        feature_df = df[features]
        
        if label_col_present:
            label_col = df[[SMD_LABEL_COL_NAME]]
            df = pd.concat([feature_df, padding_df, label_col], axis=1)
        else:
            df = pd.concat([feature_df, padding_df], axis=1)
            
    # Ensure correct column order even if no padding was needed
    elif label_col_present:
         df = pd.concat([df[features], df[[SMD_LABEL_COL_NAME]]], axis=1)
    else:
         df = df[features]
         
    return df

def load_and_preprocess_financial(raw_file_path: str, output_subdir: str):
    """
    --- UPDATED ---
    Loads Financial CSV, splits into train/test, pads, and saves two files.
    """
    output_path_train = os.path.join(OUTPUT_DIR, output_subdir, FINANCIAL_TRAIN_NAME)
    output_path_test = os.path.join(OUTPUT_DIR, output_subdir, FINANCIAL_TEST_NAME)
    
    if check_checkpoint(output_path_train) and check_checkpoint(output_path_test):
        return
        
    start_time = time.time()
    print(f"--- START Processing Financial data: {raw_file_path} ---")
    
    try:
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        print(f"[ERROR] Error reading Financial CSV: {e}")
        return

    selected_cols = FINANCIAL_FEATURES + [FINANCIAL_LABEL]
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Financial CSV missing required columns: {missing_cols}")
        return

    df = df[selected_cols].copy()
    
    df[FINANCIAL_LABEL] = df[FINANCIAL_LABEL].replace({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0}).astype(int)
    df.rename(columns={FINANCIAL_LABEL: SMD_LABEL_COL_NAME}, inplace=True)
    
    # Pad the dataframe
    df = _pad_dataframe(df, FINANCIAL_FEATURES)
    
    # --- CRITICAL FIX: Split into Train and Test Chronologically ---
    # Shuffling time series data (df.sample(frac=1)) causes data leakage.
    # We must split by time order.
    
    # Find split index
    split_idx = int(len(df) * TRAIN_TEST_SPLIT_RATIO)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    # --- END FIX ---
    
    # Save both files
    os.makedirs(os.path.join(OUTPUT_DIR, output_subdir), exist_ok=True)
    train_df.to_csv(output_path_train, index=False)
    test_df.to_csv(output_path_test, index=False)
    
    end_time = time.time()
    print(f"Successfully processed and split Financial data:")
    print(f"  -> Train set: {output_path_train} ({len(train_df)} rows)")
    print(f"  -> Test set: {output_path_test} ({len(test_df)} rows)")
    print(f"[TIME] Financial Processing took: {end_time - start_time:.2f} seconds.")


def load_and_preprocess_mitdb(raw_file_dir: str, output_subdir_train: str, output_subdir_test: str, record_list: List[str]):
    """
    --- UPDATED ---
    Splits records 80/20 and processes them into train and test folders.
    """
    if wfdb is None:
        print("[SKIP] Skipping MIT-BIH processing as 'wfdb' library is not available.")
        return

    # --- NEW: Split records into train/test lists ---
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(record_list)
    split_idx = int(len(record_list) * TRAIN_TEST_SPLIT_RATIO)
    train_records = record_list[:split_idx]
    test_records = record_list[split_idx:]
    
    print(f"--- START Processing MIT-BIH data from {raw_file_dir} ---")
    print(f"  -> {len(train_records)} records for TRAIN ({output_subdir_train})")
    print(f"  -> {len(test_records)} records for TEST ({output_subdir_test})")

    # Process both lists
    _process_mitdb_list(raw_file_dir, output_subdir_train, train_records, "Train")
    _process_mitdb_list(raw_file_dir, output_subdir_test, test_records, "Test")

def _process_mitdb_list(raw_file_dir: str, output_subdir: str, record_list: List[str], job_name: str):
    """Helper function to process a list of MIT-BIH records and save them."""
    
    start_time = time.time()
    os.makedirs(os.path.join(OUTPUT_DIR, output_subdir), exist_ok=True)
    processed_count = 0

    for record_name in record_list:
        output_path = os.path.join(OUTPUT_DIR, output_subdir, f"{record_name}_processed.csv")
        if check_checkpoint(output_path):
            continue
            
        record_path = os.path.join(raw_file_dir, record_name)
        
        try:
            signals, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            if signals.shape[1] < 2:
                print(f"  -> [SKIP] Record {record_name} has less than 2 signals.")
                continue
                
            lead_names = [name.upper() for name in fields['sig_name'][:2]]
            signal_df = pd.DataFrame(signals[:, :2], columns=lead_names)

            anomaly_labels = np.zeros(signals.shape[0])
            
            abnormal_indices = [
                ann_sample for ann_sample, ann_symbol in zip(annotation.sample, annotation.symbol) 
                if ann_symbol not in MITBIH_NORMAL_BEATS
            ]
            
            valid_abnormal_indices = [idx for idx in abnormal_indices if idx < len(anomaly_labels)]
            anomaly_labels[valid_abnormal_indices] = 1
            
            signal_df[SMD_LABEL_COL_NAME] = anomaly_labels
            
            # Pad the dataframe
            final_df = _pad_dataframe(signal_df, lead_names)
            
            final_df.to_csv(output_path, index=False)
            processed_count += 1

        except Exception as e:
            print(f"  -> [ERROR] Failed to process record {record_name}: {e}")

    end_time = time.time()
    print(f"Successfully processed {processed_count} MIT-BIH {job_name} records.")
    print(f"[TIME] MIT-BIH {job_name} Processing took: {end_time - start_time:.2f} seconds.")


def load_and_preprocess_smd_train(raw_data_dir: str, output_subdir: str):
    """ Loads SMD training data, assumes all normal (label=0). """
    
    start_time_total = time.time()
    print(f"--- START Processing SMD (Train) data from {raw_data_dir} ---")
    
    data_files = sorted(glob.glob(os.path.join(raw_data_dir, '*.txt')))
    os.makedirs(os.path.join(OUTPUT_DIR, output_subdir), exist_ok=True)
    
    processed_count = 0
    
    for data_path in data_files:
        file_name = os.path.basename(data_path)
        output_path = os.path.join(OUTPUT_DIR, output_subdir, file_name.replace('.txt', '.csv'))

        if check_checkpoint(output_path):
            continue

        start_time_file = time.time()
        
        try:
            data_df = pd.read_csv(data_path, sep=',', header=None, engine='python')
            
            if data_df.shape[1] != MAX_VARS:
                print(f"[SKIP] {file_name}: Expected {MAX_VARS} features, found {data_df.shape[1]}")
                continue
            
            data_df.columns = [f'feat_{i}' for i in range(MAX_VARS)]
            data_df[SMD_LABEL_COL_NAME] = 0
            data_df.to_csv(output_path, index=False)
            
            end_time_file = time.time()
            print(f"  -> Processed {file_name} (Train) in {end_time_file - start_time_file:.2f}s")
            processed_count += 1

        except Exception as e:
            print(f"  -> [ERROR] Failed to process {file_name}: {e}")

    end_time_total = time.time()
    print(f"SMD (Train) Processing Summary: {processed_count} files processed/updated.")
    print(f"[TIME] Total SMD (Train) Processing took: {end_time_total - start_time_total:.2f} seconds.")


def load_and_preprocess_smd_test(raw_data_dir: str, raw_labels_dir: str, output_subdir: str):
    """ Loads SMD test data and merges with separate label files. """
    
    start_time_total = time.time()
    print(f"--- START Processing SMD (Test) data from {raw_data_dir} ---")
    
    data_files = sorted(glob.glob(os.path.join(raw_data_dir, '*.txt')))
    label_files_map = {os.path.basename(f): f for f in glob.glob(os.path.join(raw_labels_dir, '*.txt'))}

    os.makedirs(os.path.join(OUTPUT_DIR, output_subdir), exist_ok=True)
    
    processed_count = 0
    
    for data_path in data_files:
        # --- FIX: Corrected the variable name from file_.name to file_name ---
        file_name = os.path.basename(data_path)
        # --- END FIX ---
        
        output_path = os.path.join(OUTPUT_DIR, output_subdir, file_name.replace('.txt', '.csv'))

        if file_name not in label_files_map:
            print(f"[SKIP] {file_name}: No matching label file found in {raw_labels_dir}")
            continue
            
        if check_checkpoint(output_path):
            continue

        start_time_file = time.time()
        label_path = label_files_map[file_name]
        
        try:
            data_df = pd.read_csv(data_path, sep=',', header=None, engine='python')
            label_df = pd.read_csv(label_path, header=None, engine='python')
            
            if data_df.shape[1] != MAX_VARS:
                print(f"[SKIP] {file_name}: Expected {MAX_VARS} features, found {data_df.shape[1]}")
                continue
            
            if len(data_df) != len(label_df):
                print(f"[SKIP] {file_name}: Data ({len(data_df)}) and Label ({len(label_df)}) length mismatch.")
                continue

            data_df.columns = [f'feat_{i}' for i in range(MAX_VARS)]
            data_df[SMD_LABEL_COL_NAME] = label_df[0].values
            data_df.to_csv(output_path, index=False)
            
            end_time_file = time.time()
            print(f"  -> Processed {file_name} (Test) in {end_time_file - start_time_file:.2f}s")
            processed_count += 1

        except Exception as e:
            print(f"  -> [ERROR] Failed to process {file_name}: {e}")

    end_time_total = time.time()
    print(f"SMD (Test) Processing Summary: {processed_count} files processed/updated.")
    print(f"[TIME] Total SMD (Test) Processing took: {end_time_total - start_time_total:.2f} seconds.")

def run_all_preprocessing(raw_base_dir: str):
    """Main function to orchestrate all preprocessing steps."""
    
    print(f"Starting Preprocessing Job. Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. SMD Data (Already split) ---
    smd_raw_base = os.path.join(raw_base_dir, SMD_RAW_DIR_NAME)
    load_and_preprocess_smd_train(os.path.join(smd_raw_base, 'train'), 'SMD_train')
    load_and_preprocess_smd_test(os.path.join(smd_raw_base, 'test'), os.path.join(smd_raw_base, 'test_label'), 'SMD_test')
    
    # --- 2. Financial Data (Split 80/20) ---
    financial_raw_path = os.path.join(raw_base_dir, 'Financial_fraud_dataset', 'financial_fraud_detection_dataset.csv')
    load_and_preprocess_financial(financial_raw_path, 'Financial')

    # --- 3. MIT-BIH Data (Split 80/20) ---
    mitdb_raw_dir = os.path.join(raw_base_dir, MITBIH_RAW_DIR_NAME)
    
    if os.path.exists(mitdb_raw_dir) and wfdb is not None:
        mitdb_records = sorted(list(set([
            f.split('.')[0] for f in os.listdir(mitdb_raw_dir) 
            if f.endswith('.dat')
        ])))
        
        if not mitdb_records:
            print(f"[WARN] No .dat files found in {mitdb_raw_dir}. Skipping MIT-BIH.")
        else:
            load_and_preprocess_mitdb(mitdb_raw_dir, 'MIT-BIH_train', 'MIT-BIH_test', mitdb_records)
    elif wfdb is None:
         print("[SKIP] Skipping MIT-BIH processing as 'wfdb' library is not available.")
    else:
        print(f"[WARN] MIT-BIH raw directory not found: {mitdb_raw_dir}. Skipping MIT-BIH.")
    
    print("\n--- Preprocessing Job Finished ---")


if __name__ == "__main__":
    # Add this to your Kaggle notebook cell if you get import errors:
    # import sys
    # sys.path.append('/kaggle/working/') # Or your project root
    
    # Set the path where your raw data folders (SMD, Financial, MIT-BIH) are stored.
    # For Colab/Kaggle, this might be: 
    # RAW_BASE_DIRECTORY = '/content/drive/MyDrive/Latest Project/data/raw' 
    RAW_BASE_DIRECTORY = 'data/raw' 
    
    run_all_preprocessing(RAW_BASE_DIRECTORY)