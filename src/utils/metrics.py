# metrics.py
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from typing import Tuple

def point_adjust_f1_score(y_true: np.ndarray, y_pred: np.ndarray, buffer: int = 0) -> Tuple[float, float, float]:
    """
    Calculates the Point-Adjusted F1, Precision, and Recall Scores.
    
    This metric ensures that if a detector flags ANY point within a true anomaly 
    segment, the entire segment is considered detected (True Positive).

    Returns:
        tuple: (pa_f1, pa_precision, pa_recall)
    """
    
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    
    is_anomaly = y_true == 1
    diff = np.diff(is_anomaly.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    if len(starts) != len(ends):
        # Handle edge cases where anomalies start at 0 or end at the last point
        if len(starts) > len(ends) and starts[0] == 0:
             ends = np.append(ends, len(y_true) - 1)
        elif len(ends) > len(starts) and ends[0] < (starts[0] if len(starts) > 0 else len(y_true)):
             starts = np.insert(starts, 0, 0)
        
        # Final trim to ensure equal length
        min_len = min(len(starts), len(ends))
        starts = starts[:min_len]
        ends = ends[:min_len]

    true_segments = list(zip(starts, ends))
    
    tp = 0 
    fp = 0 
    
    y_pred_mask = y_pred.copy().astype(bool)

    # --- Step 1: Count True Positives (Segments) ---
    for i, (start, end) in enumerate(true_segments):
        search_start = max(0, start - buffer)
        search_end = min(end + buffer + 1, len(y_pred))
        
        # Find all predictions within the buffered segment
        segment_hits_indices = np.where(y_pred_mask[search_start:search_end])[0] + search_start
        
        if len(segment_hits_indices) > 0:
            tp += 1
            # "Consume" these predictions so they can't count as FPs
            y_pred_mask[segment_hits_indices] = False 
            
    # --- Step 2: Count False Positives (Remaining Predicted Points) ---
    # Any remaining 'True' in y_pred_mask is a prediction
    # that did not fall into any true anomaly segment.
    fp = np.sum(y_pred_mask)
    
    # --- Step 3: Count False Negatives (Missed Segments) ---
    fn = len(true_segments) - tp
    
    # --- Step 4: Calculate Metrics ---
    pa_precision = tp / (tp + fp + 1e-10)
    pa_recall = tp / (tp + fn + 1e-10)
    pa_f1 = 2 * (pa_precision * pa_recall) / (pa_precision + pa_recall + 1e-10)
    
    return pa_f1, pa_precision, pa_recall


def calculate_auprc(anomaly_scores: np.ndarray, y_true_binary: np.ndarray) -> float:
    """
    Calculates the Area Under the Precision-Recall Curve (AUC-PR).
    """
    if len(np.unique(y_true_binary)) < 2:
        print(f"Warning: Only one class present in y_true (class: {np.unique(y_true_binary)}). AUPRC set to 0.0")
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true_binary, anomaly_scores)
    return auc(recall, precision)


def calculate_auroc(anomaly_scores: np.ndarray, y_true_binary: np.ndarray) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
    """
    if len(np.unique(y_true_binary)) < 2:
        print(f"Warning: Only one class present in y_true (class: {np.unique(y_true_binary)}). AUROC set to 0.0")
        return 0.0
    return roc_auc_score(y_true_binary, anomaly_scores)


def find_best_f1_threshold(anomaly_scores: np.ndarray, y_true_binary: np.ndarray, num_steps: int = 100) -> Tuple[float, float, float, float]:
    """
    Finds the optimal anomaly threshold to maximize the standard F1 score.

    Returns:
        tuple: (best_threshold, best_f1_score, best_precision, best_recall)
    """
    best_f1 = 0.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    min_score, max_score = np.min(anomaly_scores), np.max(anomaly_scores)
    
    if max_score == min_score:
        # Avoids division by zero in linspace if all scores are identical
        return 0.0, 0.0, 0.0, 0.0
        
    thresholds = np.linspace(min_score, max_score, num_steps)
    
    for threshold in thresholds:
        y_pred_binary = (anomaly_scores > threshold).astype(int)
        
        # Use zero_division=0 to handle cases where no predictions are made
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            best_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            
    return best_threshold, best_f1, best_precision, best_recall


def calculate_variable_contributions(original_window: np.ndarray, 
                                     reconstructed_window: np.ndarray) -> np.ndarray:
    """
    Calculates the contribution of each variable (channel) to the total
    reconstruction error for a single window. Used for localization.
    
    Args:
        original_window (np.ndarray): Shape (T, M_max)
        reconstructed_window (np.ndarray): Shape (T, M_max)
        
    Returns:
        np.ndarray: Shape (M_max,)
    """
    # Calculate squared error for each point: (T, M_max)
    error_matrix = (original_window - reconstructed_window) ** 2
    
    # Sum error over time to get total error per variable: (M_max,)
    error_per_variable = np.sum(error_matrix, axis=0)
    
    # Get total error for the entire window
    total_window_error = np.sum(error_per_variable)
    
    if total_window_error < 1e-10:
        return np.zeros(original_window.shape[1])
        
    # Normalize to get percentage contribution
    contributions = error_per_variable / total_window_error
    
    return contributions


# --- NEW FUNCTION ADDED ---
def find_best_point_adjust_f1_threshold(anomaly_scores: np.ndarray, y_true_binary: np.ndarray, num_steps: int = 100) -> Tuple[float, float, float, float]:
    """
    Finds the optimal anomaly threshold to maximize the POINT-ADJUSTED F1 score.

    Returns:
        tuple: (best_threshold, best_pa_f1, best_pa_precision, best_pa_recall)
    """
    best_pa_f1 = 0.0
    best_threshold = 0.0
    best_pa_precision = 0.0
    best_pa_recall = 0.0
    
    min_score, max_score = np.min(anomaly_scores), np.max(anomaly_scores)
    
    if max_score == min_score:
        # Avoids division by zero in linspace if all scores are identical
        return 0.0, 0.0, 0.0, 0.0
        
    thresholds = np.linspace(min_score, max_score, num_steps)
    
    # Check for empty y_true
    if len(y_true_binary) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    for threshold in thresholds:
        y_pred_binary = (anomaly_scores > threshold).astype(int)
        
        # --- The only change is calling this function ---
        pa_f1, pa_precision, pa_recall = point_adjust_f1_score(y_true_binary, y_pred_binary)
        # ---
        
        if pa_f1 > best_pa_f1:
            best_pa_f1 = pa_f1
            best_threshold = threshold
            best_pa_precision = pa_precision
            best_pa_recall = pa_recall
            
    return best_threshold, best_pa_f1, best_pa_precision, best_pa_recall