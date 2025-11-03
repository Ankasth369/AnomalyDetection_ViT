# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm 
import sys
from torch.utils.data import DataLoader

# --- FIX: Add missing sklearn metric imports ---
from sklearn.metrics import f1_score, precision_score, recall_score
# --- END FIX ---

# --- CRITICAL FIX: Manually add the Kaggle root to the Python path ---
# This is a more robust fix for Kaggle's complex file structure
sys.path.append("/kaggle/working/advanced/Experiment")
# --- END FIX ---

# --- Import Project Modules ---
try:
    from src.utils.data_loader import TimeSeriesDataset
    # --- CRITICAL FIX: Import the correct threshold function ---
    from src.utils.metrics import find_best_point_adjust_f1_threshold, calculate_variable_contributions, point_adjust_f1_score
    from models.ViTAdapterModel import ViTAdapterAnomalyModel
except ImportError as e:
    st.error(f"Fatal Import Error: {e}.")
    st.error("I tried adding '/kaggle/working/' to the path, but still couldn't find your files.")
    st.error("Please ensure your 'src' and 'models' folders are in '/kaggle/working/'")
    st.stop()


# --- Configuration ---
KAGGLE_WORKING_DIR = "/kaggle/working/advanced/Experiment" # Set the root path
CHECKPOINT_DIR = os.path.join(KAGGLE_WORKING_DIR, "models/trained_weights/")
DEFAULT_MODEL_NAME = "adapter_smd.pth" # Use your best-performing model (e.g., SMD)
OUTPUT_SPATIAL_SIZE_DEFAULT = 224 # Must match your loader

# --- Model & Data Loading (Cached) ---

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """Loads the trained model weights and configuration."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        time_window_len = checkpoint.get('time_window_len', 96)
        max_vars = checkpoint.get('max_vars', 38)

        model = ViTAdapterAnomalyModel(
            time_window_len=time_window_len,
            max_vars=max_vars
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model {model_path} loaded successfully.")
        return model, time_window_len, max_vars, device
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# --- CRITICAL FIX: Update run_inference to use the "FAST" loader/model ---
def run_inference(_model, data_path, time_window_len, max_vars, device):
    """
    Runs inference on an entire uploaded file and returns scores.
    Uses the "FAST" data loader and "FAST" model.
    """
    try:
        eval_dataset = TimeSeriesDataset(
            data_path=data_path,
            window_size=time_window_len,
            step_size=12, # Use the same step size as evaluation for accurate scores
            max_vars=max_vars,
            output_spatial_size=OUTPUT_SPATIAL_SIZE_DEFAULT,
            train_mode=False
        )
    except Exception as e:
        st.error(f"Error loading data file: {e}")
        return None, None, None, None, None, None, None
        
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Running inference... (0%)")
    
    # Use a batch size that fits your GPU, and num_workers=2
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    all_window_labels = []
    all_window_scores = []
    all_window_contributions = []
    
    criterion = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        # The "fast" loader returns 4 items
        for i, (image_tensors, target_tensors, labels, num_vars_batch) in enumerate(eval_loader):
            
            image_tensors = image_tensors.to(device)
            target_tensors = target_tensors.to(device)
            M_i = num_vars_batch[0].item()
            
            # The "fast" model takes the IMAGE as input
            reconstructed = _model(image_tensors)
            
            # Calculate error against the ORIGINAL time series window
            error = criterion(reconstructed[..., :M_i], target_tensors[..., :M_i])
            window_scores = torch.mean(error, dim=(1, 2))
            
            all_window_labels.extend(labels.cpu().numpy())
            all_window_scores.extend(window_scores.cpu().numpy())
            
            # Calculate variable contributions for localization
            for j in range(target_tensors.shape[0]):
                # Use target_tensors and reconstructed (which are both (T, M_max))
                original_np = target_tensors[j, :, :M_i].cpu().numpy()
                reconstructed_np = reconstructed[j, :, :M_i].cpu().numpy()
                
                # We need to create the full M_max array for contributions
                contributions_full = np.zeros(max_vars)
                contributions_real = calculate_variable_contributions(original_np, reconstructed_np)
                contributions_full[:M_i] = contributions_real
                
                all_window_contributions.append(contributions_full)
            
            percent_complete = (i + 1) / len(eval_loader)
            progress_bar.progress(percent_complete)
            progress_text.text(f"Running inference... ({int(percent_complete * 100)}%)")

    progress_bar.empty() 
    progress_text.empty()
    
    y_true_win = np.array(all_window_labels)
    y_scores_win = np.array(all_window_scores)
    
    # --- Reconstruct Point-Wise Scores (from evaluate_model.py) ---
    y_true_pointwise = eval_dataset.labels.astype(int)
    y_scores_pointwise = np.zeros_like(y_true_pointwise, dtype=float)
    point_counts = np.zeros_like(y_true_pointwise, dtype=int)
    
    # --- Create point-wise contribution array ---
    point_contributions = np.zeros((len(y_true_pointwise), max_vars))

    for i, (start, end) in enumerate(eval_dataset.indices):
        if i < len(y_scores_win): 
            window_score = y_scores_win[i]
            y_scores_pointwise[start:end] += window_score
            point_counts[start:end] += 1
            
            # Store contributions at the *start* of the window
            if i < len(all_window_contributions):
                point_contributions[start, :] = all_window_contributions[i]
        
    # Average the scores
    point_counts[point_counts == 0] = 1
    y_scores_pointwise = y_scores_pointwise / point_counts
    
    # Propagate contribution scores forward to fill gaps
    point_contributions_df = pd.DataFrame(point_contributions).replace(0, np.nan).fillna(method='ffill').fillna(0)
    point_contributions = point_contributions_df.values
    
    return y_true_win, y_scores_win, y_true_pointwise, y_scores_pointwise, point_contributions, eval_dataset.feature_cols, eval_dataset.data
# --- END OF CRITICAL FIX ---


# --- Main Streamlit App ---

st.set_page_config(layout="wide", page_title="ViT-Adapter Anomaly Detection")
st.title("👁️‍🗨️ Time-Series Anomaly Detection with ViT-Adapter")
st.markdown("Upload a preprocessed CSV file (from `data/processed/`) to visualize anomalies.")

# --- 1. Sidebar for Model and File Selection ---
with st.sidebar:
    st.header("1. Model Selection")
    
    if not os.path.exists(CHECKPOINT_DIR):
        st.error(f"Checkpoint directory not found: {CHECKPOINT_DIR}. Did you copy your 'models' folder to /kaggle/working/?")
        st.stop()
        
    model_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    if not model_files:
        st.error(f"No models found in {CHECKPOINT_DIR}. Please train a model first.")
        st.stop()

    default_model_idx = model_files.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in model_files else 0
    
    selected_model_name = st.selectbox(
        "Select Trained Model",
        options=model_files,
        index=default_model_idx
    )
    model_path = os.path.join(CHECKPOINT_DIR, selected_model_name)
    
    model, T, M_max, device = load_model(model_path)
    
    if model:
        st.success(f"Loaded '{selected_model_name}'")
        st.write(f"Device: `{device}`")
        st.write(f"Model Config: T=`{T}`, M_max=`{M_max}`")

    st.header("2. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload a preprocessed CSV file for evaluation",
        type="csv"
    )

# --- 2. Main Panel for Visualization ---
if uploaded_file and model:
    st.header(f"Analysis: {uploaded_file.name}")
    
    # Save temp file
    temp_dir = os.path.join(KAGGLE_WORKING_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    y_true_win, y_scores_win, y_true_points, point_scores, point_contributions, feature_names, raw_data = run_inference(
        model, temp_path, T, M_max, device
    )
    
    if y_true_win is not None:
        st.success("Inference complete!")
        
        # --- 3. Threshold Slider & Metrics (Deliverable 4) ---
        st.header("3. Performance Tuning & Metrics")
        
        # --- METRIC FIX: Use the correct threshold function ---
        default_thresh, _, _, _ = find_best_point_adjust_f1_threshold(point_scores, y_true_points)
        
        score_max = float(np.max(point_scores)) if len(point_scores) > 0 else (default_thresh + 0.1)
        if score_max <= default_thresh:
            score_max = default_thresh + 0.1 
            
        st.info("Adjust the threshold slider to find the best balance between precision and recall.")
        
        best_thresh = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=score_max,
            value=default_thresh,
            step=(score_max / 200) if score_max > 0 else 0.01,
            format="%.5f"
        )
        
        y_pred_points = (point_scores > best_thresh).astype(int)
        
        best_f1 = f1_score(y_true_points, y_pred_points)
        prec = precision_score(y_true_points, y_pred_points, zero_division=0)
        recall = recall_score(y_true_points, y_pred_points, zero_division=0)
        
        # We now use the point-wise arrays for all metrics
        pa_f1, pa_prec, pa_recall = point_adjust_f1_score(y_true_points, y_pred_points)
        
        st.subheader("Metrics @ Selected Threshold")
        col1, col2, col3 = st.columns(3)
        col1.metric("Point-Adjusted F1", f"{pa_f1:.4f}")
        col2.metric("Point-Adjusted Precision", f"{pa_prec:.4f}")
        col3.metric("Point-Adjusted Recall", f"{pa_recall:.4f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Standard F1", f"{best_f1:.4f}")
        col2.metric("Standard Precision", f"{prec:.4f}")
        col3.metric("Standard Recall", f"{recall:.4f}")

        
        # --- 4. Plot Visualization (Deliverables 2 & 3) ---
        st.subheader("Anomaly Localization Plot")
        
        plot_features = [col for col in feature_names if not col.startswith('padding_')]
        num_vars_to_plot = min(len(plot_features), 10) # Plot top 10 vars
        
        # Create 3 subplots: Data, Anomaly Score, Contributions
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=(f"Input Data (Top {num_vars_to_plot} Vars)", "Anomaly Score", "Anomaly Variable Contribution"),
                            row_heights=[0.5, 0.2, 0.3]) # Give more space to data
                            
        # --- Plot 1: Input Data ---
        for i in range(num_vars_to_plot):
            fig.add_trace(go.Scatter(
                y=raw_data[:, i], 
                name=plot_features[i],
                line=dict(width=1)
            ), row=1, col=1)

        # --- Plot 2: Anomaly Score ---
        fig.add_trace(go.Scatter(
            y=point_scores,
            name="Anomaly Score",
            line=dict(color='orange', width=2)
        ), row=2, col=1)
        
        # Add threshold line
        fig.add_hline(
            y=best_thresh,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({best_thresh:.4f})",
            annotation_position="bottom right",
            row=2, col=1
        )
        
        # --- Plot 3: Variable Contributions (Localization) ---
        # Get the contributions from the *first detected anomaly*
        predictions_point = (point_scores > best_thresh).astype(int)
        anomaly_indices = np.where(predictions_point == 1)[0]
        
        if len(anomaly_indices) > 0:
            first_anomaly_idx = anomaly_indices[0]
            contributions_at_anomaly = point_contributions[first_anomaly_idx, :len(plot_features)]
            
            fig.add_trace(go.Bar(
                x=plot_features,
                y=contributions_at_anomaly,
                name="Variable Contribution",
                marker_color='red'
            ), row=3, col=1)
            fig.update_yaxes(title_text="Contribution %", row=3, col=1)
        else:
            # Add placeholder text if no anomalies are found
             fig.add_annotation(
                text="No anomalies detected at this threshold.",
                xref="paper", yref="paper",
                x=0.5, y=0.1, showarrow=False,
                row=3, col=1
            )

        
        # --- Add Anomaly Regions (Deliverable 3) ---
        # Add shaded red regions for detected anomalies
        start_idx = -1
        for i, idx in enumerate(anomaly_indices):
            if start_idx == -1:
                start_idx = idx 
            
            if (i + 1 == len(anomaly_indices)) or (anomaly_indices[i+1] > idx + 1):
                end_idx = idx
                fig.add_vrect(
                    x0=start_idx, x1=end_idx,
                    fillcolor="rgba(239, 68, 68, 0.2)", # Light red
                    layer="below", line_width=0,
                    name="Predicted Anomaly"
                )
                start_idx = -1 
        
        # --- Add GROUND TRUTH Regions ---
        true_anomaly_indices = np.where(y_true_points == 1)[0]
        start_idx = -1
        for i, idx in enumerate(true_anomaly_indices):
            if start_idx == -1:
                start_idx = idx 
            
            if (i + 1 == len(true_anomaly_indices)) or (true_anomaly_indices[i+1] > idx + 1):
                end_idx = idx
                fig.add_vrect(
                    x0=start_idx, x1=end_idx,
                    fillcolor="rgba(0, 255, 0, 0.2)", # Light Green
                    layer="below", line_width=0,
                    name="True Anomaly"
                )
                start_idx = -1

        # Clean up legend to avoid duplicate entries
        names = set()
        fig.for_each_trace(
            lambda trace:
                trace.update(showlegend=False)
                if (trace.name in names)
                else names.add(trace.name)
        )

        fig.update_layout(
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.info("Please select a model and upload a data file to begin analysis.")