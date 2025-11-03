# ReconstructionDecoder.py
import torch
import torch.nn as nn

class ReconstructionDecoder(nn.Module):
    """
    A simple trainable MLP head that takes the final pooled feature vector 
    from the frozen ViT and reconstructs the original time series window.

    The reconstruction error (MSE Loss) of this output forms the anomaly score.
    """
    def __init__(self, vit_embedding_dim: int = 768, time_window_len: int = 96, max_vars: int = 38):
        """
        Initializes the decoder layers.

        Args:
            vit_embedding_dim (int): The dimension of the pooled ViT output (768 for ViT-Base).
            time_window_len (int): The number of time steps in the input window (T=96).
            max_vars (int): The maximum number of variables (M_max=38).
        """
        super().__init__()
        
        # Target output size is T * M_max (96 * 38 = 3648)
        self.output_size = time_window_len * max_vars
        
        # A simple two-layer MLP for reconstruction
        self.decoder = nn.Sequential(
            # Layer 1: Expand capacity
            nn.Linear(vit_embedding_dim, vit_embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 2: Project to the flattened output dimension
            nn.Linear(vit_embedding_dim * 2, self.output_size)
        )
        
        # Store dimensions to reshape output later
        self.time_window_len = time_window_len
        self.max_vars = max_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the ViT features and reconstructs the time series window.

        Args:
            x (torch.Tensor): The pooled ViT feature vector, shape (B, vit_embedding_dim).

        Returns:
            torch.Tensor: The reconstructed time series window, shape (B, T, M_max).
        """
        # Pass through the MLP
        reconstructed_flat = self.decoder(x)
        
        # Reshape to the target time series format (B, T, M_max)
        reconstructed = reconstructed_flat.view(
            -1, self.time_window_len, self.max_vars
        )
        
        return reconstructed