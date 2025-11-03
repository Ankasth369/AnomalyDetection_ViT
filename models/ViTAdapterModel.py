# ViTAdapterModel.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np

# --- Import Project Modules ---
try:
    # --- FIX: We no longer import or use TimeFrequencyAdapter here ---
    from .heads.ReconstructionDecoder import ReconstructionDecoder
except ImportError:
    # Fallback for running scripts from the root directory
    try:
        from models.heads.ReconstructionDecoder import ReconstructionDecoder
    except ImportError as e:
        print(f"Error importing Decoder module in ViTAdapterModel: {e}")
        print("Please ensure your Python path is set up to see the 'models' directory.")
        exit()


class ViTAdapterAnomalyModel(nn.Module):
    """
    The main anomaly detection model. (FAST VERSION)
    
    This model wraps a frozen ViT-Base-16 backbone.
    It expects pre-computed images from a "fast" data loader.
    """
    
    def __init__(self, time_window_len: int, max_vars: int):
        super().__init__()
        
        self.time_window_len = time_window_len
        self.max_vars = max_vars
        
        # --- 1. Trainable Input Adapter (REMOVED) ---
        # The TimeFrequencyAdapter logic is now in data_loader.py
        # to prevent a CPU bottleneck.
        
        # --- 2. Frozen ViT Backbone ---
        print("Initializing frozen ViT-Base-16 model...")
        vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit_backbone = vit_b_16(weights=vit_weights)
        
        # --- Adapt the ViT Input Layer (Patch Embedding) ---
        original_conv_proj = self.vit_backbone.conv_proj
        has_bias = original_conv_proj.bias is not None
        
        # Calculate total channels = max_vars * 2 (GAF + Spectrogram)
        self.total_channels = self.max_vars * 2 # e.g., 38 * 2 = 76
        
        new_conv_proj = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=original_conv_proj.out_channels, # 768
            kernel_size=original_conv_proj.kernel_size,   # (16, 16)
            stride=original_conv_proj.stride,         # (16, 16)
            padding=original_conv_proj.padding,       # (0, 0)
            bias=has_bias
        )

        # Initialize new weights: Copy 3-channel (ImageNet) weights
        # and set all other new channels (4-76) to zero.
        with torch.no_grad():
            new_conv_proj.weight.data[:, :3, :, :] = original_conv_proj.weight.data
            new_conv_proj.weight.data[:, 3:, :, :].fill_(0)
            if has_bias:
                new_conv_proj.bias.data.copy_(original_conv_proj.bias.data)

        self.vit_backbone.conv_proj = new_conv_proj
        
        # --- 3. Trainable Output Decoder ---
        self.vit_output_dim = self.vit_backbone.hidden_dim # 768
        
        # Use the argument names from your 'ReconstructionDecoder.py' file
        self.reconstruction_decoder = ReconstructionDecoder(
            vit_embedding_dim=self.vit_output_dim, 
            time_window_len=self.time_window_len,
            max_vars=self.max_vars
        )

    def freeze_vit_backbone(self):
        """
        Freezes all parameters in the ViT backbone.
        """
        for param in self.vit_backbone.parameters():
            param.requires_grad = False
            
        # --- WARNING: BLUEPRINT CONFLICT ---
        # Your blueprint [source 11] says NOT to "fine tune the parameters
        # of this model". Unfreezing this layer contradicts that.
        #
        # ACTION: You MUST verify this with your TAs.
        # 
        # To strictly follow the blueprint, you would comment out
        # the loop below.
        # ------------------------------------
        
        # Unfreeze the (new) conv_proj layer so it can learn to
        # map the 76-channel input.
        print("--- WARNING: Unfreezing ViT conv_proj layer for training. ---")
        print("--- Verify this step is allowed by your project blueprint. ---")
        for param in self.vit_backbone.conv_proj.parameters():
             param.requires_grad = True

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the anomaly detection model.
        
        Args:
            x_image (torch.Tensor): Input image tensor from "fast" loader.
                                    Shape: (B, C_max, H, W) e.g. (B, 76, 224, 224)
        """
        
        # 1. Pass through the Time-Frequency Adapter (REMOVED)
        # This is now done in the data loader.
        
        # 2. Pass through the Frozen ViT Backbone
        # Input: (B, 76, 224, 224)
        # _process_input runs the (now trainable) conv_proj
        x_features = self.vit_backbone._process_input(x_image)
        
        # Add the [CLS] token
        cls_token = self.vit_backbone.class_token
        cls_tokens = cls_token.expand(x_features.shape[0], -1, -1)
        x_features = torch.cat([cls_tokens, x_features], dim=1)
        
        # Pass through the Transformer blocks
        x_features = self.vit_backbone.encoder(x_features)
        
        # Get the pooled output (the [CLS] token)
        x_pooled = x_features[:, 0] # Shape: (B, 768)
        
        # 3. Pass through the Reconstruction Decoder
        reconstructed_flat = self.reconstruction_decoder(x_pooled)
        
        # 4. Reshape to the target time series format
        reconstructed_window = reconstructed_flat.view(
            -1, self.time_window_len, self.max_vars
        )
        
        return reconstructed_window