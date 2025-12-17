#!/usr/bin/env python3
"""
"""

import torch
import torch.nn as nn
from pytorch3dunet.unet3d.model import ResidualUNet3D, UNet3D

class TrueResidualUNet3D(nn.Module):
    """
    
    """
    
    def __init__(self, 
                 in_channels=1,
                 out_channels=1, 
                 f_maps=[16, 32, 64],
                 num_levels=3,
                 backbone='ResidualUNet3D',
                 layer_order='gcr',
                 num_groups=8,
                 conv_padding=1,
                 dropout_prob=0.1,
                 **kwargs):
        """
        Args:
            backbone: 'ResidualUNet3D' or 'UNet3D'
        """
        super().__init__()
        
        if backbone == 'ResidualUNet3D':
            self.backbone = ResidualUNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                f_maps=f_maps,
                num_levels=num_levels,
                layer_order=layer_order,
                num_groups=num_groups,
                conv_padding=conv_padding,
                dropout_prob=dropout_prob,
                final_sigmoid=False,
                **kwargs
            )
        elif backbone == 'UNet3D':
            self.backbone = UNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                f_maps=f_maps,
                num_levels=num_levels,
                layer_order=layer_order,
                num_groups=num_groups,
                conv_padding=conv_padding,
                dropout_prob=dropout_prob,
                final_sigmoid=False,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        

        if hasattr(self.backbone, 'final_activation'):
            self.backbone.final_activation = nn.Identity()
        

        self._zero_init_residual_layer()
        
        print(f"TrueResidualUNet3D created with {backbone} backbone")
        print(f"Architecture: output = input + {backbone}(input)")
        print(f"Zero-initialized final layer for perfect identity mapping")
    
    def forward(self, x):
        """
        output = input + residual_learned
        """

        residual = self.backbone(x)
        

        output = x + residual
        
        return output
    
    def get_residual(self, x):
        return self.backbone(x)
    
    def _zero_init_residual_layer(self):
        """
        """

        last_conv = None
        
        def find_last_conv(module):
            nonlocal last_conv
            for name, child in module.named_children():
                if isinstance(child, (nn.Conv3d, nn.ConvTranspose3d)):
                    last_conv = child
                else:
                    find_last_conv(child)
        
        find_last_conv(self.backbone)
        
        if last_conv is not None:

            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)
            print(f"Zero-initialized final conv layer: {type(last_conv).__name__}")
            print(f"  Weight shape: {last_conv.weight.shape}")
            print(f"  Bias: {'Yes' if last_conv.bias is not None else 'No'}")
        else:
            print("Warning: Could not find final conv layer to zero-initialize")
