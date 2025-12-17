"""
Training Factory for Event-Voxel Denoising
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.datasets.event_voxel_dataset import EventVoxelDataset
from src.training.custom_trainer import EventVoxelTrainer


class TrainingFactory:
    """
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        
        Args:
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_model(self) -> nn.Module:
        """
        
        Returns:
        """
        model_config = self.config['model']
        
        if model_config['name'] == 'ResidualUNet3D':
            try:

                try:
                    from pytorch3dunet.unet3d.model import ResidualUNet3D
                except ImportError:

                    from pytorch3dunet.unet3d import ResidualUNet3D
                

                model = ResidualUNet3D(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    f_maps=model_config.get('f_maps', 32),
                    layer_order=model_config.get('layer_order', 'gcr'),
                    num_groups=model_config.get('num_groups', 8),
                    num_levels=model_config.get('num_levels', 4),
                    final_sigmoid=True,
                    conv_kernel_size=model_config.get('conv_kernel_size', 3),
                    pool_kernel_size=model_config.get('pool_kernel_size', 2)
                )
                


                if hasattr(model, 'final_activation'):
                    import torch.nn as nn
                    original_activation = str(model.final_activation)
                    model.final_activation = nn.Identity()
                    self.logger.info(f"Forced replacement: {original_activation} -> Identity() for unbounded voxel values")
                else:
                    self.logger.warning("Could not find final_activation layer to replace")
                

                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.logger.info(f"Created ResidualUNet3D model:")
                self.logger.info(f"  - Architecture: {model_config['f_maps']} feature maps, {model_config.get('num_levels', 4)} levels")
                self.logger.info(f"  - Input channels: {model_config['in_channels']}")
                self.logger.info(f"  - Output channels: {model_config['out_channels']}")
                self.logger.info(f"  - Residual connections: Enabled for deflare task")
                self.logger.info(f"  - Total parameters: {total_params:,}")
                self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
                
                return model
                
            except ImportError as e:
                self.logger.error(f"Failed to import pytorch-3dunet ResidualUNet3D: {e}")
                self.logger.error("Please install pytorch-3dunet: conda install -c conda-forge pytorch-3dunet")
                raise
            except Exception as e:
                self.logger.error(f"Failed to create ResidualUNet3D model: {e}")
                raise
                
        elif model_config['name'] == 'TrueResidualUNet3D':

            try:
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent.parent.parent))
                from true_residual_wrapper import TrueResidualUNet3D
                
                model = TrueResidualUNet3D(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    f_maps=model_config.get('f_maps', [16, 32, 64]),
                    num_levels=model_config.get('num_levels', 3),
                    layer_order=model_config.get('layer_order', 'gcr'),
                    num_groups=model_config.get('num_groups', 8),
                    conv_padding=model_config.get('conv_padding', 1),
                    dropout_prob=model_config.get('dropout_prob', 0.1),
                    backbone=model_config.get('backbone', 'ResidualUNet3D')
                )
                

                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.logger.info(f"Created TrueResidualUNet3D model:")
                self.logger.info(f"  - Architecture: {model_config.get('f_maps', [16, 32, 64])} feature maps, {model_config.get('num_levels', 3)} levels")
                self.logger.info(f"  - Backbone: {model_config.get('backbone', 'ResidualUNet3D')}")
                self.logger.info(f"  - True residual learning: output = input + backbone(input)")
                self.logger.info(f"  - Zero-initialized final layer for perfect identity mapping")
                self.logger.info(f"  - Input channels: {model_config['in_channels']}")
                self.logger.info(f"  - Output channels: {model_config['out_channels']}")
                self.logger.info(f"  - Total parameters: {total_params:,}")
                self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
                
                return model
                
            except ImportError as e:
                self.logger.error(f"Failed to import TrueResidualUNet3D: {e}")
                self.logger.error("Make sure true_residual_wrapper.py is in the project root")
                raise
            except Exception as e:
                self.logger.error(f"Failed to create TrueResidualUNet3D model: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported model: {model_config['name']}")
    
    def create_datasets(self) -> Tuple[EventVoxelDataset, EventVoxelDataset]:
        """
        
        Returns:
            (train_dataset, val_dataset)
        """
        loader_config = self.config['loaders']
        

        train_noisy_dir = loader_config.get('train_noisy_dir')
        train_clean_dir = loader_config.get('train_clean_dir')
        val_noisy_dir = loader_config.get('val_noisy_dir')
        val_clean_dir = loader_config.get('val_clean_dir')
        

        if not train_noisy_dir:
            train_paths = loader_config.get('train_path', [])
            if isinstance(train_paths, list):
                train_noisy_dir = train_paths[0] + '/noisy' if len(train_paths) > 0 else 'train/noisy'
                train_clean_dir = train_paths[0] + '/clean' if len(train_paths) > 0 else 'train/clean'
            else:
                train_noisy_dir = str(train_paths) + '/noisy'
                train_clean_dir = str(train_paths) + '/clean'
        
        if not val_noisy_dir:
            val_paths = loader_config.get('val_path', [])
            if isinstance(val_paths, list):
                val_noisy_dir = val_paths[0] + '/noisy' if len(val_paths) > 0 else 'val/noisy' 
                val_clean_dir = val_paths[0] + '/clean' if len(val_paths) > 0 else 'val/clean'
            else:
                val_noisy_dir = str(val_paths) + '/noisy'
                val_clean_dir = str(val_paths) + '/clean'
        

        sensor_size = tuple(loader_config.get('sensor_size', [480, 640]))
        

        segment_duration_us = loader_config.get('segment_duration_us', 20000)  # 20ms
        num_bins = loader_config.get('num_bins', 8)                            # 8 bins
        num_segments = loader_config.get('num_segments', 5)                    # 5 segments
        
        self.logger.info(f"Creating datasets:")
        self.logger.info(f"  - Sensor size: {sensor_size}")
        self.logger.info(f"  - Segment config: {segment_duration_us/1000}ms/{num_bins}bins = {segment_duration_us/num_bins/1000:.2f}ms per bin")
        self.logger.info(f"  - Segments per file: {num_segments}")
        

        try:
            train_dataset = EventVoxelDataset(
                noisy_events_dir=train_noisy_dir,
                clean_events_dir=train_clean_dir,
                sensor_size=sensor_size,
                segment_duration_us=segment_duration_us,
                num_bins=num_bins,
                num_segments=num_segments
            )
            
            self.logger.info(f"Training dataset: {len(train_dataset)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to create training dataset: {e}")
            self.logger.error(f"Train paths - Noisy: {train_noisy_dir}, Clean: {train_clean_dir}")
            raise
        

        try:
            val_dataset = EventVoxelDataset(
                noisy_events_dir=val_noisy_dir,
                clean_events_dir=val_clean_dir,
                sensor_size=sensor_size,
                segment_duration_us=segment_duration_us,
                num_bins=num_bins,
                num_segments=num_segments
            )
            
            self.logger.info(f"Validation dataset: {len(val_dataset)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to create validation dataset: {e}")
            self.logger.error(f"Val paths - Noisy: {val_noisy_dir}, Clean: {val_clean_dir}")
            raise
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: EventVoxelDataset, 
                          val_dataset: EventVoxelDataset) -> Tuple[DataLoader, DataLoader]:
        """
        
        Args:
            
        Returns:
            (train_loader, val_loader)
        """
        loader_config = self.config['loaders']
        
        batch_size = loader_config.get('batch_size', 1)
        num_workers = loader_config.get('num_workers', 2)
        

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"Created data loaders:")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Num workers: {num_workers}")
        self.logger.info(f"  - Train batches: {len(train_loader)}")
        self.logger.info(f"  - Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_trainer(self, model: nn.Module, 
                      train_loader: DataLoader, 
                      val_loader: DataLoader,
                      device: str = 'cuda') -> EventVoxelTrainer:
        """
        
        Args:
            
        Returns:
        """

        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        

        trainer = EventVoxelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=device
        )
        
        return trainer
    
    def setup_complete_training(self, device: Optional[str] = None) -> EventVoxelTrainer:
        """
        
        Args:
            
        Returns:
        """
        self.logger.info("=== Setting up Event-Voxel Denoising Training ===")
        

        if device is None:
            device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        

        self.logger.info("Step 1: Creating UNet3D model...")
        model = self.create_model()
        

        self.logger.info("Step 2: Creating datasets...")
        train_dataset, val_dataset = self.create_datasets()
        

        self.logger.info("Step 3: Creating data loaders...")
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
        

        self.logger.info("Step 4: Creating trainer...")
        trainer = self.create_trainer(model, train_loader, val_loader, device)
        

        resume_path = self.config['trainer'].get('resume')
        if resume_path:
            self.logger.info(f"Step 5: Resuming from checkpoint: {resume_path}")
            success = trainer.load_checkpoint(resume_path)
            if not success:
                self.logger.warning("Failed to load checkpoint, starting from scratch")
        
        self.logger.info("=== Training setup completed ===")
        return trainer