#!/usr/bin/env python3
"""
Main entry point for Event-Voxel Denoising with Custom Training System
Supports three modes: train, test, inference

Usage:
    python main.py train --config configs/train_config.yaml
    python main.py test --config configs/test_config.yaml  
    python main.py inference --config configs/inference_config.yaml --input input.h5 --output output.h5
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config_loader import ConfigLoader
from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events
from src.training.training_factory import TrainingFactory

class EventVoxelPipeline:
    """
    Main pipeline for event-voxel denoising system
    Integrates pytorch-3dunet with custom event processing
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.config_loader = ConfigLoader()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def train_mode(self, config_path: str, debug: bool = False, debug_dir: str = 'debug_output') -> int:
        """
        Training mode - uses custom training system with pytorch-3dunet UNet3D model
        
        Args:
            config_path: Path to training configuration YAML
            debug: Enable debug mode with detailed visualization
            debug_dir: Directory for debug visualizations
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        mode_str = "DEBUG TRAINING" if debug else "CUSTOM TRAINING"
        self.logger.info(f"=== EVENT-VOXEL {mode_str} MODE ===")
        
        try:
            # Load and validate training configuration
            config = self.config_loader.load_train_config(config_path)
            self.logger.info(f"Loaded training config from: {config_path}")
            
            # Inject debug configuration
            if debug:
                config['debug'] = {
                    'enabled': True,
                    'debug_dir': debug_dir,
                    'max_iterations': 2,  # Only run 1-2 iterations in debug mode
                    'max_epochs': 1       # Only 1 epoch in debug mode
                }
                
                # Override training epochs for debug
                config['trainer']['max_num_epochs'] = 1
                
                # Create debug output directory
                debug_path = Path(debug_dir)
                debug_path.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"🐛 DEBUG MODE ENABLED")
                self.logger.info(f"🐛 Debug output: {debug_dir}")
                self.logger.info(f"🐛 Will run max {config['debug']['max_iterations']} iterations only")
                self.logger.info(f"🐛 9 visualization folders will be generated per iteration")
            else:
                config['debug'] = {'enabled': False}
            
            self.logger.info("Starting custom training system...")
            self.logger.info(f"Model: {config['model']['name']} ({config['model']['f_maps']} feature maps)")
            self.logger.info(f"Loss: {config['loss']['name']}")
            self.logger.info(f"Optimizer: {config['optimizer']['name']} (LR: {config['optimizer']['learning_rate']})")
            

            training_factory = TrainingFactory(config)
            

            trainer = training_factory.setup_complete_training()
            

            best_val_loss = trainer.train()
            
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
            self.logger.info(f"Checkpoints saved to: {config['trainer']['checkpoint_dir']}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Training mode error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
    
    def test_mode(self, config_path: str, debug: bool = False, debug_dir: str = 'debug_output', baseline: bool = False) -> int:
        """
        Testing mode - batch inference on test dataset, saving processed H5 files
        
        Two modes:
        - Normal: Process through trained UNet model (saves to *output directory)  
        - Baseline: Skip UNet, only encode-decode (saves to *baseline directory)
        
        Process each input H5 file and save the processed output
        with the same filename to a parallel output directory (input_dir + 'output').
        
        Args:
            config_path: Path to test configuration YAML
            debug: Enable debug mode with detailed visualization (optional)
            debug_dir: Directory for debug visualizations (optional)
            baseline: Enable baseline mode (encode-decode only, no UNet) (optional)
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        if baseline:
            mode_str = "BASELINE TESTING" if not debug else "DEBUG BASELINE TESTING"
        else:
            mode_str = "DEBUG TESTING" if debug else "TESTING"
        self.logger.info(f"=== EVENT-VOXEL {mode_str} MODE ===")
        
        try:
            # Load and validate test configuration  
            config = self.config_loader.load_test_config(config_path)
            self.logger.info(f"Loaded test config from: {config_path}")
            
            # Inject debug configuration (same as train mode)
            if debug:
                config['debug'] = {
                    'enabled': True,
                    'debug_dir': debug_dir
                }
                
                # Create debug output directory
                debug_path = Path(debug_dir)
                debug_path.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"🐛 DEBUG MODE ENABLED")
                self.logger.info(f"🐛 Debug output: {debug_dir}")
                self.logger.info(f"🐛 Smart sampling: visualizing every 5th batch (1st segment of each file)")
                self.logger.info(f"🐛 Expected visualizations: ~52 folders (1 per file)")
            else:
                config['debug'] = {'enabled': False}
            
            # Model setup (skip in baseline mode)
            model = None
            if not baseline:
                # Verify model checkpoint exists
                model_path = config['model']['path']
                if not Path(model_path).exists():
                    self.logger.error(f"Model checkpoint not found: {model_path}")
                    return 1
                
                self.logger.info("Starting custom evaluation...")
                self.logger.info(f"Model: {config['model']['name']} ({config['model']['f_maps']} feature maps)")
                self.logger.info(f"Checkpoint: {model_path}")
                

                training_factory = TrainingFactory(config)
                

                model = training_factory.create_model()
            else:
                self.logger.info("Baseline mode: Skipping model loading")
                # Still need training factory for dataset creation
                training_factory = TrainingFactory(config)
            # For test mode: only create validation dataset (which is our test data)
            try:
                _, test_dataset = training_factory.create_datasets()
            except:
                # If train dataset creation fails, just create val dataset directly
                from src.datasets.event_voxel_dataset import EventVoxelDataset
                loader_config = config.get('loaders', {})
                val_noisy_dir = loader_config.get('val_noisy_dir')
                val_clean_dir = loader_config.get('val_clean_dir')
                sensor_size = tuple(loader_config.get('sensor_size', [480, 640]))
                segment_duration_us = loader_config.get('segment_duration_us', 20000)
                num_bins = loader_config.get('num_bins', 8)
                num_segments = loader_config.get('num_segments', 5)
                
                test_dataset = EventVoxelDataset(
                    noisy_events_dir=val_noisy_dir,
                    clean_events_dir=val_clean_dir,
                    sensor_size=sensor_size,
                    segment_duration_us=segment_duration_us,
                    num_bins=num_bins,
                    num_segments=num_segments
                )
            
            _, test_loader = training_factory.create_dataloaders(test_dataset, test_dataset)
            

            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            if not baseline:
                checkpoint = torch.load(model_path, map_location=device)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
            

            total_loss = 0.0
            num_batches = 0
            criterion = torch.nn.MSELoss()
            
            # Get output directory configuration
            loader_config = config.get('loaders', {})
            input_dir = Path(loader_config.get('val_noisy_dir'))
            if baseline:
                output_dir = input_dir.parent / f"{input_dir.name}baseline"
            else:
                output_dir = input_dir.parent / f"{input_dir.name}output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get inference parameters
            sensor_size = tuple(loader_config.get('sensor_size', [480, 640]))
            segment_duration_us = loader_config.get('segment_duration_us', 20000)
            num_bins = loader_config.get('num_bins', 8)
            num_segments = loader_config.get('num_segments', 5)
            
            self.logger.info(f"Input directory: {input_dir}")
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info(f"Processing {len(test_dataset)} samples...")
            
            # Track processed files to avoid duplicates (each file has 5 segments)
            processed_files = set()
            

            debug_config = config.get('debug', {})
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                        
                    inputs = batch['raw'].to(device)
                    targets = batch['label'].to(device)
                    


                    if debug_config.get('enabled', False) and batch_idx % 5 == 0:
                        file_idx = batch_idx // 5
                        self.logger.info(f"🐛 Visualizing file {file_idx + 1} (batch {batch_idx})")
                        self._trigger_test_debug_visualization(
                            batch_idx, inputs, targets, batch, 
                            debug_config['debug_dir'], batch_idx
                        )
                    
                    # Baseline mode: skip UNet processing, use inputs as outputs (encode-decode only)
                    if baseline:
                        outputs = inputs.clone()  # Identity mapping (encode-decode baseline)
                        if batch_idx == 0:  # Only log once to avoid spam
                            self.logger.info("🔄 Baseline mode: Skipping UNet processing, using encode-decode only")
                    else:
                        outputs = model(inputs)
                    

                    if debug_config.get('enabled', False) and batch_idx % 5 == 0:
                        self._trigger_test_model_output_visualization(
                            batch_idx, inputs, outputs, debug_config['debug_dir'], batch_idx, model
                        )
                    
                    # Save model output for each segment
                    file_idx = batch_idx // num_segments  # File index
                    segment_idx = batch_idx % num_segments  # Segment index within file
                    
                    # Get original file information from dataset
                    if hasattr(test_dataset, 'file_pairs') and file_idx < len(test_dataset.file_pairs):
                        file_pair = test_dataset.file_pairs[file_idx]
                        input_file_path = Path(file_pair[0])  # noisy file path
                        output_file_name = input_file_path.name
                        
                        # Process and save output for this segment
                        self._save_segment_output(
                            outputs, input_file_path, output_file_name, output_dir,
                            file_idx, segment_idx, sensor_size, segment_duration_us,
                            num_bins, num_segments, processed_files
                        )
                    
                    # Calculate loss (or dummy loss for baseline mode)
                    if baseline:
                        # In baseline mode, loss is meaningless since output = input
                        loss_value = 0.0
                    else:
                        loss = criterion(outputs, targets)
                        loss_value = loss.item()
                    
                    total_loss += loss_value
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        if baseline:
                            self.logger.info(f"Batch {batch_idx}/{len(test_loader)} (Baseline mode)")
                        else:
                            self.logger.info(f"Batch {batch_idx}/{len(test_loader)}, Loss: {loss_value:.6f}")
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            # Clean up temporary directory
            temp_dir = output_dir / 'temp_segments'
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            
            if baseline:
                self.logger.info("Baseline processing completed successfully!")
                self.logger.info("(Baseline mode: encode-decode only, no UNet processing)")
            else:
                self.logger.info("Batch inference completed successfully!")
                self.logger.info(f"Average test loss: {avg_loss:.6f}")
            self.logger.info(f"Processed {len(processed_files)} files")
            self.logger.info(f"Output directory: {output_dir}")
            

            results = {
                'test_loss': avg_loss if not baseline else None,
                'mode': 'baseline' if baseline else 'normal',
                'num_samples': len(test_dataset),
                'num_files_processed': len(processed_files),
                'model_path': model_path if not baseline else None,
                'input_dir': str(input_dir),
                'output_dir': str(output_dir)
            }
            
            results_dir = Path(config.get('predictor', {}).get('output_dir', 'test_results'))
            results_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(results_dir / 'test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Test results saved to: {results_dir}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Testing mode error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
    
    def inference_mode(self, config_path: str, input_path: str = None, output_path: str = None, debug: bool = False, debug_dir: str = None, baseline: bool = False) -> int:
        """
        Inference mode - batch denoise event files (no ground truth)
        Pipeline: H5 events → segments → voxels → model → denoised voxels → events → H5
        
        Two modes:
        - Normal: Process through trained UNet model
        - Baseline: Skip UNet, only encode-decode (saves to *baseline directory)
        
        Args:
            config_path: Path to inference configuration YAML
            input_path: Input directory or single file path (if None, use config)
            output_path: Output directory or single file path (if None, use config)
            debug: Enable debug visualization mode
            debug_dir: Custom debug output directory
            baseline: Enable baseline mode (encode-decode only, no UNet)
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        mode_str = "BASELINE INFERENCE" if baseline else "INFERENCE"
        self.logger.info(f"=== EVENT-VOXEL {mode_str} MODE ===")
        
        try:
            # Load and validate inference configuration
            config = self.config_loader.load_inference_config(config_path)
            self.logger.info(f"Loaded inference config from: {config_path}")
            
            # Determine input/output paths
            if input_path is None:
                input_path = config.get('inference', {}).get('input_dir', 'DSEC_data/input')
            if output_path is None:
                default_output = config.get('inference', {}).get('output_dir', 'DSEC_data/output')
                if baseline:
                    # Add baseline suffix for baseline mode
                    default_output_path = Path(default_output)
                    output_path = str(default_output_path.parent / f"{default_output_path.name}baseline")
                else:
                    output_path = default_output
            
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # Check if single file or directory processing
            if input_path.is_file():
                return self._inference_single_file(config, input_path, output_path, debug, debug_dir, baseline)
            elif input_path.is_dir():
                return self._inference_batch_files(config, input_path, output_path, debug, debug_dir, baseline)
            else:
                raise FileNotFoundError(f"Input path not found: {input_path}")
                
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
            
        except Exception as e:
            self.logger.error(f"Inference mode error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
    
    def _inference_single_file(self, config: dict, input_file: Path, output_file: Path, debug: bool, debug_dir: str, baseline: bool = False) -> int:
        """
        Process single file inference
        
        Args:
            baseline: If True, skip UNet processing and only do encode-decode
        """
        mode_str = "baseline" if baseline else "normal"
        self.logger.info(f"Single file inference ({mode_str}): {input_file.name}")
        
        # Verify model checkpoint exists (skip in baseline mode)
        model_path = config['model']['path']
        if not baseline and not Path(model_path).exists():
            self.logger.error(f"Model checkpoint not found: {model_path}")
            return 1
        
        # Extract inference parameters
        inf_config = config['inference']
        sensor_size = tuple(inf_config['sensor_size'])
        segment_duration_us = inf_config['segment_duration_us']
        num_bins = inf_config['num_bins']
        num_segments = inf_config['num_segments']
        device = inf_config['device']
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input: {input_file}")
        self.logger.info(f"Output: {output_file}")
        if not baseline:
            self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Processing: {num_segments} segments × {segment_duration_us/1000}ms × {num_bins} bins")
        self.logger.info(f"Device: {device}")
        if baseline:
            self.logger.info("Baseline mode: Skipping model loading")
        
        # Load model (skip in baseline mode)
        model = None
        if not baseline:
            self.logger.info("Loading trained model...")
            model = self._load_model(config['model'], device)
        
        # Load input events
        self.logger.info("Loading input events...")
        events_np = load_h5_events(str(input_file))
        self.logger.info(f"Loaded {len(events_np)} events")
        
        # Process segments
        self.logger.info("Processing segments through denoising pipeline...")
        denoised_segments = []
        debug_data = []  # Store debug data for each segment
        
        for segment_idx in range(num_segments):
            self.logger.info(f"Processing segment {segment_idx + 1}/{num_segments}...")
            
            # Extract segment events
            segment_events = self._extract_segment_events(
                events_np, segment_idx, num_segments
            )
            
            if len(segment_events) == 0:
                self.logger.warning(f"Segment {segment_idx} is empty, skipping")
                continue
            
            # Encode to voxel
            input_voxel = events_to_voxel(
                segment_events,
                num_bins=num_bins,
                sensor_size=sensor_size,
                fixed_duration_us=segment_duration_us
            )
            
            # Baseline mode: skip UNet processing, use input as output (encode-decode only)
            if baseline:
                output_voxel = input_voxel.clone()  # Identity mapping for baseline
                if segment_idx == 0:  # Only log once to avoid spam
                    self.logger.info("🔄 Baseline mode: Skipping UNet processing, using encode-decode only")
            else:
                # Prepare tensor for model
                input_tensor = input_voxel.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 8, H, W)
                
                # Model inference
                model.eval()
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                
                # Remove batch and channel dimensions
                output_voxel = output_tensor.squeeze(0).squeeze(0).cpu()  # (8, H, W)
            
            # Decode back to events
            output_events = voxel_to_events(
                output_voxel,
                total_duration=segment_duration_us,
                sensor_size=sensor_size
            )
            
            # Adjust timestamps to global timeline
            if len(output_events) > 0:
                t_min_original = events_np[:, 0].min()
                t_max_original = events_np[:, 0].max()
                total_duration_original = t_max_original - t_min_original
                
                segment_start_global = t_min_original + segment_idx * (total_duration_original / num_segments)
                
                # Adjust timestamps
                output_events[:, 0] = output_events[:, 0] + segment_start_global - output_events[:, 0].min()
                
                denoised_segments.append(output_events)
            
            # Store debug data if needed
            if debug:
                debug_data.append({
                    'segment_idx': segment_idx,
                    'input_events': segment_events,
                    'input_voxel': input_voxel,
                    'output_events': output_events,
                    'output_voxel': output_voxel
                })
            
            self.logger.info(f"Segment {segment_idx + 1}: {len(segment_events)} → {len(output_events) if len(output_events) > 0 else 0} events")
        
        # Merge all segments
        if denoised_segments:
            final_events = np.vstack(denoised_segments)
            final_events = final_events[np.argsort(final_events[:, 0])]
        else:
            self.logger.warning("No events produced by denoising pipeline")
            final_events = np.empty((0, 4))
        
        # Save output
        self.logger.info(f"Saving {len(final_events)} denoised events to: {output_file}")
        self._save_events_to_h5(final_events, str(output_file))
        
        # Debug visualization if enabled
        if debug and debug_data:
            self._generate_inference_debug_visualization(
                debug_data, input_file.stem, debug_dir or 'debug_output',
                sensor_size, segment_duration_us
            )
        
        self.logger.info(f"Inference completed: {len(events_np)} → {len(final_events)} events")
        return 0
    
    def _inference_batch_files(self, config: dict, input_dir: Path, output_dir: Path, debug: bool, debug_dir: str, baseline: bool = False) -> int:
        """
        Process batch directory inference
        
        Args:
            baseline: If True, skip UNet processing and only do encode-decode
        """
        mode_str = "baseline" if baseline else "normal"
        self.logger.info(f"Batch inference ({mode_str}): {input_dir} → {output_dir}")
        
        # Find all H5 files
        h5_files = list(input_dir.glob("*.h5"))
        if not h5_files:
            self.logger.error(f"No H5 files found in: {input_dir}")
            return 1
        
        self.logger.info(f"Found {len(h5_files)} H5 files to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        processed_files = []
        failed_files = []
        
        for i, input_file in enumerate(h5_files, 1):
            self.logger.info(f"Processing file {i}/{len(h5_files)}: {input_file.name}")
            
            output_file = output_dir / input_file.name
            
            try:
                result = self._inference_single_file(config, input_file, output_file, debug, debug_dir, baseline)
                if result == 0:
                    processed_files.append(input_file.name)
                else:
                    failed_files.append(input_file.name)
            except Exception as e:
                self.logger.error(f"Failed to process {input_file.name}: {e}")
                failed_files.append(input_file.name)
                continue
        
        # Summary
        self.logger.info("Batch inference completed!")
        self.logger.info(f"Successfully processed: {len(processed_files)} files")
        if failed_files:
            self.logger.warning(f"Failed files: {len(failed_files)} - {failed_files}")
        self.logger.info(f"Output directory: {output_dir}")
        
        return 0 if not failed_files else 1
    
    def _generate_inference_debug_visualization(self, debug_data: list, file_stem: str, debug_dir: str, sensor_size: tuple, segment_duration_us: int):
        """
        Generate debug visualization for inference (no ground truth)
        Similar to test debug but without target comparisons
        """
        from src.data_processing.professional_visualizer import visualize_events_and_voxel
        
        debug_output_dir = Path(debug_dir)
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process first few segments for visualization (to avoid too many files)
        segments_to_viz = min(3, len(debug_data))
        
        for i in range(segments_to_viz):
            data = debug_data[i]
            segment_idx = data['segment_idx']
            
            # Create debug folder for this file and segment
            viz_dir = debug_output_dir / f"inference_{file_stem}_seg_{segment_idx}"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Input events visualization
                input_events_dir = viz_dir / "1_input_events"
                if len(data['input_events']) > 0:
                    visualize_events_and_voxel(
                        data['input_events'],
                        data['input_voxel'],
                        sensor_size,
                        str(input_events_dir),
                        f"input_seg_{segment_idx}"
                    )
                
                # Output events visualization
                output_events_dir = viz_dir / "2_output_events"
                if len(data['output_events']) > 0:
                    visualize_events_and_voxel(
                        data['output_events'],
                        data['output_voxel'],
                        sensor_size,
                        str(output_events_dir),
                        f"output_seg_{segment_idx}"
                    )
                
                # Summary info
                with open(viz_dir / "debug_summary.txt", 'w') as f:
                    f.write(f"📊 Inference Debug Summary - {file_stem}\n")
                    f.write(f"Segment: {segment_idx}\n")
                    f.write(f"Input events: {len(data['input_events'])}\n")
                    f.write(f"Output events: {len(data['output_events'])}\n")
                    f.write(f"Event ratio: {len(data['output_events'])/len(data['input_events']):.3f}\n")
                    f.write(f"Sensor size: {sensor_size}\n")
                    f.write(f"Segment duration: {segment_duration_us/1000}ms\n")
                    
                    # Voxel statistics
                    input_voxel = data['input_voxel']
                    output_voxel = data['output_voxel']
                    f.write(f"\n📈 Voxel Statistics:\n")
                    f.write(f"Input - mean: {input_voxel.mean():.4f}, std: {input_voxel.std():.4f}\n")
                    f.write(f"Output - mean: {output_voxel.mean():.4f}, std: {output_voxel.std():.4f}\n")
                    
            except Exception as e:
                self.logger.warning(f"Debug visualization failed for segment {segment_idx}: {e}")
                continue
        
        self.logger.info(f"🎨 Inference debug visualization saved to: {debug_output_dir}")
    
    def _load_model(self, model_config: dict, device: str):
        """Load trained UNet3D model using training factory"""
        try:
            # Use training factory to create model
            config = {'model': model_config}
            training_factory = TrainingFactory(config)
            model = training_factory.create_model()
            
            # Load checkpoint
            checkpoint = torch.load(model_config['path'], map_location=device)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.to(device)
            
            self.logger.info(f"Model loaded successfully: {model_config.get('name', 'UNet3D')}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _extract_segment_events(self, events_np: np.ndarray, segment_idx: int, num_segments: int) -> np.ndarray:
        """Extract specific segment events from full event array"""
        if len(events_np) == 0:
            return events_np
        
        t_min = events_np[:, 0].min()
        t_max = events_np[:, 0].max()
        total_duration = t_max - t_min
        
        segment_start = t_min + segment_idx * (total_duration / num_segments)
        segment_end = t_min + (segment_idx + 1) * (total_duration / num_segments)
        
        mask = (events_np[:, 0] >= segment_start) & (events_np[:, 0] < segment_end)
        return events_np[mask]
    
    def _save_events_to_h5(self, events_np: np.ndarray, output_path: str):
        """Save events to H5 file in DSEC format"""
        import h5py
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            events_group = f.create_group('events')
            
            if len(events_np) > 0:
                # Use gzip compression level 9 to match input file format
                events_group.create_dataset('t', data=events_np[:, 0].astype(np.int64), compression='gzip', compression_opts=9)
                events_group.create_dataset('x', data=events_np[:, 1].astype(np.uint16), compression='gzip', compression_opts=9)  # Fix: uint16 to match input
                events_group.create_dataset('y', data=events_np[:, 2].astype(np.uint16), compression='gzip', compression_opts=9)  # Fix: uint16 to match input
                events_group.create_dataset('p', data=events_np[:, 3].astype(np.int8), compression='gzip', compression_opts=9)
            else:
                # Empty datasets with same compression
                events_group.create_dataset('t', data=np.array([], dtype=np.int64), compression='gzip', compression_opts=9)
                events_group.create_dataset('x', data=np.array([], dtype=np.uint16), compression='gzip', compression_opts=9)
                events_group.create_dataset('y', data=np.array([], dtype=np.uint16), compression='gzip', compression_opts=9)
                events_group.create_dataset('p', data=np.array([], dtype=np.int8), compression='gzip', compression_opts=9)
    
    def _generate_inference_visualization(self, input_events: np.ndarray, output_events: np.ndarray, viz_config: dict):
        """Generate visualization for inference results (optional)"""
        try:
            from src.data_processing.professional_visualizer import visualize_complete_pipeline
            
            output_dir = viz_config.get('output_dir', 'inference_debug')
            segment_idx = viz_config.get('segment_to_visualize', 1)
            
            self.logger.info(f"Generating inference visualization in: {output_dir}")
            
            # Note: This is a simplified version - full visualization would need
            # to reconstruct voxels from events for comparison
            
        except ImportError:
            self.logger.warning("Visualization not available (missing dependencies)")
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
    
    def _trigger_test_debug_visualization(self, batch_idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                                        batch: dict, debug_dir: str, iteration: int):
        """
        """
        try:
            from pathlib import Path
            

            iteration_dir = Path(debug_dir) / f"test_optimized_iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🐛 TEST MODE: Generating debug visualizations for Batch {batch_idx}")
            self.logger.info(f"🐛 Output directory: {iteration_dir}")
            

            input_voxel = inputs[0, 0].cpu()
            target_voxel = targets[0, 0].cpu() # (8, H, W)
            

            from src.data_processing.decode import voxel_to_events
            

            input_events_np = voxel_to_events(input_voxel, total_duration=20000, sensor_size=(480, 640))
            target_events_np = voxel_to_events(target_voxel, total_duration=20000, sensor_size=(480, 640))
            

            from src.data_processing.professional_visualizer import visualize_events, visualize_voxel
            


            input_events_dir = iteration_dir / "1_input_events"
            input_events_dir.mkdir(exist_ok=True)
            visualize_events(input_events_np, sensor_size=(480, 640), output_dir=str(input_events_dir), 
                           name="test_input_events", num_time_slices=8)
            

            input_voxel_dir = iteration_dir / "3_input_voxel"
            input_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(input_voxel, sensor_size=(480, 640), output_dir=str(input_voxel_dir), 
                          name="test_input_voxel", duration_ms=20)
            


            target_events_dir = iteration_dir / "4_target_events"
            target_events_dir.mkdir(exist_ok=True)
            visualize_events(target_events_np, sensor_size=(480, 640), output_dir=str(target_events_dir), 
                           name="test_target_events", num_time_slices=8)
            

            target_voxel_dir = iteration_dir / "6_target_voxel"
            target_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(target_voxel, sensor_size=(480, 640), output_dir=str(target_voxel_dir), 
                          name="test_target_voxel", duration_ms=20)
            
            self.logger.info(f"🐛 TEST: Generated input and target visualizations (1,3,4,6/9) in {iteration_dir}")
            
        except Exception as e:
            self.logger.warning(f"🐛 TEST: Debug visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _trigger_test_model_output_visualization(self, batch_idx: int, inputs: torch.Tensor, outputs: torch.Tensor, 
                                               debug_dir: str, iteration: int, model):
        """
        """
        try:
            from pathlib import Path
            
            iteration_dir = Path(debug_dir) / f"test_optimized_iter_{iteration:03d}"
            

            input_voxel = inputs[0, 0].cpu()   # (8, H, W)
            output_voxel = outputs[0, 0].cpu()
            

            is_true_residual = hasattr(model, 'get_residual') or 'TrueResidualUNet3D' in str(type(model))
            
            if is_true_residual:

                residual_voxel = output_voxel - input_voxel  # output = input + residual
                
                self.logger.info(f"🐛 TEST: True residual learning detected:")
                self.logger.info(f"🐛   Input mean: {input_voxel.mean():.4f}, std: {input_voxel.std():.4f}")
                self.logger.info(f"🐛   Residual mean: {residual_voxel.mean():.4f}, std: {residual_voxel.std():.4f}")
                self.logger.info(f"🐛   Output mean: {output_voxel.mean():.4f}, std: {output_voxel.std():.4f}")
                self.logger.info(f"🐛   Identity check: output ≈ input + residual = {torch.allclose(output_voxel, input_voxel + residual_voxel, atol=1e-6)}")
                

                residual_voxel_dir = iteration_dir / "8_residual_voxel"
                residual_voxel_dir.mkdir(exist_ok=True)
                
                from src.data_processing.professional_visualizer import visualize_voxel
                visualize_voxel(residual_voxel, sensor_size=(480, 640), output_dir=str(residual_voxel_dir), 
                              name="test_residual_voxel", duration_ms=20)
                

                if residual_voxel.abs().sum() > 1e-6:
                    from src.data_processing.decode import voxel_to_events
                    residual_events_np = voxel_to_events(residual_voxel, total_duration=20000, sensor_size=(480, 640))
                    
                    residual_events_dir = iteration_dir / "8_residual_events"
                    residual_events_dir.mkdir(exist_ok=True)
                    
                    from src.data_processing.professional_visualizer import visualize_events
                    visualize_events(residual_events_np, sensor_size=(480, 640), output_dir=str(residual_events_dir), 
                                   name="test_residual_events", num_time_slices=8)
            

            from src.data_processing.decode import voxel_to_events
            output_events_np = voxel_to_events(output_voxel, total_duration=20000, sensor_size=(480, 640))
            
            from src.data_processing.professional_visualizer import visualize_events, visualize_voxel
            

            output_events_dir = iteration_dir / "7_output_events"
            output_events_dir.mkdir(exist_ok=True)
            visualize_events(output_events_np, sensor_size=(480, 640), output_dir=str(output_events_dir), 
                           name="test_output_events", num_time_slices=8)
            

            output_voxel_dir = iteration_dir / "9_output_voxel"
            output_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(output_voxel, sensor_size=(480, 640), output_dir=str(output_voxel_dir), 
                          name="test_output_voxel", duration_ms=20)
            
            folder_count = "8" if is_true_residual else "6"
            self.logger.info(f"🐛 TEST: Generated final output visualizations (7,9/{folder_count}) in {iteration_dir}")
            if is_true_residual:
                self.logger.info(f"🐛 TEST: Generated residual visualizations (8/8) in {iteration_dir}")
                self.logger.info(f"🐛 TEST: All debug visualizations completed! (8 folders total: 1,3,4,6,7,8,9)")
            else:
                self.logger.info(f"🐛 TEST: All debug visualizations completed! (6 folders total: 1,3,4,6,7,9)")
            

            self._generate_test_debug_summary(iteration_dir, batch_idx, iteration, is_true_residual)
            
        except Exception as e:
            self.logger.warning(f"🐛 TEST: Model output visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _generate_test_debug_summary(self, iteration_dir: Path, batch_idx: int, iteration: int, is_true_residual: bool = False):
        try:
            summary_file = iteration_dir / "debug_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write(f"TEST Mode Debug Visualization Summary\n")
                f.write(f"====================================\n")
                f.write(f"Test Iteration: {iteration}\n")
                f.write(f"Batch: {batch_idx}\n")
                f.write(f"Mode: Testing (No Training)\n")
                f.write(f"True Residual Learning: {'Yes' if is_true_residual else 'No'}\n")
                
                if is_true_residual:
                    f.write(f"\n8 Visualization Folders (True Residual Learning):\n")
                    f.write(f"1. 1_input_events/          - Input events (3D+2D+temporal) comprehensive\n")
                    f.write(f"2. 3_input_voxel/           - Input voxel temporal bins\n")
                    f.write(f"3. 4_target_events/         - Target events (3D+2D+temporal) comprehensive\n") 
                    f.write(f"4. 6_target_voxel/          - Target voxel temporal bins\n")
                    f.write(f"5. 7_output_events/         - Model output events (input + residual) comprehensive\n")
                    f.write(f"6. 8_residual_voxel/        - Learned residual voxel temporal bins\n")
                    f.write(f"7. 8_residual_events/       - Learned residual events (if non-zero)\n")
                    f.write(f"8. 9_output_voxel/          - Model output voxel temporal bins\n")
                else:
                    f.write(f"\n6 Visualization Folders (Standard Model):\n")
                    f.write(f"1. 1_input_events/          - Input events (3D+2D+temporal) comprehensive\n")
                    f.write(f"2. 3_input_voxel/           - Input voxel temporal bins\n")
                    f.write(f"3. 4_target_events/         - Target events (3D+2D+temporal) comprehensive\n") 
                    f.write(f"4. 6_target_voxel/          - Target voxel temporal bins\n")
                    f.write(f"5. 7_output_events/         - Model output events (3D+2D+temporal) comprehensive\n")
                    f.write(f"6. 9_output_voxel/          - Model output voxel temporal bins\n")
                
                f.write(f"\nTesting Mode: Evaluate checkpoint performance on test data\n")
                f.write(f"No training - only inference and visualization\n")
                f.write(f"Same data processing pipeline as training validation\n")
                
            self.logger.info(f"🐛 TEST: Debug summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.warning(f"🐛 TEST: Failed to generate debug summary: {e}")
    
    def _save_segment_output(self, outputs: torch.Tensor, input_file_path: Path, output_file_name: str,
                           output_dir: Path, file_idx: int, segment_idx: int, sensor_size: tuple,
                           segment_duration_us: int, num_bins: int, num_segments: int, processed_files: set):
        """
        Save model output for a single segment - optimized for 100ms fixed input
        Uses in-memory buffering to eliminate disk I/O waste
        
        Args:
            outputs: Model output tensor (1, 1, 8, H, W)
            input_file_path: Original input file path (for reference only)
            output_file_name: Output file name
            output_dir: Output directory
            file_idx: File index
            segment_idx: Segment index within file (0-4)
            sensor_size: Sensor size (H, W)  
            segment_duration_us: Duration per segment (20ms)
            num_bins: Number of temporal bins (8)
            num_segments: Total segments per file (5)
            processed_files: Set to track processed files
        """
        try:
            # Initialize file buffer if not exists
            if not hasattr(self, '_segment_buffers'):
                self._segment_buffers = {}  # file_idx -> [seg0_events, seg1_events, ...]
            
            if file_idx not in self._segment_buffers:
                self._segment_buffers[file_idx] = [None] * num_segments
            
            # Remove batch and channel dimensions: (1, 1, 8, H, W) -> (8, H, W)
            output_voxel = outputs[0, 0].cpu()
            
            # Convert voxel back to events
            output_events = voxel_to_events(
                output_voxel, 
                total_duration=segment_duration_us,  # 20ms per segment
                sensor_size=sensor_size
            )
            
            # Fixed 100ms input: simple timestamp mapping without H5 re-reading
            if len(output_events) > 0:
                # For 100ms fixed input: segment timestamps are predictable
                # segment_idx=0: 0-20ms, segment_idx=1: 20-40ms, etc.
                segment_start_us = segment_idx * segment_duration_us  # 0, 20000, 40000, 60000, 80000
                segment_end_us = (segment_idx + 1) * segment_duration_us
                
                # Simple linear remapping to correct time range
                if len(output_events) > 0:
                    t_decoded_min = output_events[:, 0].min()
                    t_decoded_max = output_events[:, 0].max()
                    t_decoded_range = t_decoded_max - t_decoded_min
                    
                    if t_decoded_range > 0:
                        # Linear rescale to 20ms segment
                        scale = segment_duration_us / t_decoded_range
                        output_events[:, 0] = segment_start_us + (output_events[:, 0] - t_decoded_min) * scale
                    else:
                        # All events at same time - place in middle of segment
                        output_events[:, 0] = (segment_start_us + segment_end_us) / 2
            
            # Store in memory buffer (no disk I/O)
            self._segment_buffers[file_idx][segment_idx] = output_events
            
            # Check if this is the last segment of the file
            if segment_idx == num_segments - 1:
                # Merge all segments in memory and save H5 directly
                self._merge_and_save_file_segments_optimized(
                    file_idx, output_file_name, output_dir
                )
                processed_files.add(file_idx)
                
                # Clean up memory buffer for this file
                del self._segment_buffers[file_idx]
                
                self.logger.info(f"✅ Saved complete file: {output_file_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to save segment output for {output_file_name}: {e}")
    
    def _merge_and_save_file_segments_optimized(self, file_idx: int, output_file_name: str, output_dir: Path):
        """
        Merge all segments of a file in memory and save as H5 (no temporary files)
        
        Args:
            file_idx: File index
            output_file_name: Output file name
            output_dir: Output directory
        """
        try:
            all_segments = []
            
            # Merge segments from memory buffer (no disk I/O)
            for segment_events in self._segment_buffers[file_idx]:
                if segment_events is not None and len(segment_events) > 0:
                    all_segments.append(segment_events)
            
            # Merge segments
            if all_segments:
                final_events = np.vstack(all_segments)
                # Sort by timestamp (important for correct temporal order)
                final_events = final_events[np.argsort(final_events[:, 0])]
            else:
                final_events = np.empty((0, 4))
            
            # Save as H5 file directly (no temporary files)
            output_file_path = output_dir / output_file_name
            self._save_events_to_h5(final_events, str(output_file_path))
            
        except Exception as e:
            self.logger.error(f"Failed to merge segments for {output_file_name}: {e}")


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Event-Voxel Denoising with pytorch-3dunet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py train --config configs/train_config.yaml
  
  # Testing (normal)
  python main.py test --config configs/test_config.yaml
  
  # Testing (baseline: encode-decode only)  
  python main.py test --config configs/test_config.yaml --baseline
  
  # Inference
  python main.py inference --config configs/inference_config.yaml \\
    --input testdata/noisy_events.h5 --output results/denoised_events.h5
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Training mode
    train_parser = subparsers.add_parser('train', help='Train denoising model')
    train_parser.add_argument('--config', required=True, 
                            help='Path to training configuration YAML')
    train_parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode with detailed visualization (runs only 1-2 iterations)')
    train_parser.add_argument('--debug-dir', type=str, default='debug_output',
                            help='Directory for debug visualizations (default: debug_output)')
    
    # Testing mode
    test_parser = subparsers.add_parser('test', help='Evaluate trained model')
    test_parser.add_argument('--config', required=True,
                           help='Path to test configuration YAML')
    test_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode with detailed visualization (runs only 1-2 batches)')
    test_parser.add_argument('--debug-dir', type=str, default='debug_output',
                           help='Directory for debug visualizations (default: debug_output)')
    test_parser.add_argument('--baseline', action='store_true',
                           help='Enable baseline mode: skip UNet processing, only encode-decode (saves to *baseline directory)')
    
    # Inference mode
    inference_parser = subparsers.add_parser('inference', help='Batch denoise event files (no ground truth)')
    inference_parser.add_argument('--config', required=True,
                                help='Path to inference configuration YAML')
    inference_parser.add_argument('--input', required=False,
                                help='Input directory or file (if not specified, use config)')
    inference_parser.add_argument('--output', required=False,
                                help='Output directory or file (if not specified, use config)')
    inference_parser.add_argument('--debug', action='store_true',
                                help='Enable debug visualization mode (no ground truth comparisons)')
    inference_parser.add_argument('--debug-dir', default='debug_output',
                                help='Debug output directory (default: debug_output)')
    inference_parser.add_argument('--baseline', action='store_true',
                                help='Enable baseline mode: skip UNet processing, only encode-decode (saves to *baseline directory)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
    
    # Initialize pipeline
    pipeline = EventVoxelPipeline()
    
    # Execute requested mode
    if args.mode == 'train':
        return pipeline.train_mode(args.config, getattr(args, 'debug', False), getattr(args, 'debug_dir', 'debug_output'))
    elif args.mode == 'test':
        return pipeline.test_mode(args.config, getattr(args, 'debug', False), getattr(args, 'debug_dir', 'debug_output'), getattr(args, 'baseline', False))
    elif args.mode == 'inference':
        return pipeline.inference_mode(args.config, getattr(args, 'input', None), getattr(args, 'output', None), 
                                      getattr(args, 'debug', False), getattr(args, 'debug_dir', 'debug_output'), getattr(args, 'baseline', False))
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == '__main__':
    sys.exit(main())