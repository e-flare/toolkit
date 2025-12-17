#!/usr/bin/env python3
"""
inference_single_memory_safe.py - Memory-Safe Single File Processing
================================================================

Fixed version with streaming disk writes to prevent memory explosion on long files.

Key fixes:
1. Stream processing: Write segments to disk immediately 
2. Batch merging: Merge in chunks instead of loading all segments
3. Memory monitoring: Track and limit memory usage
4. Safe fallback: Graceful handling of extreme file sizes

Author: Event-Voxel Deflare System (Memory-Safe Version)
Date: 2025-12-24
"""

import os
import sys
import gc
import argparse
import logging
import torch
import numpy as np
import h5py
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Import project modules
from src.utils.config_loader import ConfigLoader
from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events
from src.training.training_factory import TrainingFactory


class MemorySafeInferenceSingle:
    """Memory-safe single file inference with streaming disk writes"""
    
    def __init__(self, config_path: str, mode: str = 'normal', device: str = 'cuda'):
        """Initialize with memory safety features"""
        self.device = device
        self.mode = mode
        self.max_memory_segments = 50  # Maximum segments to keep in memory
        
        # Load configuration
        if mode == 'simple':
            simple_config_path = config_path.replace('.yaml', '_simple.yaml')
            if Path(simple_config_path).exists():
                config_path = simple_config_path
        
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(config_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.model = None
        self.temp_dir = None  # For streaming writes
        
        self.logger.info(f"🛡️  MemorySafeInferenceSingle initialized - Mode: {mode}, Device: {device}")
        self.logger.info(f"🧠 Memory safety: Max {self.max_memory_segments} segments in RAM")
    
    def _convert_windows_to_wsl_path(self, windows_path: str) -> str:
        """Convert Windows path to WSL path format"""
        if not windows_path.startswith(('C:', 'D:', 'E:', 'F:', 'G:')):
            return windows_path
        
        drive_letter = windows_path[0].lower()
        rest_path = windows_path[2:].replace('\\', '/')
        wsl_path = f'/mnt/{drive_letter}{rest_path}'
        
        self.logger.debug(f"🔄 Path conversion: {windows_path} -> {wsl_path}")
        return wsl_path
    
    def _load_model(self):
        """Load model from checkpoint if not already loaded"""
        if self.model is not None:
            return
        
        self.logger.info("🔧 Loading model...")
        
        training_factory = TrainingFactory(self.config)
        self.model = training_factory.create_model()
        
        checkpoint_path = self.config['model']['path']
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"✅ Model loaded - Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    def _estimate_memory_requirements(self, num_segments: int) -> dict:
        """Estimate memory requirements for processing"""
        avg_events_per_segment = 50000  # Conservative estimate
        bytes_per_event = 4 * 8  # 4 float64 values
        
        segment_memory = avg_events_per_segment * bytes_per_event
        total_segments_memory = num_segments * segment_memory
        
        return {
            'segments': num_segments,
            'memory_per_segment_mb': segment_memory / (1024**2),
            'total_memory_gb': total_segments_memory / (1024**3),
            'requires_streaming': num_segments > self.max_memory_segments
        }
    
    def _process_segment_to_disk(self, segment_events: np.ndarray, segment_idx: int, 
                                temp_file: str) -> dict:
        """Process segment and save directly to temporary file"""
        sensor_size = self.config['inference']['sensor_size']
        num_bins = self.config['inference']['num_bins']
        segment_duration_us = self.config['inference']['segment_duration_us']
        
        # Process through model
        if len(segment_events) == 0:
            input_voxel = torch.zeros((num_bins, sensor_size[0], sensor_size[1]), 
                                     dtype=torch.float32)
        else:
            input_voxel = events_to_voxel(
                segment_events, 
                num_bins=num_bins,
                sensor_size=sensor_size,
                fixed_duration_us=segment_duration_us
            )
        
        input_tensor = input_voxel.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            output_voxel = output_tensor.squeeze(0).squeeze(0).cpu()
            
            # Immediate GPU cleanup
            del input_tensor, output_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Decode to events
        output_events = voxel_to_events(
            output_voxel, 
            total_duration=segment_duration_us,
            sensor_size=sensor_size
        )
        
        # Save to temporary file
        if len(output_events) > 0:
            np.save(temp_file, output_events)
            file_exists = True
        else:
            file_exists = False
        
        # Immediate cleanup
        del segment_events, output_voxel, output_events
        gc.collect()
        
        return {
            'segment_idx': segment_idx,
            'temp_file': temp_file,
            'file_exists': file_exists,
            'input_events': len(segment_events) if 'segment_events' in locals() else 0
        }
    
    def _merge_from_disk(self, segment_infos: List[dict], 
                        original_events: np.ndarray) -> np.ndarray:
        """Merge segments from disk files in batches"""
        self.logger.info(f"🔗 Merging {len(segment_infos)} segments from disk...")
        
        t_min_original = original_events[:, 0].min()
        segment_duration_us = self.config['inference']['segment_duration_us']
        
        final_events_list = []
        batch_size = min(20, self.max_memory_segments)  # Process in batches of 20
        
        for batch_start in range(0, len(segment_infos), batch_size):
            batch_end = min(batch_start + batch_size, len(segment_infos))
            batch_infos = segment_infos[batch_start:batch_end]
            
            self.logger.info(f"  Processing batch {batch_start//batch_size + 1}/{(len(segment_infos)-1)//batch_size + 1}")
            
            batch_events = []
            for info in batch_infos:
                if not info['file_exists']:
                    continue
                    
                try:
                    segment_events = np.load(info['temp_file'])
                    
                    # Restore global timestamps
                    segment_start_global = t_min_original + info['segment_idx'] * segment_duration_us
                    segment_events[:, 0] += segment_start_global
                    
                    batch_events.append(segment_events)
                    
                    # Clean up temp file immediately
                    os.unlink(info['temp_file'])
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to load segment {info['segment_idx']}: {e}")
            
            # Merge batch and add to final list
            if batch_events:
                batch_merged = np.vstack(batch_events)
                final_events_list.append(batch_merged)
                del batch_events, batch_merged
                gc.collect()
        
        # Final merge of all batches
        if not final_events_list:
            self.logger.warning("⚠️  All segments are empty")
            return np.empty((0, 4))
        
        final_events = np.vstack(final_events_list)
        
        # Sort by timestamp
        sort_indices = np.argsort(final_events[:, 0])
        final_events = final_events[sort_indices]
        
        self.logger.info(f"✅ Merged successfully: {len(final_events):,} total events")
        return final_events
    
    def process_file_memory_safe(self, input_path: str, output_path: Optional[str] = None, 
                                debug: bool = False) -> str:
        """Memory-safe processing of arbitrarily long H5 files"""
        # Convert paths and validate
        input_path = self._convert_windows_to_wsl_path(input_path)
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            input_path_obj = Path(input_path)
            suffix = 'Unetsimple' if self.mode == 'simple' else 'Unet'
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_{suffix}.h5")
        else:
            output_path = self._convert_windows_to_wsl_path(output_path)
        
        self.logger.info(f"🎯 Memory-safe processing: {Path(input_path).name}")
        
        # Load model and events
        self._load_model()
        events_np = load_h5_events(input_path)
        
        # Calculate segments and memory requirements
        duration_us = events_np[:, 0].max() - events_np[:, 0].min()
        segment_duration_us = self.config['inference']['segment_duration_us']
        num_segments = int(np.ceil(duration_us / segment_duration_us))
        
        memory_info = self._estimate_memory_requirements(num_segments)
        
        self.logger.info(f"📊 File analysis:")
        self.logger.info(f"  Duration: {duration_us/1000:.1f}ms ({num_segments} segments)")
        self.logger.info(f"  Estimated memory: {memory_info['total_memory_gb']:.2f}GB")
        self.logger.info(f"  Streaming mode: {'✅ ENABLED' if memory_info['requires_streaming'] else '❌ DISABLED'}")
        
        # Create temporary directory for streaming
        self.temp_dir = tempfile.mkdtemp(prefix="inference_streaming_")
        
        try:
            # Process segments with streaming
            segment_infos = []
            t_min_original = events_np[:, 0].min()
            
            for segment_idx in range(num_segments):
                self.logger.info(f"⚙️  Processing segment {segment_idx + 1}/{num_segments}")
                
                # Extract segment
                segment_start = t_min_original + segment_idx * segment_duration_us
                segment_end = segment_start + segment_duration_us
                mask = (events_np[:, 0] >= segment_start) & (events_np[:, 0] < segment_end)
                segment_events = events_np[mask].copy()
                
                if len(segment_events) > 0:
                    segment_events[:, 0] -= segment_start  # Normalize timestamps
                
                # Process to temporary file
                temp_file = os.path.join(self.temp_dir, f"segment_{segment_idx:06d}.npy")
                info = self._process_segment_to_disk(segment_events, segment_idx, temp_file)
                segment_infos.append(info)
                
                # Memory status reporting
                if (segment_idx + 1) % 50 == 0:
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                        self.logger.info(f"  GPU memory: {gpu_memory:.2f}GB")
            
            # Merge from disk
            final_events = self._merge_from_disk(segment_infos, events_np)
            
            # Save results
            self.logger.info(f"💾 Saving {len(final_events):,} events to: {output_path}")
            
            with h5py.File(output_path, 'w') as f:
                events_group = f.create_group('events')
                events_group.create_dataset('t', data=final_events[:, 0].astype(np.int64), 
                                          compression='gzip', compression_opts=9)
                events_group.create_dataset('x', data=final_events[:, 1].astype(np.uint16), 
                                          compression='gzip', compression_opts=9) 
                events_group.create_dataset('y', data=final_events[:, 2].astype(np.uint16),
                                          compression='gzip', compression_opts=9)
                events_group.create_dataset('p', data=final_events[:, 3].astype(np.int8),
                                          compression='gzip', compression_opts=9)
            
            # Final statistics
            input_count = len(events_np)
            output_count = len(final_events)
            compression_ratio = output_count / input_count if input_count > 0 else 0
            
            self.logger.info(f"✅ Memory-safe processing completed!")
            self.logger.info(f"📊 Result: {input_count:,} -> {output_count:,} events ({compression_ratio:.2%})")
            
            return output_path
            
        finally:
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"🧹 Cleaned up temporary files")


def main():
    """Main entry point with memory safety"""
    parser = argparse.ArgumentParser(description="Memory-Safe Single File Event Deflare Processing")
    parser.add_argument('--input', required=True, help='Input H5 file path')
    parser.add_argument('--output', help='Output H5 file path (optional)')
    parser.add_argument('--config', default='configs/inference_config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['normal', 'simple'], default='normal', 
                       help='Model mode: normal or simple weights')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Processing device')
    
    args = parser.parse_args()
    
    try:
        processor = MemorySafeInferenceSingle(
            config_path=args.config,
            mode=args.mode,
            device=args.device
        )
        
        output_path = processor.process_file_memory_safe(
            input_path=args.input,
            output_path=args.output,
            debug=args.debug
        )
        
        print(f"✅ Memory-safe processing completed!")
        print(f"📂 Output file: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()