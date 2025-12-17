"""
EventVoxelDataset - PyTorch Dataset for Events↔Voxel conversion with pytorch-3dunet
Implements segmented processing: 100ms → 5×20ms segments for memory optimization
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from pathlib import Path
from typing import List, Tuple, Dict, Union
import logging

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_processing.encode import load_h5_events, events_to_voxel

class EventVoxelDataset(Dataset):
    """
    PyTorch Dataset for event-based denoising with voxel representation
    
    Key Features:
    - Segmented processing: 100ms → 5×20ms segments (memory optimization)
    - Fixed temporal resolution: 20ms/8bins = 2.5ms per bin (training consistency)
    - pytorch-3dunet compatibility: (C, Z, Y, X) format
    - Paired data: noisy input events ↔ clean ground truth events
    - File-level caching: 5x I/O optimization (each file read once for 5 segments)
    """
    
    def __init__(self, 
                 noisy_events_dir: str,
                 clean_events_dir: str, 
                 sensor_size: Tuple[int, int] = (480, 640),
                 segment_duration_us: int = 20000,  # 20ms per segment
                 num_bins: int = 8,                 # 8 bins per 20ms segment
                 num_segments: int = 5,             # 5 segments per 100ms file
                 transform=None):
        """
        Initialize EventVoxelDataset for deflare task
        
        Args:
            noisy_events_dir: Directory containing input events with flare (background_with_flare_events)
            clean_events_dir: Directory containing clean ground truth events (background_with_light_events)
            sensor_size: (Height, Width) of sensor resolution
            segment_duration_us: Duration of each segment in microseconds (20ms)
            num_bins: Number of temporal bins per segment (8)
            num_segments: Number of segments per 100ms file (5)  
            transform: Optional data transforms
        """
        self.noisy_events_dir = Path(noisy_events_dir)
        self.clean_events_dir = Path(clean_events_dir)
        self.sensor_size = sensor_size
        self.segment_duration_us = segment_duration_us
        self.num_bins = num_bins
        self.num_segments = num_segments
        self.transform = transform
        

        self._current_file_idx = None
        self._current_events = None  # (noisy_events, clean_events)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Scan and match H5 file pairs
        self.file_pairs = self._scan_and_match_files()
        
        # Calculate total samples: each 100ms file → 5 segments
        self.total_samples = len(self.file_pairs) * self.num_segments
        
        self.logger.info(f"EventVoxelDataset initialized:")
        self.logger.info(f"  - {len(self.file_pairs)} H5 file pairs found")
        self.logger.info(f"  - {self.total_samples} total samples ({self.num_segments} segments per file)")
        self.logger.info(f"  - Segment config: {segment_duration_us/1000}ms/{num_bins}bins = {segment_duration_us/num_bins/1000:.2f}ms per bin")
        self.logger.info(f"  - Sensor size: {sensor_size}")
    
    def _scan_and_match_files(self) -> List[Tuple[str, str]]:
        """
        Scan directories and match noisy/clean H5 file pairs
        
        Returns:
            List of (noisy_file_path, clean_file_path) tuples
        """
        if not self.noisy_events_dir.exists():
            raise ValueError(f"Noisy events directory not found: {self.noisy_events_dir}")
        if not self.clean_events_dir.exists():
            raise ValueError(f"Clean events directory not found: {self.clean_events_dir}")
        
        # Get all H5 files from both directories
        noisy_files = list(self.noisy_events_dir.glob("*.h5"))
        clean_files = list(self.clean_events_dir.glob("*.h5"))
        
        # Extract base names for matching (deflare dataset naming convention)
        # Flare files: composed_XXXXX_bg_flare.h5 → composed_XXXXX
        # Clean files: composed_XXXXX_bg_light.h5 → composed_XXXXX
        noisy_names = {}
        for f in noisy_files:
            # Remove _bg_flare suffix to get base name
            base_name = f.stem.replace('_bg_flare', '')
            noisy_names[base_name] = f
        
        clean_names = {}
        for f in clean_files:
            # Remove _bg_light suffix to get base name  
            base_name = f.stem.replace('_bg_light', '')
            clean_names[base_name] = f
        
        # Find matching pairs
        file_pairs = []
        for base_name in noisy_names.keys():
            if base_name in clean_names:
                file_pairs.append((str(noisy_names[base_name]), str(clean_names[base_name])))
            else:
                self.logger.warning(f"No matching clean file for noisy file: {base_name}")
        
        if not file_pairs:
            raise ValueError("No matching H5 file pairs found!")
        
        self.logger.info(f"Found {len(file_pairs)} matching H5 file pairs")
        return file_pairs
    
    def _extract_segment_events(self, events_np: np.ndarray, segment_idx: int) -> np.ndarray:
        """
        Extract a specific 20ms segment from 100ms events (optimized for uniform data)
        
        Args:
            events_np: Full event array (N, 4) [t, x, y, p]
            segment_idx: Segment index (0-4)
            
        Returns:
            Segment events array (M, 4) [t, x, y, p]
        """
        if len(events_np) == 0:
            return events_np
        
        # Optimized for uniform 0-100ms data: use fixed 20ms segments
        # segment_idx=0: 0-20ms, segment_idx=1: 20-40ms, etc.
        t_min = events_np[:, 0].min()
        t_max = events_np[:, 0].max()
        total_duration = t_max - t_min
        
        # Fixed segment boundaries (20ms each for 100ms total)
        segment_duration = total_duration / self.num_segments  # Should be ~20ms for uniform data
        segment_start = t_min + segment_idx * segment_duration
        segment_end = segment_start + segment_duration
        
        # Extract events in this time window
        mask = (events_np[:, 0] >= segment_start) & (events_np[:, 0] < segment_end)
        segment_events = events_np[mask]
        
        return segment_events
    
    def __len__(self) -> int:
        """Return total number of samples (file_pairs × segments_per_file)"""
        return self.total_samples
    
    def _get_events_smart_cached(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        
        
        Args:
            file_idx: File index
            
        Returns:
            Tuple of (noisy_events, clean_events)
        """

        if self._current_file_idx == file_idx and self._current_events is not None:

            return self._current_events
        

        if self._current_file_idx is not None:
            self.logger.debug(f"File switched {self._current_file_idx}→{file_idx}, clearing cache")
            self._current_events = None
        

        noisy_file_path, clean_file_path = self.file_pairs[file_idx]
        
        noisy_events = load_h5_events(noisy_file_path)
        clean_events = load_h5_events(clean_file_path)
        

        self._current_file_idx = file_idx
        self._current_events = (noisy_events, clean_events)
        
        self.logger.debug(f"Loaded file {file_idx}: {len(noisy_events)} noisy, {len(clean_events)} clean events")
        
        return self._current_events

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample with smart single-file cache
        
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with 'raw' and 'label' voxels (1, num_bins, H, W)
        """
        # Calculate file and segment
        file_idx = idx // self.num_segments
        segment_idx = idx % self.num_segments
        
        try:

            noisy_events, clean_events = self._get_events_smart_cached(file_idx)
            
            # Extract the specific 20ms segment
            noisy_segment = self._extract_segment_events(noisy_events, segment_idx)
            clean_segment = self._extract_segment_events(clean_events, segment_idx)
            
            # Convert to voxel grids (8 bins for 20ms segment)
            noisy_voxel = events_to_voxel(
                noisy_segment,
                num_bins=self.num_bins,
                sensor_size=self.sensor_size,
                fixed_duration_us=self.segment_duration_us
            )  # Shape: (8, H, W)
            
            clean_voxel = events_to_voxel(
                clean_segment,
                num_bins=self.num_bins,
                sensor_size=self.sensor_size,
                fixed_duration_us=self.segment_duration_us
            )  # Shape: (8, H, W)
            
            # Add channel dimension for pytorch-3dunet: (8, H, W) → (1, 8, H, W)
            # pytorch-3dunet expects (C, Z, Y, X) where:
            # C = channels (1), Z = depth/time (8), Y = height, X = width
            noisy_voxel = noisy_voxel.unsqueeze(0)  # (1, 8, H, W)
            clean_voxel = clean_voxel.unsqueeze(0)  # (1, 8, H, W)
            
            # Apply transforms if specified
            if self.transform:
                noisy_voxel = self.transform(noisy_voxel)
                clean_voxel = self.transform(clean_voxel)
            
            return {
                'raw': noisy_voxel,      # Input for pytorch-3dunet
                'label': clean_voxel     # Ground truth for pytorch-3dunet
            }
            
        except Exception as e:
            self.logger.error(f"Error processing sample {idx} (file {file_idx}, segment {segment_idx}): {e}")
            # Return zeros as fallback
            zero_voxel = torch.zeros((1, self.num_bins, self.sensor_size[0], self.sensor_size[1]))
            return {
                'raw': zero_voxel,
                'label': zero_voxel
            }
    
    def clear_cache(self):
        if self._current_events is not None:
            self.logger.debug(f"Manually cleared cache for file {self._current_file_idx}")
            self._current_file_idx = None
            self._current_events = None


class EventVoxelInferenceDataset(Dataset):
    """
    Dataset for inference mode - processes single H5 files without paired labels
    Used for standalone denoising inference
    """
    
    def __init__(self, 
                 events_file: str,
                 sensor_size: Tuple[int, int] = (480, 640),
                 segment_duration_us: int = 20000,
                 num_bins: int = 8,
                 num_segments: int = 5):
        """
        Initialize inference dataset for a single events file
        
        Args:
            events_file: Path to H5 events file  
            sensor_size: (Height, Width) of sensor resolution
            segment_duration_us: Duration per segment (20ms)
            num_bins: Number of bins per segment (8)
            num_segments: Number of segments per file (5)
        """
        self.events_file = Path(events_file)
        self.sensor_size = sensor_size
        self.segment_duration_us = segment_duration_us
        self.num_bins = num_bins
        self.num_segments = num_segments
        
        if not self.events_file.exists():
            raise ValueError(f"Events file not found: {events_file}")
        
        # Load the full events data once
        self.events_np = load_h5_events(str(self.events_file))
        
        logging.info(f"EventVoxelInferenceDataset: {len(self.events_np)} events, {num_segments} segments")
    
    def _extract_segment_events(self, segment_idx: int) -> np.ndarray:
        """Extract specific segment events"""
        if len(self.events_np) == 0:
            return self.events_np
        
        t_min = self.events_np[:, 0].min()
        t_max = self.events_np[:, 0].max()
        total_duration = t_max - t_min
        
        segment_start = t_min + segment_idx * (total_duration / self.num_segments)
        segment_end = t_min + (segment_idx + 1) * (total_duration / self.num_segments)
        
        mask = (self.events_np[:, 0] >= segment_start) & (self.events_np[:, 0] < segment_end)
        return self.events_np[mask]
    
    def __len__(self) -> int:
        """Return number of segments"""
        return self.num_segments
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a segment voxel for inference
        
        Args:
            idx: Segment index (0-4)
            
        Returns:
            Voxel tensor (1, num_bins, H, W)
        """
        segment_events = self._extract_segment_events(idx)
        
        voxel = events_to_voxel(
            segment_events,
            num_bins=self.num_bins,
            sensor_size=self.sensor_size,
            fixed_duration_us=self.segment_duration_us
        )
        
        # Add channel dimension: (8, H, W) → (1, 8, H, W)
        return voxel.unsqueeze(0)