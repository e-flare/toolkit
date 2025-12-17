#!/usr/bin/env python3
"""
Memory-efficient DSEC Dataset Loader for EventMamba-FX
Loads events in chunks to avoid memory issues
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import hdf5plugin  # Required for DSEC compressed H5 files
import numpy as np
import os
import glob
import random
from typing import List, Tuple, Dict


class DSECEventDatasetEfficient(Dataset):
    """
    Memory-efficient DSEC Event Dataset that loads time windows without loading entire files
    🚨 CRITICAL FIX: Removed sequence_length limit to return all events in time window
    """

    def __init__(self, dsec_path: str, flare_path: str, time_window_us: int = 1000000,
                 camera_side: str = 'left', max_files: int = None):
        """
        Args:
            dsec_path: Path to background events directory (data/bg_events)
            flare_path: Path to flare events (currently unused)
            time_window_us: Time window in microseconds (1 second = 1000000)
            camera_side: 'left' or 'right' camera (not used for simple bg_events structure)
            max_files: Maximum number of files to use (None = use all files)
        """
        self.dsec_path = dsec_path
        self.flare_path = flare_path
        self.time_window_us = time_window_us
        self.camera_side = camera_side
        self.max_files = max_files

        # Find all available H5 files
        self.h5_files = self._find_h5_files()
        
        if max_files is not None:
            print(f"Found {len(self.h5_files)} background event files, using first {min(len(self.h5_files), max_files)}")
            # Limit files for memory safety
            self.h5_files = self.h5_files[:max_files]
        else:
            print(f"Found {len(self.h5_files)} background event files, using all files")

        # Pre-load lightweight metadata without loading full arrays
        self.file_metadata = self._load_file_metadata_efficient()

        # Calculate total number of valid 1-second windows
        self.total_windows = sum(meta['num_windows'] for meta in self.file_metadata)
        print(f"Total available 1-second windows: {self.total_windows}")

    def _find_h5_files(self) -> List[str]:
        """Find all H5 files in background events directory"""
        pattern = os.path.join(self.dsec_path, "*.h5")
        files = glob.glob(pattern)
        return sorted(files)

    def _load_file_metadata_efficient(self) -> List[Dict]:
        """Load only timestamp info to determine windows, without loading full arrays"""
        metadata = []
        
        for file_path in self.h5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    # Only read first and last timestamps
                    t_min = f['events/t'][0]
                    t_max = f['events/t'][-1]
                    num_events = len(f['events/t'])
                    
                    # Calculate number of 1-second windows available
                    duration_us = t_max - t_min
                    num_windows = max(1, int(duration_us // self.time_window_us))
                    
                    metadata.append({
                        'file_path': file_path,
                        't_min': t_min,
                        't_max': t_max,
                        'num_events': num_events,
                        'num_windows': num_windows,
                        'duration_s': duration_us / 1000000
                    })
                    
                    print(f"File: {os.path.basename(file_path)}, Events: {num_events:,}, Windows: {num_windows}, Duration: {duration_us/1000000:.1f}s")
                    
            except Exception as e:
                print(f"Error loading metadata from {file_path}: {e}")
                continue
                
        return metadata

    def _load_time_window_efficient(self, file_path: str, window_start_us: int, window_end_us: int) -> np.ndarray:
        """
        Load only events within the specified time window using binary search for efficiency
        """
        with h5py.File(file_path, 'r') as f:
            t_array = f['events/t']
            
            # Binary search for start and end indices
            start_idx = np.searchsorted(t_array, window_start_us, side='left')
            end_idx = np.searchsorted(t_array, window_end_us, side='right')
            
            # Load only the events in this time window
            if start_idx < end_idx:
                x = f['events/x'][start_idx:end_idx]
                y = f['events/y'][start_idx:end_idx]
                t = f['events/t'][start_idx:end_idx]
                p = f['events/p'][start_idx:end_idx]
                
                x = x.astype(np.int32)
                y = y.astype(np.int32)  
                t = t.astype(np.int64)
                p = np.where(p > 0, 1, -1).astype(np.int8)
                
                # Stack into format [x, y, t, p] with proper dtypes
                events = np.column_stack([x, y, t, p]).astype(np.float64)
                return events
            else:
                # Return empty array if no events in window
                return np.empty((0, 4), dtype=np.float64)

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Find which file and window this index corresponds to
        current_idx = 0
        for file_meta in self.file_metadata:
            if current_idx + file_meta['num_windows'] > idx:
                # This file contains our target window
                window_idx = idx - current_idx
                break
            current_idx += file_meta['num_windows']
        else:
            # Fallback to first file, first window
            file_meta = self.file_metadata[0]
            window_idx = 0

        # Calculate time window boundaries
        window_start = file_meta['t_min'] + window_idx * self.time_window_us
        window_end = window_start + self.time_window_us

        # Load events for this specific time window
        events = self._load_time_window_efficient(
            file_meta['file_path'], window_start, window_end
        )

        # 🚨 CRITICAL FIX: Return ALL events in time window, no artificial limits
        # This allows epoch-level processing to work with complete sequences
        if len(events) == 0:
            # Handle empty window - return empty array
            events = np.empty((0, 4), dtype=np.float64)
        else:
            # 🚨 CRITICAL: Normalize timestamps to start from 0
            # This is essential for proper merging with flare events
            t_min = events[:, 2].min()
            events[:, 2] = events[:, 2] - t_min

        # Return raw numpy array for epoch-level processing
        # Labels will be generated at epoch level when merging with flare events
        return events


def create_dsec_dataloaders_efficient(config):
    """
    Create efficient DSEC dataloaders that don't load entire files into memory
    """
    
    # Create dataset with no file limit (use all files)
    train_dataset = DSECEventDatasetEfficient(
        dsec_path=config['data']['dsec_path'],
        flare_path=config['data']['flare_path'],
        time_window_us=config['data']['time_window_us'],
        camera_side='left',
        max_files=None  # Use all available files
    )
    
    # For validation and test, use same dataset but different sampling
    val_dataset = train_dataset  # Could create separate logic later
    test_dataset = train_dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
    )
    
    return train_loader, val_loader, test_loader