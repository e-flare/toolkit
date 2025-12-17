import numpy as np
import torch
import h5py
import yaml
import argparse
import os
import sys
from pathlib import Path

# Add event_utils-master to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'event_utils-master'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'event_utils-master', 'lib'))

# Removed buggy import - implement our own simple voxel function

def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "voxel_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_h5_events(file_path):
    """Load events from H5 file in DSEC format
    
    Args:
        file_path: Path to H5 file
        
    Returns:
        events_np: NumPy array (N, 4) with [t, x, y, p]
    """
    with h5py.File(file_path, 'r') as f:
        events_group = f['events']
        t = events_group['t'][:]      # timestamps in microseconds
        x = events_group['x'][:]      # x coordinates  
        y = events_group['y'][:]      # y coordinates
        p = events_group['p'][:]      # polarities (raw format)
        
    # Apply universal polarity rule: 1 is positive, everything else is negative
    # This handles all formats: {0,1}, {-1,1}, {0,1,2}, etc.
    unique_polarities = np.unique(p)
    p_converted = np.where(p == 1, 1, -1)
    
    # Report conversion (only in debug mode)
    pos_count = len(p[p == 1])
    neg_count = len(p) - pos_count
    # Remove verbose logging - only print in debug mode
    # print(f"Applied universal polarity rule: 1→positive, non-1→negative")
    # print(f"Original values: {unique_polarities}")
    # print(f"Converted: {pos_count:,} positive (+1), {neg_count:,} negative (-1)")
        
    # Stack into (N, 4) array: [t, x, y, p]
    events_np = np.column_stack([t, x, y, p_converted])
    return events_np

def events_to_voxel(events_np, num_bins=32, sensor_size=(480, 640), fixed_duration_us=100000):
    """Convert events to voxel grid representation using FIXED time intervals
    
    CRITICAL: Uses fixed time duration for training/testing consistency!
    
    Args:
        events_np: NumPy array (N, 4) with [t, x, y, p]
        num_bins: Number of temporal bins (default 32 for better temporal resolution)
        sensor_size: Sensor resolution (H, W)
        fixed_duration_us: Fixed time duration in microseconds (default 100ms = 100,000μs)
                          This ensures consistent temporal resolution across datasets
        
    Returns:
        voxel: PyTorch tensor (num_bins, H, W)
    """
    if len(events_np) == 0:
        return torch.zeros((num_bins, sensor_size[0], sensor_size[1]))
    
    # Initialize voxel grid
    voxel = torch.zeros((num_bins, sensor_size[0], sensor_size[1]), dtype=torch.float32)
    
    # Extract event components
    ts = events_np[:, 0]  # timestamps
    xs = events_np[:, 1].astype(int)  # x coordinates
    ys = events_np[:, 2].astype(int)  # y coordinates  
    ps = events_np[:, 3]  # polarities
    
    # Use FIXED time interval for consistency across datasets
    # This is critical for training/testing generalization
    t_min = ts.min()
    dt = fixed_duration_us / num_bins  # Fixed bin duration
    
    # Remove verbose logging - only print in debug mode
    # print(f"Using FIXED temporal bins: {num_bins} bins × {dt/1000:.2f}ms = {fixed_duration_us/1000:.1f}ms total")
    # print(f"Data time range: {t_min:.0f} - {ts.max():.0f}μs ({(ts.max()-t_min)/1000:.1f}ms)")
    
    # Assign events to temporal bins using fixed intervals
    bin_indices = np.clip(((ts - t_min) / dt).astype(int), 0, num_bins - 1)
    
    # Safe vectorized accumulation - avoiding memory leaks from numpy/torch interaction
    # Filter events within sensor bounds
    valid_mask = (xs >= 0) & (xs < sensor_size[1]) & (ys >= 0) & (ys < sensor_size[0])
    
    if valid_mask.any():
        valid_bins = bin_indices[valid_mask]
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask] 
        valid_ps = ps[valid_mask]
        

        # Convert to torch tensors for pure PyTorch operations
        bins_tensor = torch.from_numpy(valid_bins).long()
        xs_tensor = torch.from_numpy(valid_xs).long() 
        ys_tensor = torch.from_numpy(valid_ys).long()
        ps_tensor = torch.from_numpy(valid_ps).float()
        
        # Use PyTorch's index_add_ for safe in-place accumulation
        # Create linear indices: bin * H * W + y * W + x
        linear_indices = bins_tensor * (sensor_size[0] * sensor_size[1]) + \
                        ys_tensor * sensor_size[1] + xs_tensor
        
        # Reshape voxel to 1D for index_add_, then back to 3D
        voxel_1d = voxel.view(-1)
        voxel_1d.index_add_(0, linear_indices, ps_tensor)
        # voxel retains its original shape due to view operation
        
        # Clean up temporary variables
        del valid_bins, valid_xs, valid_ys, valid_ps, valid_mask
        del bins_tensor, xs_tensor, ys_tensor, ps_tensor, linear_indices
    
    return voxel

def save_voxel(voxel, output_path, format='pt'):
    """Save voxel grid to file
    
    Args:
        voxel: PyTorch tensor
        output_path: Output file path
        format: File format ('pt' or 'npy')
    """
    if format == 'pt':
        torch.save(voxel, output_path)
    elif format == 'npy':
        np.save(output_path, voxel.numpy())
    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    parser = argparse.ArgumentParser(description='Convert events to voxel grid')
    parser.add_argument('--input_file', required=True, help='Input H5 file path')
    parser.add_argument('--output_voxel_file', required=True, help='Output voxel file path')
    parser.add_argument('--config', help='Config YAML file path')
    parser.add_argument('--num_bins', type=int, help='Number of temporal bins (overrides config)')
    parser.add_argument('--sensor_size', nargs=2, type=int, help='Sensor size H W (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    num_bins = args.num_bins if args.num_bins else config['num_bins']
    sensor_size = tuple(args.sensor_size) if args.sensor_size else (config['sensor_size']['height'], config['sensor_size']['width'])
    
    print(f"Loading events from: {args.input_file}")
    events_np = load_h5_events(args.input_file)
    print(f"Loaded {len(events_np)} events")
    print(f"Time range: {events_np[:, 0].min():.0f} - {events_np[:, 0].max():.0f} μs")
    
    print(f"Converting to voxel grid: {num_bins} bins, {sensor_size} resolution")
    voxel = events_to_voxel(events_np, num_bins=num_bins, sensor_size=sensor_size)
    print(f"Voxel shape: {voxel.shape}")
    print(f"Voxel range: {voxel.min():.3f} - {voxel.max():.3f}")
    
    # Save voxel grid
    output_format = config['file_formats']['voxel_output']
    save_voxel(voxel, args.output_voxel_file, format=output_format)
    print(f"Voxel grid saved to: {args.output_voxel_file}")
    
    # Debug mode
    if args.debug:
        debug_dir = config['debug']['output_dir']
        print(f"Running debug mode, saving visualizations to: {debug_dir}")
        
        try:
            from .professional_visualizer import visualize_events_and_voxel
        except ImportError:
            # Fallback for direct script execution
            from professional_visualizer import visualize_events_and_voxel
        
        # Create comprehensive visualizations - both events and voxel with FIXED 100ms/32bins
        visualize_events_and_voxel(events_np, voxel, sensor_size, debug_dir, "encoder")

if __name__ == '__main__':
    main()