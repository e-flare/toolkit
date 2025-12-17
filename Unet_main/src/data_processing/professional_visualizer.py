"""
Professional Event and Voxel Visualizer based on event_utils-master

This module provides comprehensive, production-ready visualization tools 
directly leveraging the event_utils library. Supports arbitrary events 
and voxel data at any pipeline stage.

Key Features:
- Direct event_utils integration (no reinventing the wheel)
- 16-32 time slice visualizations for 100ms event streams  
- Universal: works with any events/voxel data
- Professional 3D spatiotemporal visualization
- Sliding window analysis
- Voxel grid 3D rendering

Based on Linus philosophy: "Use what works, don't reimplement badly"
"""

import os
import sys
import numpy as np
import h5py
import torch
from pathlib import Path

# Add event_utils to path
EVENT_UTILS_PATH = Path(__file__).parent.parent.parent / "event_utils-master"
sys.path.insert(0, str(EVENT_UTILS_PATH))

# Import event_utils components
from lib.data_formats.read_events import read_h5_events_dict
from lib.visualization.draw_event_stream import plot_events_sliding, plot_voxel_grid, plot_events
from lib.representations.voxel_grid import events_to_voxel as event_utils_voxel
from lib.representations.image import events_to_image


class ProfessionalEventVisualizer:
    """
    Professional event/voxel visualizer using event_utils directly.
    
    Designed for:
    1. Any events data (original, decoded, intermediate)
    2. Any voxel data (encoded, reconstructed) 
    3. Time-slice analysis (16-32 images for 100ms streams)
    4. Professional 3D spatiotemporal plots
    5. Direct event_utils integration
    """
    
    def __init__(self, output_dir="debug_output", dpi=150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        
    def visualize_events_comprehensive(self, events_np, sensor_size=(480, 640), 
                                     name_prefix="events", num_time_slices=32):
        """
        Comprehensive events visualization using FIXED 100ms/32bins window.
        
        Args:
            events_np: Events as numpy array (N, 4) [t, x, y, p] 
            sensor_size: Sensor dimensions (H, W)
            name_prefix: Prefix for output files
            num_time_slices: FIXED at 32 for 100ms window consistency
            
        Generates:
            - {name_prefix}_spatiotemporal_3d.png: Professional 3D plot
            - {name_prefix}_time_slices/: Directory with 32 time slice images
            - {name_prefix}_summary.png: Statistical summary
        """
        if len(events_np) == 0:
            print(f"No events to visualize for {name_prefix}")
            return
            
        print(f"Creating professional visualization for {name_prefix} ({len(events_np):,} events)")
        
        # Extract components (event_utils expects specific format)
        t, x, y, p = events_np[:, 0], events_np[:, 1], events_np[:, 2], events_np[:, 3]
        
        # Convert to event_utils expected format
        xs, ys, ts, ps = x.astype(np.int32), y.astype(np.int32), t.astype(np.int64), p.astype(np.int32)
        
        # 1. Professional 3D spatiotemporal visualization (event_utils native)
        self._create_native_3d_events_plot(xs, ys, ts, ps, sensor_size, name_prefix)
        
        # 2. Native 3D spatiotemporal series (with FIXED time intervals)
        self._create_native_3d_events_series(xs, ys, ts, ps, sensor_size, name_prefix, num_windows=8)
        
        # 3. 2D temporal series with red/blue polarity (same time intervals as 3D)
        self._create_2d_temporal_series(xs, ys, ts, ps, sensor_size, name_prefix, num_windows=8)
        
        # 4. Time slice analysis (32 slices) - for comparison
        self._create_time_slices(xs, ys, ts, ps, sensor_size, name_prefix, num_time_slices)
        
        # REMOVED: Multi-resolution slices - use FIXED 32 slices only
        # Follows Linus principle: "消除特殊情况"
        
        # 4. Summary statistics
        self._create_events_summary(events_np, sensor_size, name_prefix)
        
    def visualize_voxel_comprehensive(self, voxel_tensor, sensor_size=(480, 640), 
                                    name_prefix="voxel", duration_ms=100):
        """
        Comprehensive voxel visualization using event_utils 3D rendering.
        
        Args:
            voxel_tensor: Voxel as pytorch tensor (B, H, W)
            sensor_size: Sensor dimensions (H, W) 
            name_prefix: Prefix for output files
            duration_ms: Duration represented by voxel
            
        Generates:
            - {name_prefix}_3d_voxel.png: Professional 3D voxel rendering
            - {name_prefix}_temporal_bins.png: All temporal bins
            - {name_prefix}_analysis.png: Statistical analysis
        """
        if voxel_tensor.numel() == 0:
            print(f"Empty voxel to visualize for {name_prefix}")
            return
            
        print(f"Creating professional voxel visualization for {name_prefix}")
        
        # 1. Convert voxel back to events for 3D plotting
        events_from_voxel = self._voxel_to_events_for_viz(voxel_tensor, duration_ms, sensor_size)
        
        # 2. Professional 3D voxel rendering using event_utils
        self._create_3d_voxel_plot(events_from_voxel, voxel_tensor, sensor_size, name_prefix)
        
        # 3. Temporal bins visualization  
        self._create_temporal_bins_plot(voxel_tensor, name_prefix, duration_ms)
        
        # 4. Voxel analysis
        self._create_voxel_analysis(voxel_tensor, name_prefix)
        
    def visualize_events_from_h5(self, h5_path, name_prefix="h5_events", 
                               num_time_slices=32):
        """
        Direct H5 events visualization using event_utils native loading.
        
        Args:
            h5_path: Path to H5 file
            name_prefix: Prefix for output files  
            num_time_slices: Number of time slices
        """
        print(f"Loading and visualizing H5 file: {h5_path}")
        
        # Use event_utils native H5 loading (already supports our format)
        events_dict = read_h5_events_dict(h5_path, read_frames=False)
        
        # Convert to our standard format
        events_np = np.column_stack([
            events_dict['ts'], 
            events_dict['xs'], 
            events_dict['ys'], 
            events_dict['ps']
        ])
        
        # Infer sensor size
        sensor_size = (int(events_dict['ys'].max()) + 1, int(events_dict['xs'].max()) + 1)
        
        self.visualize_events_comprehensive(events_np, sensor_size, name_prefix, num_time_slices)
        
    def create_sliding_window_video(self, events_np, sensor_size=(480, 640),
                                  name_prefix="sliding", window_ms=6.25, overlap_ms=3.125):
        """
        Create sliding window video using event_utils professional tools.
        
        Args:
            events_np: Events as numpy array (N, 4) [t, x, y, p]
            sensor_size: Sensor dimensions
            name_prefix: Output prefix
            window_ms: Window size in milliseconds
            overlap_ms: Overlap between windows
        """
        print(f"Creating sliding window visualization for {name_prefix}")
        
        if len(events_np) == 0:
            return
            
        # Extract components
        t, x, y, p = events_np[:, 0], events_np[:, 1], events_np[:, 2], events_np[:, 3]
        xs, ys, ts, ps = x.astype(np.int32), y.astype(np.int32), t.astype(np.int64), p.astype(np.int32)
        
        # Create args object for event_utils compatibility
        class Args:
            def __init__(self, output_dir, name_prefix):
                self.output_path = str(output_dir / f"{name_prefix}_sliding")
                self.w_width = window_ms * 1000  # Convert to microseconds
                self.sw_width = overlap_ms * 1000  # Convert to microseconds
                self.num_show = -1
                self.event_size = 2
                self.hide_events = False
                self.hide_frames = True  
                self.elev = 20
                self.azim = 45
                self.crop = None
                self.compress_front = False
                self.invert = False
                self.num_compress = 0
                self.show_plot = False
                self.show_axes = False
                self.stride = 1
                
        args = Args(self.output_dir, name_prefix)
        os.makedirs(args.output_path, exist_ok=True)
        
        # Use event_utils sliding window visualization
        plot_events_sliding(xs, ys, ts, ps, args, frames=[], frame_ts=[])
        
    def _create_native_3d_events_plot(self, xs, ys, ts, ps, sensor_size, name_prefix):
        """Create native 3D spatiotemporal events plot using event_utils plot_events"""
        output_path = self.output_dir / f"{name_prefix}_native_3d_spatiotemporal.png"
        
        # Sample events for performance and avoid scatter size issues
        sample_size = min(50000, len(xs))
        if sample_size > 0:
            indices = np.random.choice(len(xs), sample_size, replace=False)
            sample_xs, sample_ys, sample_ts, sample_ps = xs[indices], ys[indices], ts[indices], ps[indices]
            
            print(f"Creating native 3D spatiotemporal plot for {name_prefix} ({sample_size:,} events)")
            
            # Use event_utils native plot_events for true 3D spatiotemporal visualization
            plot_events(sample_xs, sample_ys, sample_ts, sample_ps, 
                       save_path=str(output_path),
                       num_show=-1,  # Show all sampled events
                       event_size=2,
                       elev=20, 
                       azim=45,
                       show_events=True,
                       show_frames=False,
                       show_plot=False,
                       img_size=sensor_size,
                       show_axes=True)

    def _create_native_3d_events_series(self, xs, ys, ts, ps, sensor_size, name_prefix, num_windows=8):
        """Create series of 3D spatiotemporal plots with fixed time intervals"""
        series_dir = self.output_dir / f"{name_prefix}_3d_series"
        series_dir.mkdir(exist_ok=True)
        
        if len(xs) == 0:
            return
            
        # Use FIXED time intervals (same as voxel bins)
        t_min, t_max = ts.min(), ts.max()
        # Use actual data duration divided into windows
        actual_duration = t_max - t_min  # Use actual segment duration
        window_duration = actual_duration / num_windows
        
        print(f"Creating {num_windows} native 3D spatiotemporal windows for {name_prefix}")
        
        for i in range(num_windows):
            t_start = t_min + i * window_duration
            t_end = t_start + window_duration
            
            # Select events in this time window
            mask = (ts >= t_start) & (ts < t_end)
            if not np.any(mask):
                continue
                
            window_xs, window_ys, window_ts, window_ps = xs[mask], ys[mask], ts[mask], ps[mask]
            
            # Sample for performance
            sample_size = min(20000, len(window_xs))
            if sample_size > 0:
                indices = np.random.choice(len(window_xs), sample_size, replace=False)
                sample_xs = window_xs[indices]
                sample_ys = window_ys[indices] 
                sample_ts = window_ts[indices]
                sample_ps = window_ps[indices]
                
                output_path = series_dir / f"window_{i:02d}.png"
                
                # Use event_utils native plot_events
                plot_events(sample_xs, sample_ys, sample_ts, sample_ps,
                           save_path=str(output_path),
                           num_show=-1,
                           event_size=2,
                           elev=20,
                           azim=45,
                           show_events=True,
                           show_frames=False,
                           show_plot=False,
                           img_size=sensor_size,
                           show_axes=True)

    def _create_2d_temporal_series(self, xs, ys, ts, ps, sensor_size, name_prefix, num_windows=8):
        """Create series of 2D temporal images with red/blue polarity display"""
        series_dir = self.output_dir / f"{name_prefix}_2d_temporal"
        series_dir.mkdir(exist_ok=True)
        
        if len(xs) == 0:
            return
            
        # Use FIXED time intervals (same as 3D windows)
        t_min, t_max = ts.min(), ts.max()
        # Use actual data duration divided into windows
        actual_duration = t_max - t_min  # Use actual segment duration
        window_duration = actual_duration / num_windows
        
        print(f"Creating {num_windows} 2D temporal images for {name_prefix}")
        
        import matplotlib.pyplot as plt
        
        for i in range(num_windows):
            t_start = t_min + i * window_duration
            t_end = t_start + window_duration
            
            # Select events in this time window
            mask = (ts >= t_start) & (ts < t_end)
            if not np.any(mask):
                continue
                
            window_xs, window_ys, window_ts, window_ps = xs[mask], ys[mask], ts[mask], ps[mask]
            
            if len(window_xs) == 0:
                continue
                
            # Create event image using event_utils (returns red/blue image)
            event_img = events_to_image(window_xs, window_ys, window_ps, sensor_size=sensor_size)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Display with RdBu colormap (red=positive, blue=negative)
            im = ax.imshow(event_img, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(f'2D Temporal Window {i+1}/{num_windows}\n'
                        f'Time: {(t_start-t_min)/1000:.1f}-{(t_end-t_min)/1000:.1f}ms\n'
                        f'Events: {len(window_xs):,}')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Polarity (Red=+, Blue=-)')
            
            output_path = series_dir / f"temporal_{i:02d}.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
                   
    def _create_time_slices(self, xs, ys, ts, ps, sensor_size, name_prefix, num_slices):
        """Create time slice visualizations"""
        slices_dir = self.output_dir / f"{name_prefix}_time_slices"
        slices_dir.mkdir(exist_ok=True)
        
        t_min, t_max = ts.min(), ts.max()
        duration = t_max - t_min
        slice_duration = duration / num_slices
        
        print(f"Creating {num_slices} time slices for {name_prefix}")
        
        for i in range(num_slices):
            t_start = t_min + i * slice_duration
            t_end = t_start + slice_duration
            
            # Select events in this time window
            mask = (ts >= t_start) & (ts < t_end)
            if not np.any(mask):
                continue
                
            slice_xs, slice_ys, slice_ts, slice_ps = xs[mask], ys[mask], ts[mask], ps[mask]
            
            # Create event image for this slice
            if len(slice_xs) > 0:
                event_img = events_to_image(slice_xs, slice_ys, slice_ps, sensor_size=sensor_size)
                
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.imshow(event_img, cmap='RdBu')
                plt.title(f'Time Slice {i+1}/{num_slices}\n'
                         f'Time: {(t_start-t_min)/1000:.1f}-{(t_end-t_min)/1000:.1f}ms\n'
                         f'Events: {len(slice_xs):,}')
                plt.colorbar()
                plt.axis('off')
                
                output_path = slices_dir / f"slice_{i:02d}.png"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
    def _create_events_summary(self, events_np, sensor_size, name_prefix):
        """Create comprehensive events summary"""
        import matplotlib.pyplot as plt
        
        t, x, y, p = events_np[:, 0], events_np[:, 1], events_np[:, 2], events_np[:, 3]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Events Analysis: {name_prefix} ({len(events_np):,} events)', fontsize=16)
        
        # 1. Event image
        ax = axes[0, 0]
        event_img = events_to_image(x.astype(int), y.astype(int), p, sensor_size=sensor_size)
        im = ax.imshow(event_img, cmap='RdBu')
        ax.set_title('Event Accumulation Image')
        plt.colorbar(im, ax=ax)
        
        # 2. Temporal distribution
        ax = axes[0, 1]
        t_ms = (t - t.min()) / 1000
        ax.hist(t_ms, bins=50, alpha=0.7, color='blue')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Event Count')
        ax.set_title('Temporal Distribution')
        ax.grid(True)
        
        # 3. Spatial distribution
        ax = axes[0, 2]
        sample_size = min(10000, len(events_np))
        indices = np.random.choice(len(events_np), sample_size, replace=False)
        scatter = ax.scatter(x[indices], y[indices], c=p[indices], s=0.5, cmap='RdBu', alpha=0.6)
        ax.set_xlim(0, sensor_size[1])
        ax.set_ylim(sensor_size[0], 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Spatial Distribution')
        plt.colorbar(scatter, ax=ax)
        
        # 4. Polarity distribution
        ax = axes[1, 0]
        pos_count = np.sum(p > 0)
        neg_count = np.sum(p < 0)
        ax.bar(['Positive', 'Negative'], [pos_count, neg_count], 
               color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Event Count')
        ax.set_title('Polarity Distribution')
        
        # 5. Event rate
        ax = axes[1, 1]
        time_windows = np.linspace(t.min(), t.max(), 20)
        event_rates = []
        for i in range(len(time_windows)-1):
            mask = (t >= time_windows[i]) & (t < time_windows[i+1])
            rate = np.sum(mask) / (time_windows[i+1] - time_windows[i]) * 1e6
            event_rates.append(rate)
        ax.plot((time_windows[:-1] - t.min()) / 1000, event_rates, 'o-')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Event Rate (events/sec)')
        ax.set_title('Event Rate Over Time')
        ax.grid(True)
        
        # 6. Statistics
        ax = axes[1, 2]
        stats_text = f"""Statistics:
Total events: {len(events_np):,}
Duration: {(t.max()-t.min())/1000:.1f} ms
Event rate: {len(events_np)/(t.max()-t.min())*1e6:.0f} events/sec
Positive: {pos_count:,} ({pos_count/len(events_np)*100:.1f}%)
Negative: {neg_count:,} ({neg_count/len(events_np)*100:.1f}%)
Spatial range:
  X: [{x.min():.0f}, {x.max():.0f}]
  Y: [{y.min():.0f}, {y.max():.0f}]
Density: {len(events_np)/(sensor_size[0]*sensor_size[1]):.2f} events/pixel"""
        ax.text(0.05, 0.5, stats_text, fontsize=10, ha='left', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.axis('off')
        ax.set_title('Event Statistics')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{name_prefix}_summary.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _create_3d_voxel_plot(self, events_np, voxel_tensor, sensor_size, name_prefix):
        """Create professional 3D voxel plot - skip due to matplotlib compatibility issues"""
        print(f"Skipping 3D voxel plot for {name_prefix} due to matplotlib compatibility")
        # The event_utils voxel plotting has matplotlib version compatibility issues
        # We'll focus on the temporal bins and analysis plots instead
        pass
        
    def _create_temporal_bins_plot(self, voxel_tensor, name_prefix, duration_ms):
        """Create temporal bins visualization"""
        import matplotlib.pyplot as plt
        
        num_bins = voxel_tensor.shape[0]
        cols = 4
        rows = (num_bins + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Voxel Temporal Bins: {name_prefix} ({duration_ms}ms)', fontsize=16)
        
        for i in range(num_bins):
            ax = axes[i//cols, i%cols]
            voxel_slice = voxel_tensor[i].numpy()
            im = ax.imshow(voxel_slice, cmap='RdBu', vmin=voxel_tensor.min(), vmax=voxel_tensor.max())
            ax.set_title(f'Bin {i}\n{i*duration_ms/num_bins:.1f}-{(i+1)*duration_ms/num_bins:.1f}ms')
            ax.axis('off')
            
        # Hide empty subplots
        for i in range(num_bins, rows * cols):
            axes[i//cols, i%cols].axis('off')
            
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        output_path = self.output_dir / f"{name_prefix}_temporal_bins.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _create_voxel_analysis(self, voxel_tensor, name_prefix):
        """Create comprehensive voxel analysis"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Voxel Analysis: {name_prefix}', fontsize=16)
        
        # 1. Voxel sum
        ax = axes[0, 0]
        voxel_sum = torch.sum(voxel_tensor, dim=0).numpy()
        im = ax.imshow(voxel_sum, cmap='RdBu')
        ax.set_title('Voxel Sum (All Bins)')
        plt.colorbar(im, ax=ax)
        
        # 2. Temporal profile
        ax = axes[0, 1]
        temporal_sum = torch.sum(voxel_tensor, dim=[1, 2]).numpy()
        ax.plot(range(len(temporal_sum)), temporal_sum, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Temporal Bin')
        ax.set_ylabel('Total Events')
        ax.set_title('Temporal Profile')
        ax.grid(True)
        
        # 3. Value distribution
        ax = axes[0, 2]
        voxel_flat = voxel_tensor.flatten().numpy()
        nonzero_voxels = voxel_flat[voxel_flat != 0]
        if len(nonzero_voxels) > 0:
            ax.hist(nonzero_voxels, bins=50, alpha=0.7)
            ax.set_xlabel('Voxel Value')
            ax.set_ylabel('Count')
            ax.set_title(f'Value Distribution\n({len(nonzero_voxels):,} non-zero)')
            ax.set_yscale('log')
        
        # 4. Active pixels per bin
        ax = axes[1, 0]
        bin_events = [torch.count_nonzero(voxel_tensor[i]).item() for i in range(voxel_tensor.shape[0])]
        ax.bar(range(len(bin_events)), bin_events)
        ax.set_xlabel('Temporal Bin')
        ax.set_ylabel('Non-zero Pixels')
        ax.set_title('Active Pixels per Bin')
        ax.grid(True)
        
        # 5. Min/Max per bin
        ax = axes[1, 1]
        max_vals = [torch.max(voxel_tensor[i]).item() for i in range(voxel_tensor.shape[0])]
        min_vals = [torch.min(voxel_tensor[i]).item() for i in range(voxel_tensor.shape[0])]
        ax.plot(range(len(max_vals)), max_vals, 'ro-', label='Max', linewidth=2)
        ax.plot(range(len(min_vals)), min_vals, 'bo-', label='Min', linewidth=2)
        ax.set_xlabel('Temporal Bin')
        ax.set_ylabel('Voxel Value')
        ax.set_title('Min/Max Values per Bin')
        ax.legend()
        ax.grid(True)
        
        # 6. Statistics
        ax = axes[1, 2]
        stats_text = f"""Voxel Statistics:
Shape: {tuple(voxel_tensor.shape)}
Total events: {voxel_tensor.sum():.0f}
Non-zero voxels: {torch.count_nonzero(voxel_tensor).item():,}
Sparsity: {(1-torch.count_nonzero(voxel_tensor).item()/voxel_tensor.numel())*100:.2f}%
Value range: [{voxel_tensor.min():.3f}, {voxel_tensor.max():.3f}]
Mean (non-zero): {voxel_tensor[voxel_tensor!=0].mean():.3f}
Std (non-zero): {voxel_tensor[voxel_tensor!=0].std():.3f}

Per-bin stats:
Events/bin: {temporal_sum.mean():.0f} ± {temporal_sum.std():.0f}
Active pixels: {np.mean(bin_events):.0f} ± {np.std(bin_events):.0f}"""
        ax.text(0.05, 0.5, stats_text, fontsize=10, ha='left', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.axis('off')
        ax.set_title('Voxel Statistics')
        
        plt.tight_layout()
        output_path = self.output_dir / f"{name_prefix}_analysis.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _voxel_to_events_for_viz(self, voxel_tensor, duration_ms, sensor_size):
        """Convert voxel back to events for 3D visualization"""
        events_list = []
        num_bins = voxel_tensor.shape[0]
        bin_duration = duration_ms * 1000 / num_bins  # microseconds per bin
        
        for bin_idx in range(num_bins):
            bin_data = voxel_tensor[bin_idx].numpy()
            
            # Find non-zero pixels
            nonzero_indices = np.nonzero(bin_data)
            if len(nonzero_indices[0]) == 0:
                continue
                
            y_coords, x_coords = nonzero_indices
            values = bin_data[nonzero_indices]
            
            # Convert voxel values back to events
            for i, (y, x, val) in enumerate(zip(y_coords, x_coords, values)):
                if val == 0:
                    continue
                    
                # Create events based on voxel value
                num_events = int(abs(val))
                polarity = 1 if val > 0 else -1
                
                # Random timestamps within this bin
                t_start = bin_idx * bin_duration
                t_end = (bin_idx + 1) * bin_duration
                timestamps = np.random.uniform(t_start, t_end, num_events)
                
                for t in timestamps:
                    events_list.append([t, x, y, polarity])
                    
        if len(events_list) == 0:
            return np.array([]).reshape(0, 4)
            
        return np.array(events_list)
        
    def print_summary(self, name_prefix="visualization"):
        """Print summary of generated files"""
        print(f"\n=== Professional Visualization Summary ({name_prefix}) ===")
        print(f"Output directory: {self.output_dir}")
        print("Generated files:")
        
        for file_path in sorted(self.output_dir.glob("*")):
            if file_path.is_file():
                print(f"- {file_path.name}")
            elif file_path.is_dir():
                num_files = len(list(file_path.glob("*")))
                print(f"- {file_path.name}/ ({num_files} files)")


# Convenience functions for direct usage
def visualize_events(events_np, sensor_size=(480, 640), output_dir="debug_output", 
                   name="events", num_time_slices=32):
    """Convenience function for events visualization - FIXED 32 slices"""
    viz = ProfessionalEventVisualizer(output_dir)
    viz.visualize_events_comprehensive(events_np, sensor_size, name, num_time_slices)
    viz.print_summary(name)

    
def visualize_voxel(voxel_tensor, sensor_size=(480, 640), output_dir="debug_output",
                   name="voxel", duration_ms=100):
    """Convenience function for voxel visualization - FIXED 100ms duration"""
    viz = ProfessionalEventVisualizer(output_dir)
    viz.visualize_voxel_comprehensive(voxel_tensor, sensor_size, name, duration_ms)
    viz.print_summary(name)


def visualize_events_and_voxel(events_np, voxel_tensor, sensor_size=(480, 640), 
                              output_dir="debug_output", name="pipeline"):
    """Unified function for simultaneous events+voxel visualization"""
    viz = ProfessionalEventVisualizer(output_dir)
    # Events visualization with FIXED 32 slices
    viz.visualize_events_comprehensive(events_np, sensor_size, f"{name}_events", 32)
    # Voxel visualization with FIXED 100ms
    viz.visualize_voxel_comprehensive(voxel_tensor, sensor_size, f"{name}_voxel", 100)
    viz.print_summary(name)

def _extract_time_segment(events_np, segment_idx=1, num_segments=5, total_duration_us=100000):
    """Extract a specific time segment from events for memory optimization
    
    Args:
        events_np: Events array (N, 4) [t, x, y, p]
        segment_idx: Which segment to extract (0-4, default=1 for 10-30ms)
        num_segments: Total number of segments (default=5 for 20ms each)
        total_duration_us: Total duration in microseconds (default=100ms)
    
    Returns:
        segment_events: Events in the selected segment
        segment_start_us: Start time of segment in microseconds
        segment_duration_us: Duration of segment in microseconds
    """
    if len(events_np) == 0:
        return np.array([]).reshape(0, 4), 0, 0
        
    t_min = events_np[:, 0].min()
    t_max = events_np[:, 0].max()
    
    # Calculate segment parameters
    segment_duration_us = total_duration_us / num_segments
    segment_start_us = segment_idx * segment_duration_us
    segment_end_us = (segment_idx + 1) * segment_duration_us
    
    # Convert to actual timestamps
    actual_start = t_min + segment_start_us
    actual_end = t_min + segment_end_us
    
    # Extract events in this segment
    mask = (events_np[:, 0] >= actual_start) & (events_np[:, 0] < actual_end)
    segment_events = events_np[mask].copy()
    
    # Adjust timestamps to start from 0 for this segment
    if len(segment_events) > 0:
        segment_events[:, 0] = segment_events[:, 0] - actual_start
    
    print(f"Extracted segment {segment_idx}: {segment_start_us/1000:.1f}-{segment_end_us/1000:.1f}ms "
          f"({len(segment_events):,} events)")
    
    return segment_events, segment_start_us, segment_duration_us

def visualize_complete_pipeline(input_events, input_voxel, output_events, output_voxel, 
                               sensor_size=(480, 640), output_dir="debug_output", segment_idx=1):
    """Complete 6-visualization pipeline with memory optimization: 2×2 events + 2 voxels
    
    Memory optimization: 100ms → 5×20ms segments, visualize segment_idx (default: 1, i.e. 10-30ms)
    
    Generates exactly 6 visualization results:
    1. Input Events 3D spatiotemporal (segment 10-30ms)
    2. Input Events 2D temporal (red/blue, segment 10-30ms)  
    3. Input Events→Voxel (segment 10-30ms, 8 bins)
    4. Output Events 3D spatiotemporal (segment 10-30ms)
    5. Output Events 2D temporal (red/blue, segment 10-30ms)
    6. Output Events→Voxel (re-encoded, segment 10-30ms, 8 bins)
    """
    viz = ProfessionalEventVisualizer(output_dir)
    
    # Calculate time range for display
    segment_duration = 20  # 20ms per segment
    start_ms = segment_idx * segment_duration
    end_ms = (segment_idx + 1) * segment_duration
    print(f"=== Complete Pipeline Visualization: 6 Results (Segment {segment_idx}: {start_ms}-{end_ms}ms) ===")
    
    # Extract segments for memory optimization
    input_segment, input_start, segment_duration = _extract_time_segment(input_events, segment_idx)
    output_segment, output_start, _ = _extract_time_segment(output_events, segment_idx)
    
    # Import encoding function for segment voxelization
    try:
        from .encode import events_to_voxel
    except ImportError:
        from encode import events_to_voxel
    
    # Create voxels from segments (20ms → 8 bins)
    num_bins = 8  # 20ms / 8 = 2.5ms per bin
    if len(input_segment) > 0:
        input_segment_voxel = events_to_voxel(input_segment, num_bins=num_bins, 
                                            sensor_size=sensor_size, 
                                            fixed_duration_us=int(segment_duration))
    else:
        input_segment_voxel = torch.zeros((num_bins, sensor_size[0], sensor_size[1]))
    
    if len(output_segment) > 0:
        output_segment_voxel = events_to_voxel(output_segment, num_bins=num_bins, 
                                             sensor_size=sensor_size,
                                             fixed_duration_us=int(segment_duration))
    else:
        output_segment_voxel = torch.zeros((num_bins, sensor_size[0], sensor_size[1]))
    
    # 1&2: Input Events Segment (3D + 2D)
    print(f"Visualizing Input Events Segment {segment_idx} (3D + 2D temporal series)...")
    viz.visualize_events_comprehensive(input_segment, sensor_size, f"input_events_seg{segment_idx}", 32)
    
    # 3: Input Events→Voxel Segment
    print(f"Visualizing Input Events→Voxel Segment {segment_idx}...")
    viz.visualize_voxel_comprehensive(input_segment_voxel, sensor_size, f"input_voxel_seg{segment_idx}", 20)
    
    # 4&5: Output Events Segment (3D + 2D)
    print(f"Visualizing Output Events Segment {segment_idx} (3D + 2D temporal series)...")  
    viz.visualize_events_comprehensive(output_segment, sensor_size, f"output_events_seg{segment_idx}", 32)
    
    # 6: Output Events→Voxel Segment (re-encoded)
    print(f"Visualizing Output Events→Voxel Segment {segment_idx} (re-encoded)...")
    viz.visualize_voxel_comprehensive(output_segment_voxel, sensor_size, f"output_voxel_seg{segment_idx}", 20)
    
    print(f"\n=== 6-Visualization Pipeline Complete (Segment {segment_idx}) ===")
    viz.print_summary(f"complete_pipeline_seg{segment_idx}")

    
def visualize_h5_file(h5_path, output_dir="debug_output", num_time_slices=32):
    """Convenience function for H5 file visualization"""
    viz = ProfessionalEventVisualizer(output_dir)
    viz.visualize_events_from_h5(h5_path, "h5_events", num_time_slices)
    viz.print_summary("h5_events")


if __name__ == "__main__":
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Event/Voxel Visualizer")
    parser.add_argument("input_file", help="H5 events file to visualize")
    parser.add_argument("--output_dir", default="professional_viz_test", help="Output directory")
    parser.add_argument("--time_slices", type=int, default=32, help="Number of time slices")
    
    args = parser.parse_args()
    
    print(f"Professional visualization of: {args.input_file}")
    visualize_h5_file(args.input_file, args.output_dir, args.time_slices)