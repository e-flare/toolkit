#!/usr/bin/env python3
"""

debug：
1. DSEC
2. DVS
3. 
4. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Optional, List
import cv2


class EventVisualizer:

    def __init__(self, output_dir: str, resolution: Tuple[int, int] = (640, 480),
                 background_color: str = "black"):

        Args:
            output_dir: 
            resolution:  (width, height)
            background_color:  ("white"  "black")
        """
        self.output_dir = output_dir
        self.resolution = resolution  # (width, height)
        self.width, self.height = resolution
        self.background_color = background_color.lower()

        os.makedirs(output_dir, exist_ok=True)

        self.colors = {
            'background_pos': (1, 0, 0, 0.6),
            'background_neg': (0, 0, 1, 0.6),
            'flare_pos': (1, 1, 0, 0.8),
            'flare_neg': (1, 0.5, 0, 0.8),
        }

        if self.background_color == "white":
            self.event_colors = {
                'ON': 'darkred',
                'OFF': 'darkblue',
                'edge_color': 'black',
                'text_color': 'black',
                'grid_color': 'gray',
                'facecolor': 'white'
            }
        else:
            self.event_colors = {
                'ON': 'red',
                'OFF': 'cyan',
                'edge_color': 'white',
                'text_color': 'white',
                'grid_color': 'gray',
                'facecolor': 'black'
            }
    
    def analyze_and_visualize_pipeline(self, 
                                     background_events: np.ndarray,
                                     flare_events: np.ndarray, 
                                     merged_events: np.ndarray,
                                     labels: np.ndarray,
                                     sample_idx: int) -> Dict[str, float]:
        
        Args:
            background_events: DSEC [N1, 4]
            flare_events: DVS [N2, 4]  
            merged_events:  [N_total, 4]
            labels:  [N_total] (0=, 1=)
            sample_idx: 
            
        Returns:
            Dict
        """
        print(f"\n📊 Analyzing Event Pipeline for Sample {sample_idx}")
        print("=" * 60)
        
        density_stats = self._analyze_event_densities(
            background_events, flare_events, merged_events, labels
        )
        
        self._visualize_spatial_distribution(
            background_events, flare_events, merged_events, labels, sample_idx
        )
        
        self._visualize_temporal_distribution(
            background_events, flare_events, merged_events, labels, sample_idx
        )
        
        self._create_multi_resolution_event_visualizations(
            background_events, flare_events, merged_events, labels, sample_idx
        )
        
        self._generate_statistics_report(
            background_events, flare_events, merged_events, labels, 
            density_stats, sample_idx
        )
        
        return density_stats
    
    def _analyze_event_densities(self, background_events: np.ndarray,
                                flare_events: np.ndarray,
                                merged_events: np.ndarray,
                                labels: np.ndarray) -> Dict[str, float]:
        
        def calculate_density(events: np.ndarray) -> Tuple[float, float]:
            if len(events) == 0:
                return 0.0, 0.0
            
            time_span_us = events[:, 2].max() - events[:, 2].min()
            time_span_ms = time_span_us / 1000.0
            
            if time_span_ms <= 0:
                return 0.0, time_span_ms
                
            density = len(events) / time_span_ms
            return density, time_span_ms
        
        bg_density, bg_duration = calculate_density(background_events)
        flare_density, flare_duration = calculate_density(flare_events)
        merged_density, merged_duration = calculate_density(merged_events)
        
        if len(labels) > 0:
            bg_mask = labels == 0
            flare_mask = labels == 1
            
            merged_bg_events = merged_events[bg_mask] if bg_mask.any() else np.empty((0, 4))
            merged_flare_events = merged_events[flare_mask] if flare_mask.any() else np.empty((0, 4))
            
            merged_bg_density, _ = calculate_density(merged_bg_events)
            merged_flare_density, _ = calculate_density(merged_flare_events)
        else:
            merged_bg_density = 0.0
            merged_flare_density = 0.0
        
        stats = {
            'background_events': len(background_events),
            'flare_events': len(flare_events),
            'merged_events': len(merged_events),
            'background_density': bg_density,
            'flare_density': flare_density,
            'merged_density': merged_density,
            'merged_background_density': merged_bg_density,
            'merged_flare_density': merged_flare_density,
            'background_duration_ms': bg_duration,
            'flare_duration_ms': flare_duration,
            'merged_duration_ms': merged_duration,
        }
        
        return stats
    
    def _visualize_spatial_distribution(self, background_events: np.ndarray,
                                       flare_events: np.ndarray,
                                       merged_events: np.ndarray,
                                       labels: np.ndarray,
                                       sample_idx: int):

        bg_color = self.event_colors['facecolor']
        text_color = self.event_colors['text_color']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=bg_color)
        fig.suptitle(f'Event Spatial Distribution - Sample {sample_idx}', fontsize=16, color=text_color)

        for ax in axes.flat:
            ax.set_facecolor(bg_color)
        
        ax = axes[0, 0]
        if len(background_events) > 0:
            pos_mask = background_events[:, 3] == 1
            neg_mask = background_events[:, 3] == 0
            
            if pos_mask.any():
                ax.scatter(background_events[pos_mask, 0], background_events[pos_mask, 1], 
                          c='red', s=0.1, alpha=0.6, label=f'ON ({pos_mask.sum()})')
            if neg_mask.any():
                ax.scatter(background_events[neg_mask, 0], background_events[neg_mask, 1],
                          c='blue', s=0.1, alpha=0.6, label=f'OFF ({neg_mask.sum()})')
        
        ax.set_title(f'DSEC Background Events ({len(background_events)})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        if len(flare_events) > 0:
            pos_mask = flare_events[:, 3] == 1
            neg_mask = flare_events[:, 3] == 0
            
            if pos_mask.any():
                ax.scatter(flare_events[pos_mask, 0], flare_events[pos_mask, 1],
                          c='yellow', s=0.5, alpha=0.8, label=f'ON ({pos_mask.sum()})')
            if neg_mask.any():
                ax.scatter(flare_events[neg_mask, 0], flare_events[neg_mask, 1],
                          c='orange', s=0.5, alpha=0.8, label=f'OFF ({neg_mask.sum()})')
        
        ax.set_title(f'DVS Flare Events ({len(flare_events)})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        if len(merged_events) > 0 and len(labels) > 0:
            bg_mask = labels == 0
            flare_mask = labels == 1
            
            if bg_mask.any():
                bg_events = merged_events[bg_mask]
                pos_bg = bg_events[bg_events[:, 3] == 1]
                neg_bg = bg_events[bg_events[:, 3] == 0]
                
                if len(pos_bg) > 0:
                    ax.scatter(pos_bg[:, 0], pos_bg[:, 1], c='red', s=0.1, alpha=0.6, 
                              label=f'BG ON ({len(pos_bg)})')
                if len(neg_bg) > 0:
                    ax.scatter(neg_bg[:, 0], neg_bg[:, 1], c='blue', s=0.1, alpha=0.6,
                              label=f'BG OFF ({len(neg_bg)})')
            
            if flare_mask.any():
                flare_events_merged = merged_events[flare_mask]
                pos_flare = flare_events_merged[flare_events_merged[:, 3] == 1]
                neg_flare = flare_events_merged[flare_events_merged[:, 3] == 0]
                
                if len(pos_flare) > 0:
                    ax.scatter(pos_flare[:, 0], pos_flare[:, 1], c='yellow', s=0.5, alpha=0.8,
                              label=f'Flare ON ({len(pos_flare)})')
                if len(neg_flare) > 0:
                    ax.scatter(neg_flare[:, 0], neg_flare[:, 1], c='orange', s=0.5, alpha=0.8,
                              label=f'Flare OFF ({len(neg_flare)})')
        
        ax.set_title(f'Merged Events by Type ({len(merged_events)})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        if len(merged_events) > 0:
            hist, xedges, yedges = np.histogram2d(
                merged_events[:, 0], merged_events[:, 1],
                bins=[64, 48], range=[[0, self.width], [0, self.height]]
            )
            
            im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='hot',
                          extent=[0, self.width, 0, self.height])
            ax.set_title('Event Density Heatmap')
            plt.colorbar(im, ax=ax, label='Event Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'spatial_distribution_sample_{sample_idx}.png'),
                   dpi=150, bbox_inches='tight', facecolor=bg_color)
        plt.close()
    
    def _visualize_temporal_distribution(self, background_events: np.ndarray,
                                        flare_events: np.ndarray,
                                        merged_events: np.ndarray,
                                        labels: np.ndarray,
                                        sample_idx: int):

        bg_color = self.event_colors['facecolor']
        text_color = self.event_colors['text_color']

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor=bg_color)
        fig.suptitle(f'Event Temporal Distribution - Sample {sample_idx}', fontsize=16, color=text_color)

        for ax in axes.flat:
            ax.set_facecolor(bg_color)
        
        ax = axes[0]
        
        all_times = []
        if len(background_events) > 0:
            all_times.extend(background_events[:, 2])
        if len(flare_events) > 0:
            all_times.extend(flare_events[:, 2])
        if len(merged_events) > 0:
            all_times.extend(merged_events[:, 2])
            
        if all_times:
            t_min, t_max = min(all_times), max(all_times)
            time_bins = np.linspace(t_min, t_max, 100)
            
            if len(background_events) > 0:
                ax.hist(background_events[:, 2], bins=time_bins, alpha=0.6, 
                       label=f'Background ({len(background_events)})', color='blue')
            
            if len(flare_events) > 0:
                ax.hist(flare_events[:, 2], bins=time_bins, alpha=0.8,
                       label=f'Flare ({len(flare_events)})', color='orange')
            
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Event Count')
        ax.set_title('Event Count Distribution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        
        def plot_interval_histogram(events, label, color):
            if len(events) > 1:
                intervals = np.diff(np.sort(events[:, 2]))
                intervals_ms = intervals / 1000.0
                
                intervals_ms = intervals_ms[intervals_ms < np.percentile(intervals_ms, 95)]
                
                ax.hist(intervals_ms, bins=50, alpha=0.7, label=f'{label} (median: {np.median(intervals_ms):.2f}ms)',
                       color=color, density=True)
        
        plot_interval_histogram(background_events, 'Background', 'blue')
        plot_interval_histogram(flare_events, 'Flare', 'orange')
        
        ax.set_xlabel('Inter-event Interval (ms)')
        ax.set_ylabel('Density')  
        ax.set_title('Inter-event Interval Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        ax = axes[2]
        
        if len(merged_events) > 0 and len(labels) > 0:
            sorted_indices = np.argsort(merged_events[:, 2])
            sorted_times = merged_events[sorted_indices, 2]
            sorted_labels = labels[sorted_indices]
            
            bg_cumsum = np.cumsum(sorted_labels == 0)
            flare_cumsum = np.cumsum(sorted_labels == 1)
            total_cumsum = np.arange(1, len(sorted_times) + 1)
            
            ax.plot(sorted_times, bg_cumsum, label='Background Events', color='blue', linewidth=2)
            ax.plot(sorted_times, flare_cumsum, label='Flare Events', color='orange', linewidth=2)
            ax.plot(sorted_times, total_cumsum, label='Total Events', color='black', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Cumulative Event Count')
        ax.set_title('Cumulative Event Count Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'temporal_distribution_sample_{sample_idx}.png'),
                   dpi=150, bbox_inches='tight', facecolor=bg_color)
        plt.close()
    
    def _generate_statistics_report(self, background_events: np.ndarray,
                                   flare_events: np.ndarray,
                                   merged_events: np.ndarray,
                                   labels: np.ndarray,
                                   density_stats: Dict[str, float],
                                   sample_idx: int):
        
        report_path = os.path.join(self.output_dir, f'statistics_report_sample_{sample_idx}.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Event Pipeline Analysis Report - Sample {sample_idx}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Event Counts:\n")
            f.write(f"  DSEC Background Events: {len(background_events):,}\n")
            f.write(f"  DVS Flare Events: {len(flare_events):,}\n")
            f.write(f"  Merged Total Events: {len(merged_events):,}\n")
            
            if len(labels) > 0:
                bg_count = (labels == 0).sum()
                flare_count = (labels == 1).sum()
                f.write(f"  Merged Background Events: {bg_count:,}\n")
                f.write(f"  Merged Flare Events: {flare_count:,}\n")
            
            f.write("\n")
            
            f.write("Event Densities (events/ms):\n")
            f.write(f"  DSEC Background: {density_stats['background_density']:.1f}\n")
            f.write(f"  DVS Flare: {density_stats['flare_density']:.1f}\n")
            f.write(f"  Merged Total: {density_stats['merged_density']:.1f}\n")
            f.write(f"  Merged Background: {density_stats['merged_background_density']:.1f}\n")
            f.write(f"  Merged Flare: {density_stats['merged_flare_density']:.1f}\n")
            f.write("\n")
            
            f.write("Time Durations (ms):\n")
            f.write(f"  Background Duration: {density_stats['background_duration_ms']:.1f}\n")
            f.write(f"  Flare Duration: {density_stats['flare_duration_ms']:.1f}\n")
            f.write(f"  Merged Duration: {density_stats['merged_duration_ms']:.1f}\n")
            f.write("\n")
            
            f.write("Polarity Distribution:\n")
            
            def analyze_polarity(events, name):
                if len(events) > 0:
                    pos_count = (events[:, 3] == 1).sum()
                    neg_count = (events[:, 3] == 0).sum()
                    pos_ratio = pos_count / len(events) * 100
                    f.write(f"  {name}: ON={pos_count:,} ({pos_ratio:.1f}%), OFF={neg_count:,} ({100-pos_ratio:.1f}%)\n")
                else:
                    f.write(f"  {name}: No events\n")
            
            analyze_polarity(background_events, "DSEC Background")
            analyze_polarity(flare_events, "DVS Flare")
            analyze_polarity(merged_events, "Merged Total")
            
            f.write("\n")
            
            f.write("Spatial Distribution:\n")
            
            def analyze_spatial(events, name):
                if len(events) > 0:
                    x_range = f"{events[:, 0].min():.0f}-{events[:, 0].max():.0f}"
                    y_range = f"{events[:, 1].min():.0f}-{events[:, 1].max():.0f}"
                    x_center = events[:, 0].mean()
                    y_center = events[:, 1].mean()
                    f.write(f"  {name}: X=[{x_range}] Y=[{y_range}] Center=({x_center:.1f},{y_center:.1f})\n")
                else:
                    f.write(f"  {name}: No events\n")
            
            analyze_spatial(background_events, "DSEC Background")
            analyze_spatial(flare_events, "DVS Flare")
            analyze_spatial(merged_events, "Merged Total")
            
        print(f"📄 Statistics report saved: {report_path}")
        
        print(f"📊 Key Statistics for Sample {sample_idx}:")
        print(f"   DSEC Background: {len(background_events):,} events, {density_stats['background_density']:.1f} events/ms")
        print(f"   DVS Flare: {len(flare_events):,} events, {density_stats['flare_density']:.1f} events/ms")  
        print(f"   Merged Total: {len(merged_events):,} events, {density_stats['merged_density']:.1f} events/ms")
        
        target_min = 500
        if density_stats['merged_density'] < target_min:
            print(f"   ⚠️  Merged density too low: {density_stats['merged_density']:.1f} < {target_min}")
        else:
            print(f"   ✅ Merged density acceptable: {density_stats['merged_density']:.1f} events/ms")
    
    def _create_multi_resolution_event_visualizations(self, 
                                                    background_events: np.ndarray,
                                                    flare_events: np.ndarray, 
                                                    merged_events: np.ndarray,
                                                    labels: np.ndarray,
                                                    sample_idx: int):
        
        DSECDVS，
        。
        """
        print(f"🔍 Creating multi-resolution event visualizations for sample {sample_idx}...")
        
        sample_viz_dir = os.path.join(self.output_dir, f"multi_resolution_sample_{sample_idx}")
        os.makedirs(sample_viz_dir, exist_ok=True)
        
        resolution_strategies = [0.5, 1, 2, 4]
        
        event_types = {
            'dsec_background': background_events,
            'merged_total': merged_events
        }
        
        if len(flare_events) > 0:
            event_types['dvs_flare'] = flare_events
        
        for event_type, events in event_types.items():
            if len(events) == 0:
                print(f"   Skipping {event_type}: no events")
                continue
                
            type_dir = os.path.join(sample_viz_dir, event_type)
            os.makedirs(type_dir, exist_ok=True)
            
            print(f"   Processing {event_type}: {len(events)} events")
            
            if len(events) > 0:
                t_min = events[:, 2].min()
                t_max = events[:, 2].max()
                total_duration_us = t_max - t_min
                
                if total_duration_us <= 1000:
                    print(f"   {event_type} has short duration ({total_duration_us:.0f}μs), creating event-count based image sequence")
                    self._create_event_sequence_visualization(events, type_dir, event_type)
                    continue
                
                for resolution_factor in resolution_strategies:
                    resolution_dir = os.path.join(type_dir, f"{resolution_factor}x_resolution")
                    os.makedirs(resolution_dir, exist_ok=True)
                    
                    window_duration_us = total_duration_us / resolution_factor
                    num_windows = int(np.ceil(total_duration_us / window_duration_us))
                    
                    print(f"     {resolution_factor}x: {window_duration_us:.0f}μs windows, {num_windows} total")
                    
                    for window_idx in range(num_windows):
                        window_start = t_min + window_idx * window_duration_us
                        window_end = min(window_start + window_duration_us, t_max)
                        
                        mask = (events[:, 2] >= window_start) & (events[:, 2] < window_end)
                        window_events = events[mask]
                        
                        if len(window_events) == 0:
                            continue
                            
                        self._create_dvs_style_event_image(
                            window_events, resolution_dir, 
                            f"window_{window_idx:03d}_{resolution_factor}x", 
                            event_type, window_start, window_end
                        )
                
                print(f"   ✅ Created visualizations for {event_type}")
        
        print(f"✅ Multi-resolution visualizations saved to: {sample_viz_dir}")
    
    def _create_single_event_visualization(self, events: np.ndarray, output_dir: str,
                                         filename: str, event_type: str,
                                         t_start: float, t_end: float):
        import matplotlib.pyplot as plt

        bg_color = self.event_colors['facecolor']
        text_color = self.event_colors['text_color']

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        if len(events) == 0:
            ax.text(0.5, 0.5, 'No events in this window', 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2]
            p = events[:, 3]
            
            unique_polarities = np.unique(p)
            if np.any(unique_polarities < 0):
                pos_mask = p > 0
                neg_mask = p < 0
            else:
                pos_mask = p > 0
                neg_mask = p == 0
            
            if np.any(pos_mask):
                ax.scatter(x[pos_mask], y[pos_mask], c='red', s=1, alpha=0.7, label=f'ON ({np.sum(pos_mask)})')
            
            if np.any(neg_mask):
                ax.scatter(x[neg_mask], y[neg_mask], c='blue', s=1, alpha=0.7, label=f'OFF ({np.sum(neg_mask)})')
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{event_type.replace("_", " ").title()}\n'
                    f'Time: {(t_start/1000):.1f}-{(t_end/1000):.1f}ms ({len(events)} events)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
        plt.close()

        return output_path
    
    def _create_event_sequence_visualization(self, events: np.ndarray, output_dir: str, event_type: str):
        import matplotlib.pyplot as plt

        if len(events) == 0:
            return

        sequence_dir = os.path.join(output_dir, "event_sequence")
        os.makedirs(sequence_dir, exist_ok=True)

        events_per_frame = max(8, len(events) // 10)
        num_frames = max(1, len(events) // events_per_frame)

        print(f"     Creating {num_frames} frames with ~{events_per_frame} events each")

        for frame_idx in range(num_frames):
            start_idx = frame_idx * events_per_frame
            end_idx = min(start_idx + events_per_frame, len(events))
            frame_events = events[start_idx:end_idx]

            if len(frame_events) == 0:
                continue

            bg_color = self.event_colors['facecolor']
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor=bg_color)
            ax.set_facecolor(bg_color)

            x = frame_events[:, 0]
            y = frame_events[:, 1]
            p = frame_events[:, 3]

            unique_polarities = np.unique(p)
            if np.any(unique_polarities < 0):
                pos_mask = p > 0
                neg_mask = p < 0
            else:
                pos_mask = p > 0
                neg_mask = p == 0

            if np.any(pos_mask):
                ax.scatter(x[pos_mask], y[pos_mask], c=self.event_colors['ON'], s=8, alpha=0.9,
                          label=f'ON ({np.sum(pos_mask)})',
                          edgecolors=self.event_colors['edge_color'], linewidth=0.2)

            if np.any(neg_mask):
                ax.scatter(x[neg_mask], y[neg_mask], c=self.event_colors['OFF'], s=8, alpha=0.9,
                          label=f'OFF ({np.sum(neg_mask)})',
                          edgecolors=self.event_colors['edge_color'], linewidth=0.2)

            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.invert_yaxis()
            ax.set_xlabel('X coordinate', color=self.event_colors['text_color'])
            ax.set_ylabel('Y coordinate', color=self.event_colors['text_color'])
            ax.set_title(f'{event_type.replace("_", " ").title()} - Frame {frame_idx+1}/{num_frames}\\n'
                        f'Events: {len(frame_events)} ({start_idx}-{end_idx})',
                        color=self.event_colors['text_color'])
            ax.legend(facecolor=bg_color, edgecolor=self.event_colors['edge_color'],
                     labelcolor=self.event_colors['text_color'])
            ax.grid(True, alpha=0.2, color=self.event_colors['grid_color'])
            ax.tick_params(colors=self.event_colors['text_color'])

            output_path = os.path.join(sequence_dir, f"frame_{frame_idx:03d}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
            plt.close()

        print(f"     ✅ Saved {num_frames} event sequence frames to {sequence_dir}")
    
    def _create_dvs_style_event_image(self, events: np.ndarray, output_dir: str,
                                     filename: str, event_type: str,
                                     t_start: float, t_end: float):
        import matplotlib.pyplot as plt

        bg_color = self.event_colors['facecolor']
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        if len(events) == 0:
            ax.text(0.5, 0.5, 'No events in this window',
                   ha='center', va='center', transform=ax.transAxes,
                   color=self.event_colors['text_color'])
        else:
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2]
            p = events[:, 3]

            unique_polarities = np.unique(p)
            if np.any(unique_polarities < 0):
                pos_mask = p > 0
                neg_mask = p < 0
            else:
                pos_mask = p > 0
                neg_mask = p == 0

            if np.any(pos_mask):
                ax.scatter(x[pos_mask], y[pos_mask], c=self.event_colors['ON'], s=6, alpha=0.8,
                          label=f'ON ({np.sum(pos_mask)})',
                          edgecolors=self.event_colors['edge_color'], linewidth=0.1)

            if np.any(neg_mask):
                ax.scatter(x[neg_mask], y[neg_mask], c=self.event_colors['OFF'], s=6, alpha=0.8,
                          label=f'OFF ({np.sum(neg_mask)})',
                          edgecolors=self.event_colors['edge_color'], linewidth=0.1)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.set_xlabel('X coordinate', color=self.event_colors['text_color'])
        ax.set_ylabel('Y coordinate', color=self.event_colors['text_color'])
        ax.set_title(f'{event_type.replace("_", " ").title()}\\n'
                    f'Time: {(t_start/1000):.1f}-{(t_end/1000):.1f}ms ({len(events)} events)',
                    color=self.event_colors['text_color'])
        ax.legend(facecolor=bg_color, edgecolor=self.event_colors['edge_color'],
                 labelcolor=self.event_colors['text_color'])
        ax.grid(True, alpha=0.2, color=self.event_colors['grid_color'])
        ax.tick_params(colors=self.event_colors['text_color'])

        output_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
        plt.close()

        return output_path