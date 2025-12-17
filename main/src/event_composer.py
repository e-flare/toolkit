"""
Event Composer for EventMamba-FX
=================================

Step 2: Compose background events + flare events → merged events.
Output: Standard DVS H5 format for bg_events and merge_events.

Key Features:
- Read pre-generated flare events from Step 1
- Load DSEC background events
- Temporal merging with configurable strategies
- Standard DVS H5 format output
- Debug visualization for all three event types
"""

import os
import time
import random
import glob
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional

from src.dsec_efficient import DSECEventDatasetEfficient
from src.event_visualization_utils import EventVisualizer


class EventComposer:
    """
     - Step 2
     +  → 
    """
    
    def __init__(self, config: Dict):
        """

        Args:
            config: 
        """
        self.config = config
        self.composition_config = config.get('composition', {})

        self.test_mode = config.get('test_mode', False)
        generation_config = config['generation']

        if self.test_mode:
            output_paths = generation_config['test_output_paths']
            debug_paths = generation_config.get('test_debug_paths', generation_config['debug_paths'])
        else:
            output_paths = generation_config['output_paths']
            debug_paths = generation_config['debug_paths']

        self.flare_events_dir = output_paths['flare_events']
        self.light_source_events_dir = output_paths['light_source_events']

        self.background_only_dir = None
        if self.test_mode and 'background_only_events' in output_paths:
            self.background_only_dir = output_paths['background_only_events']
            os.makedirs(self.background_only_dir, exist_ok=True)
        
        self.merge_method = self.composition_config.get('merge_method', 'simple')
        self.generate_both_methods = self.composition_config.get('generate_both_methods', False)
        
        self.all_methods = [
            'simple',
            'simple_timeRandom',
            'physics_noRandom',
            'physics',
            'physics_noRandom_noTen',
            'physics_full'
        ]
        
        self.output_dirs = {}

        if self.test_mode:
            output_root = os.path.dirname(output_paths['flare_events'])  # output/test
        else:
            output_root = 'output/data'

        methods_to_create = [self.merge_method] if not self.generate_both_methods else self.all_methods

        for method in methods_to_create:
            if method in self.all_methods:
                self.output_dirs[method] = {
                    'stage1': os.path.join(output_root, f'{method}_method', 'background_with_light_events'),
                    'stage2': os.path.join(output_root, f'{method}_method', 'background_with_flare_events')
                }

        for method_name, paths in self.output_dirs.items():
            os.makedirs(paths['stage1'], exist_ok=True)
            os.makedirs(paths['stage2'], exist_ok=True)

        self.debug_mode = config.get('debug_mode', False)
        if self.debug_mode:
            self.debug_dir = debug_paths['event_composition']
            os.makedirs(self.debug_dir, exist_ok=True)

            bg_color = config.get('visualization_background', 'black')
            self.visualization_background = bg_color

            resolution = (config['data']['resolution_w'], config['data']['resolution_h'])
            self.visualizer = EventVisualizer(self.debug_dir, resolution, background_color=bg_color)
            print(f"🎯 EventComposer Debug Mode: {self.debug_dir}")
        else:
            self.visualization_background = 'black'
        
        self.dsec_dataset = DSECEventDatasetEfficient(
            dsec_path=config['data']['dsec_path'],
            flare_path="",
            time_window_us=config['data']['time_window_us']
        )
        
        self.bg_duration_ms = 100.0
        
        self.composition_start_id = self._get_next_composition_id()
        
        mode_label = "TEST" if self.test_mode else "TRAIN/VAL"
        print(f"🚀 EventComposer initialized (Dual-Stage Composition - {mode_label} Mode):")
        print(f"  ✅ CORRECTED LOGIC - Three separate compositions:")
        print(f"    - Stage 1: Background + Light Source → Clean scene")
        print(f"    - Stage 2: Background + Flare → Flare-contaminated scene")
        print(f"  Merge method: {self.merge_method}")
        print(f"  Generate both methods: {self.generate_both_methods}")
        print(f"  Inputs:")
        print(f"    - Flare events: {self.flare_events_dir}")
        print(f"    - Light Source events: {self.light_source_events_dir}")
        print(f"    - Background events: DSEC Dataset (randomly sampled)")
        print(f"  Outputs:")
        for method_name, paths in self.output_dirs.items():
            print(f"    - Method '{method_name}':")
            print(f"      - Stage 1 (BG+Light): {paths['stage1']}")
            print(f"      - Stage 2 (BG+Flare): {paths['stage2']}")
        if self.background_only_dir:
            print(f"    - 🆕 Test Mode Background-Only: {self.background_only_dir}")
        print(f"  DSEC dataset size: {len(self.dsec_dataset)} time windows")
        print(f"  Background duration: {self.bg_duration_ms:.0f}ms (fixed)")
        print(f"  Composition start ID: {self.composition_start_id} (continuing from existing files)")  # 🆕
        print(f"  Debug mode: {self.debug_mode}")
    
    def _get_next_composition_id(self) -> int:
        """
        ID，
        
        Returns:
            ID
        """
        import glob
        
        max_existing = 0
        
        for method_name, paths in self.output_dirs.items():
            stage1_files = glob.glob(os.path.join(paths['stage1'], "*.h5"))
            stage2_files = glob.glob(os.path.join(paths['stage2'], "*.h5"))
            
            method_max = max(len(stage1_files), len(stage2_files))
            max_existing = max(max_existing, method_max)
            
            if method_max > 0:
                print(f"📁 Found {method_name} files: {len(stage1_files)} stage1 + {len(stage2_files)} stage2")
        
        if max_existing > 0:
            print(f"🔢 Starting composition ID from: {max_existing}")
        
        return max_existing
    
    def load_flare_events(self, flare_file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        H5，

        Args:
            flare_file_path: H5

        Returns:
            Tuple of (events, metadata)
            - events:  [N, 4]  [t, x, y, p] (DVS)
            - metadata: H5
        """
        with h5py.File(flare_file_path, 'r') as f:
            events_group = f['events']

            t = events_group['t'][:]
            x = events_group['x'][:]
            y = events_group['y'][:]
            p = events_group['p'][:]

            events = np.column_stack([t, x, y, p])

            metadata = {}
            for key in events_group.attrs.keys():
                metadata[key] = events_group.attrs[key]

            return events.astype(np.float64), metadata
    
    def generate_background_events(self) -> Tuple[np.ndarray, Dict]:
        """
         - 100ms

        Returns:
            Tuple of (background_events, background_metadata)
            - background_events:  [N, 4]  [x, y, t, p] ()
            - background_metadata: DSEC
        """
        duration_ms = self.bg_duration_ms
        duration_us = int(duration_ms * 1000)

        idx = random.randint(0, len(self.dsec_dataset) - 1)

        dsec_file_basename = f"file_{idx}"
        if hasattr(self.dsec_dataset, 'file_metadata'):
            current_idx = 0
            for file_meta in self.dsec_dataset.file_metadata:
                if current_idx + file_meta['num_windows'] > idx:
                    dsec_file_name = file_meta['file_path']
                    dsec_file_basename = os.path.basename(dsec_file_name).replace('.h5', '')
                    break
                current_idx += file_meta['num_windows']

        # ==================== MODIFICATION: START ====================
        raw_events = self.dsec_dataset[idx]

        if len(raw_events) == 0:
            return np.empty((0, 4), dtype=np.float64), {'dsec_file': dsec_file_basename, 'time_window_ms': '0-0'}

        
        x = np.asarray(raw_events[:, 0], dtype=np.float64)
        y = np.asarray(raw_events[:, 1], dtype=np.float64)
        t = np.asarray(raw_events[:, 2], dtype=np.float64)
        p = np.asarray(raw_events[:, 3], dtype=np.float64)
        
        background_events = np.column_stack([x, y, t, p]).astype(np.float64)
        # ===================== MODIFICATION: END =====================
        
        start_time_ms = 0
        end_time_ms = 0

        if len(background_events) > 0:
            t_min = background_events[:, 2].min()
            t_max = background_events[:, 2].max()
            current_duration = t_max - t_min

            if current_duration > duration_us:
                max_start_offset = current_duration - duration_us
                start_offset = random.uniform(0, max_start_offset)
                start_time = t_min + start_offset
                end_time = start_time + duration_us

                start_time_ms = int(start_time / 1000)
                end_time_ms = int(end_time / 1000)

                mask = (background_events[:, 2] >= start_time) & (background_events[:, 2] < end_time)
                background_events = background_events[mask]

            if len(background_events) > 0:
                t_min_bg = background_events[:, 2].min()
                background_events[:, 2] = background_events[:, 2] - t_min_bg

        bg_metadata = {
            'dsec_file': dsec_file_basename,
            'time_window_ms': f"{start_time_ms}-{end_time_ms}"
        }

        return (background_events if len(background_events) > 0 else np.empty((0, 4), dtype=np.float64)), bg_metadata
    
    def convert_flare_to_project_format(self, flare_events: np.ndarray) -> np.ndarray:
        """
        DVS
        
        Args:
            flare_events: [N, 4] DVS [t, x, y, p]
            
        Returns:
            [N, 4]  [x, y, t, p]
        """
        if len(flare_events) == 0:
            return np.empty((0, 4))
        
        project_events = np.zeros_like(flare_events)
        project_events[:, 0] = flare_events[:, 1]  # x
        project_events[:, 1] = flare_events[:, 2]  # y
        project_events[:, 2] = flare_events[:, 0]  # t
        project_events[:, 3] = flare_events[:, 3]  # p
        
        if len(project_events) > 0:
            import random
            t_min = project_events[:, 2].min()
            project_events[:, 2] = project_events[:, 2] - t_min
            
            random_offset_us = random.uniform(0, 5000)
            project_events[:, 2] = project_events[:, 2] + random_offset_us
        
        project_events[:, 3] = np.where(project_events[:, 3] > 0, 1, -1)
        
        return project_events
    
    def convert_to_dvs_format(self, events: np.ndarray) -> np.ndarray:
        """
        DVS
        
        Args:
            events: [N, 4]  [x, y, t, p]
            
        Returns:
            [N, 4] DVS [t, x, y, p]
        """
        if len(events) == 0:
            return np.empty((0, 4))
        
        dvs_events = np.zeros_like(events)
        dvs_events[:, 0] = events[:, 2]  # t
        dvs_events[:, 1] = events[:, 0]  # x
        dvs_events[:, 2] = events[:, 1]  # y
        dvs_events[:, 3] = events[:, 3]  # p
        
        return dvs_events
    
    def _merge_events_physics(self, events1: np.ndarray, events2: np.ndarray, 
                              weight1: float, weight2: float) -> np.ndarray:
        """
        Merges two event streams based on a physically-grounded probabilistic model.
        
        Args:
            events1: First event stream (e.g., background).
            events2: Second event stream (e.g., light source or flare).
            weight1: The intensity weight to accumulate for each event in events1.
            weight2: The intensity weight to accumulate for each event in events2.
            
        Returns:
            The merged event array.
        """
        
        params = self.composition_config.get('physics_params', {})
        jitter_us = params.get('temporal_jitter_us', 50)
        epsilon_raw = params.get('epsilon', 1e-9)
        epsilon = float(epsilon_raw)
        W, H = self.config['data']['resolution_w'], self.config['data']['resolution_h']

        Y_est1 = np.zeros((H, W), dtype=np.float32)
        x1, y1 = None, None
        if len(events1) > 0:
            x1 = np.clip(events1[:, 0].astype(np.int32), 0, W-1)
            y1 = np.clip(events1[:, 1].astype(np.int32), 0, H-1)
            np.add.at(Y_est1, (y1, x1), weight1)

        Y_est2 = np.zeros((H, W), dtype=np.float32)
        x2, y2 = None, None
        if len(events2) > 0:
            x2 = np.clip(events2[:, 0].astype(np.int32), 0, W-1)
            y2 = np.clip(events2[:, 1].astype(np.int32), 0, H-1)
            np.add.at(Y_est2, (y2, x2), weight2)

        
        A_det = Y_est2 / (Y_est1 + Y_est2 + epsilon)
        
        stochastic_strength = float(params.get('stochastic_strength', 0.0))
        
        if stochastic_strength > 0:
            noise_scale_map = 4.0 * A_det * (1.0 - A_det)
            
            random_noise = np.random.uniform(-1.0, 1.0, size=A_det.shape)
            
            A_stochastic = A_det + stochastic_strength * noise_scale_map * random_noise
            
            A = np.clip(A_stochastic, 0.0, 1.0)
        else:
            A = A_det
        
        self._last_weight_map = A
        
        if len(events1) > 0 and x1 is not None and y1 is not None:
            prob_keep1 = 1.0 - A[y1, x1]
            mask1 = np.random.rand(len(events1)) < prob_keep1
            kept_events1 = events1[mask1]
        else:
            kept_events1 = np.empty((0, 4), dtype=np.float64)
            
        if len(events2) > 0 and x2 is not None and y2 is not None:
            prob_keep2 = A[y2, x2]
            mask2 = np.random.rand(len(events2)) < prob_keep2
            kept_events2 = events2[mask2]
        else:
            kept_events2 = np.empty((0, 4), dtype=np.float64)

        if len(kept_events1) == 0 and len(kept_events2) == 0:
            return np.empty((0, 4), dtype=np.float64)
        elif len(kept_events1) == 0:
            merged_events = kept_events2
        elif len(kept_events2) == 0:
            merged_events = kept_events1
        else:
            merged_events = np.vstack([kept_events1, kept_events2])
        
        if jitter_us > 0 and len(merged_events) > 0:
            time_jitter = np.random.uniform(-jitter_us, jitter_us, len(merged_events))
            merged_events[:, 2] += time_jitter
            
        if len(merged_events) > 0:
            sort_indices = np.argsort(merged_events[:, 2])
            merged_events = merged_events[sort_indices]
            
        return merged_events
    
    def _merge_events_simple_timeRandom(self, events1: np.ndarray, events2: np.ndarray) -> np.ndarray:
        merged = self._merge_events_simple(events1, events2)
        
        if len(merged) > 0:
            params = self.composition_config.get('physics_params', {})
            jitter_us = params.get('temporal_jitter_us', 10)
            if jitter_us > 0:
                time_jitter = np.random.uniform(-jitter_us, jitter_us, len(merged))
                merged[:, 2] += time_jitter
                
                sort_indices = np.argsort(merged[:, 2])
                merged = merged[sort_indices]
        
        return merged
    
    def _merge_events_physics_noRandom(self, events1: np.ndarray, events2: np.ndarray, 
                                      weight1: float, weight2: float) -> np.ndarray:
        return self._merge_events_physics_core(events1, events2, weight1, weight2, jitter_us=0)
    
    def _merge_events_physics_noRandom_noTen(self, events1: np.ndarray, events2: np.ndarray,
                                           weight1: float, weight2: float) -> np.ndarray:
        return self._merge_events_physics_noRandom(events1, events2, weight1, weight2)
    
    def _merge_events_physics_full(self, events1: np.ndarray, events2: np.ndarray, 
                                  weight1: float, weight2: float) -> np.ndarray:
        return self._merge_events_physics2(events1, events2, weight1, weight2)
    
    def _merge_events_physics_core(self, events1: np.ndarray, events2: np.ndarray, 
                                  weight1: float, weight2: float, jitter_us: float = None) -> np.ndarray:
        if jitter_us is None:
            params = self.composition_config.get('physics_params', {})
            jitter_us = params.get('temporal_jitter_us', 10)
        
        original_jitter = self.composition_config.get('physics_params', {}).get('temporal_jitter_us', 10)
        
        if 'physics_params' not in self.composition_config:
            self.composition_config['physics_params'] = {}
        self.composition_config['physics_params']['temporal_jitter_us'] = jitter_us
        
        result = self._merge_events_physics(events1, events2, weight1, weight2)
        
        self.composition_config['physics_params']['temporal_jitter_us'] = original_jitter
        
        return result

    def _merge_events_physics2(self, events1: np.ndarray, events2: np.ndarray, 
                               weight1: float, weight2: float) -> np.ndarray:
        """
         - 100msN，A(x,y,t)
        
        Args:
            events1: First event stream (e.g., background).
            events2: Second event stream (e.g., light source or flare).
            weight1: The intensity weight to accumulate for each event in events1.
            weight2: The intensity weight to accumulate for each event in events2.
            
        Returns:
            The merged event array with temporal weight adaptation.
        """
        import gc
        
        params = self.composition_config.get('physics2_params', {})
        num_segments = params.get('num_temporal_segments', 10)
        jitter_us = params.get('temporal_jitter_us', 10)
        epsilon = float(params.get('epsilon', 1e-9))
        stochastic_strength = float(params.get('stochastic_strength', 0.1))
        
        W, H = self.config['data']['resolution_w'], self.config['data']['resolution_h']
        
        if len(events1) == 0 and len(events2) == 0:
            return np.empty((0, 4), dtype=np.float64)
        elif len(events1) == 0:
            return events2.copy()
        elif len(events2) == 0:
            return events1.copy()
        
        all_times = []
        if len(events1) > 0:
            all_times.append(events1[:, 2])
        if len(events2) > 0:
            all_times.append(events2[:, 2])
        
        all_times = np.concatenate(all_times)
        t_min, t_max = all_times.min(), all_times.max()
        total_duration = t_max - t_min
        
        if total_duration <= 0:
            return self._merge_events_physics(events1, events2, weight1, weight2)
        
        segment_duration = total_duration / num_segments
        
        print(f"      Physics2: {num_segments} segments, {segment_duration/1000:.1f}ms each")
        
        merged_segments = []
        weight_maps_temporal = []
        
        for seg_idx in range(num_segments):
            seg_start = t_min + seg_idx * segment_duration
            seg_end = seg_start + segment_duration
            
            if len(events1) > 0:
                mask1 = (events1[:, 2] >= seg_start) & (events1[:, 2] < seg_end)
                seg_events1 = events1[mask1]
            else:
                seg_events1 = np.empty((0, 4), dtype=np.float64)
                
            if len(events2) > 0:
                mask2 = (events2[:, 2] >= seg_start) & (events2[:, 2] < seg_end)
                seg_events2 = events2[mask2]
            else:
                seg_events2 = np.empty((0, 4), dtype=np.float64)
            
            if len(seg_events1) == 0 and len(seg_events2) == 0:
                weight_maps_temporal.append(np.zeros((H, W), dtype=np.float32))
                continue
            
            A_seg = self._compute_segment_weight_map(seg_events1, seg_events2, 
                                                   weight1, weight2, epsilon, 
                                                   stochastic_strength, W, H)
            weight_maps_temporal.append(A_seg.copy())
            
            seg_merged = self._apply_segment_gating(seg_events1, seg_events2, A_seg, jitter_us)
            
            if len(seg_merged) > 0:
                merged_segments.append(seg_merged)
            
            del seg_events1, seg_events2, A_seg
            if seg_idx % 5 == 0:
                gc.collect()
        
        if merged_segments:
            final_merged = np.vstack(merged_segments)
            
            if len(final_merged) > 0:
                sort_indices = np.argsort(final_merged[:, 2])
                final_merged = final_merged[sort_indices]
        else:
            final_merged = np.empty((0, 4), dtype=np.float64)
        
        self._last_temporal_weight_maps = weight_maps_temporal
        self._last_temporal_segments_info = {
            'num_segments': num_segments,
            'segment_duration_us': segment_duration,
            't_min': t_min,
            't_max': t_max
        }
        
        del merged_segments, weight_maps_temporal, all_times
        gc.collect()
        
        return final_merged
    
    def _compute_segment_weight_map(self, events1: np.ndarray, events2: np.ndarray, 
                                  weight1: float, weight2: float, epsilon: float,
                                  stochastic_strength: float, W: int, H: int) -> np.ndarray:
        Y_est1 = np.zeros((H, W), dtype=np.float32)
        if len(events1) > 0:
            x1 = np.clip(events1[:, 0].astype(np.int32), 0, W-1)
            y1 = np.clip(events1[:, 1].astype(np.int32), 0, H-1)
            np.add.at(Y_est1, (y1, x1), weight1)

        Y_est2 = np.zeros((H, W), dtype=np.float32)
        if len(events2) > 0:
            x2 = np.clip(events2[:, 0].astype(np.int32), 0, W-1)
            y2 = np.clip(events2[:, 1].astype(np.int32), 0, H-1)
            np.add.at(Y_est2, (y2, x2), weight2)

        A_det = Y_est2 / (Y_est1 + Y_est2 + epsilon)
        
        if stochastic_strength > 0:
            noise_scale_map = 4.0 * A_det * (1.0 - A_det)
            random_noise = np.random.uniform(-1.0, 1.0, size=A_det.shape)
            A_stochastic = A_det + stochastic_strength * noise_scale_map * random_noise
            A = np.clip(A_stochastic, 0.0, 1.0)
        else:
            A = A_det
            
        return A
    
    def _apply_segment_gating(self, events1: np.ndarray, events2: np.ndarray, 
                            A: np.ndarray, jitter_us: float) -> np.ndarray:
        kept_events = []
        
        if len(events1) > 0:
            W = self.config['data']['resolution_w']
            x1 = np.clip(events1[:, 0].astype(np.int32), 0, W-1)
            y1 = np.clip(events1[:, 1].astype(np.int32), 0, self.config['data']['resolution_h']-1)
            prob_keep1 = 1.0 - A[y1, x1]
            mask1 = np.random.rand(len(events1)) < prob_keep1
            if np.any(mask1):
                kept_events.append(events1[mask1])
        
        if len(events2) > 0:
            W = self.config['data']['resolution_w']
            x2 = np.clip(events2[:, 0].astype(np.int32), 0, W-1)
            y2 = np.clip(events2[:, 1].astype(np.int32), 0, self.config['data']['resolution_h']-1)
            prob_keep2 = A[y2, x2]
            mask2 = np.random.rand(len(events2)) < prob_keep2
            if np.any(mask2):
                kept_events.append(events2[mask2])
        
        if kept_events:
            merged = np.vstack(kept_events)
            if jitter_us > 0 and len(merged) > 0:
                time_jitter = np.random.uniform(-jitter_us, jitter_us, len(merged))
                merged[:, 2] += time_jitter
            return merged
        else:
            return np.empty((0, 4), dtype=np.float64)
    
    def _merge_events_simple(self, background_events: np.ndarray, flare_events: np.ndarray) -> np.ndarray:
        """
         - 
        
        Args:
            background_events: [N, 4]  [x, y, t, p]
            flare_events: [N, 4]  [x, y, t, p]
            
        Returns:
             [N, 4]  [x, y, t, p]
        """
        if len(background_events) == 0 and len(flare_events) == 0:
            return np.empty((0, 4))
        elif len(background_events) == 0:
            return flare_events
        elif len(flare_events) == 0:
            return background_events
        
        all_events = np.vstack([background_events, flare_events])
        
        sort_indices = np.argsort(all_events[:, 2])
        merged_events = all_events[sort_indices]
        
        return merged_events

    def merge_events(self, events1: np.ndarray, events2: np.ndarray,
                     method: str = "simple", 
                     weight1: float = 1.0, weight2: float = 1.0) -> np.ndarray:
        """
         - 6
        
        Args:
            events1: [N, 4]  [x, y, t, p] -  ()
            events2: [N, 4]  [x, y, t, p] -  (/)  
            method: ，6:
                - simple: 
                - simple_timeRandom: +
                - physics_noRandom: A+
                - physics: A+
                - physics_noRandom_noTen: A++()
                - physics_full: A+(physics2)
            weight1:  (physics)
            weight2:  (physics)
            
        Returns:
             [N, 4]  [x, y, t, p]
        """
        method_dispatch = {
            'simple': lambda: self._merge_events_simple(events1, events2),
            'simple_timeRandom': lambda: self._merge_events_simple_timeRandom(events1, events2),
            'physics_noRandom': lambda: self._merge_events_physics_noRandom(events1, events2, weight1, weight2),
            'physics': lambda: self._merge_events_physics(events1, events2, weight1, weight2),
            'physics_noRandom_noTen': lambda: self._merge_events_physics_noRandom_noTen(events1, events2, weight1, weight2),
            'physics_full': lambda: self._merge_events_physics_full(events1, events2, weight1, weight2)
        }
        
        if method not in method_dispatch:
            raise ValueError(f"Unknown merge method: {method}. Supported: {list(method_dispatch.keys())}")
        
        return method_dispatch[method]()
    
    def save_events_dvs_format(self, events: np.ndarray, output_path: str, metadata: Optional[Dict] = None):
        """
        DVSH5
        
        Args:
            events: ，DVS
            output_path: 
            metadata: 
        """
        if len(events) == 0:
            print(f"⚠️  Warning: No events to save for {output_path}")
            return
        
        if events.shape[1] == 4:
            t_col_0 = events[:, 0]
            t_col_2 = events[:, 2]
            
            if np.mean(t_col_0) > np.mean(t_col_2) and np.std(t_col_0) > np.std(t_col_2):
                dvs_events = events
            else:
                dvs_events = self.convert_to_dvs_format(events)
        else:
            raise ValueError(f"Unexpected event array shape: {events.shape}")
        
        with h5py.File(output_path, 'w') as f:
            events_group = f.create_group('events')
            
            events_group.create_dataset('t', data=dvs_events[:, 0].astype(np.int64), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=dvs_events[:, 1].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=dvs_events[:, 2].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=dvs_events[:, 3].astype(np.int8), 
                                      compression='gzip', compression_opts=9)
            
            events_group.attrs['num_events'] = len(dvs_events)
            events_group.attrs['resolution_height'] = self.config['data']['resolution_h']
            events_group.attrs['resolution_width'] = self.config['data']['resolution_w']
            events_group.attrs['composition_time'] = time.time()
            
            if metadata:
                for key, value in metadata.items():
                    events_group.attrs[key] = value
    
    def compose_single_sequence(self, flare_file_path: str, light_source_file_path: str, sequence_id: int) -> Tuple[str, str]:
        """
         - 
        
        Args:
            flare_file_path: 
            light_source_file_path: 
            sequence_id: ID
            
        Returns:
            (bg_light_file, full_scene_file) 
        """
        start_time = time.time()
        
        print(f"  Processing flare file: {os.path.basename(flare_file_path)}")
        print(f"  Processing light source file: {os.path.basename(light_source_file_path)}")

        flare_events_dvs, flare_metadata = self.load_flare_events(flare_file_path)
        flare_events_project = self.convert_flare_to_project_format(flare_events_dvs)

        light_source_events_dvs, light_metadata = self.load_flare_events(light_source_file_path)
        light_source_events_project = self.convert_flare_to_project_format(light_source_events_dvs)

        background_events_project, bg_metadata = self.generate_background_events()

        flare_img_name = "unknown"
        if 'flare_image_path' in flare_metadata:
            flare_img_path = flare_metadata['flare_image_path']
            if isinstance(flare_img_path, (str, bytes)):
                flare_img_path = flare_img_path.decode() if isinstance(flare_img_path, bytes) else flare_img_path
                flare_img_name = os.path.splitext(os.path.basename(flare_img_path))[0]

        dsec_name = bg_metadata.get('dsec_file', 'unknown').replace('.h5', '')
        time_window = bg_metadata.get('time_window_ms', '0-0')
        descriptive_suffix = f"{flare_img_name}_{dsec_name}_t{time_window}"
        
        print(f"    Background events: {len(background_events_project):,}")
        print(f"    Light source events: {len(light_source_events_project):,}")
        print(f"    Flare events: {len(flare_events_project):,}")
        
        def _run_composition_for_method(method_name: str):
            print(f"    Running composition for method: '{method_name}'")
            
            params = self.composition_config.get('physics_params', {})
            
            # --- Stage 1: BG + Light Source ---
            bg_weight = params.get('background_event_weight', 0.2)
            light_weight = params.get('light_source_event_weight', 1.0)
            s1_merged = self.merge_events(background_events_project, 
                                          light_source_events_project, 
                                          method=method_name,
                                          weight1=bg_weight, weight2=light_weight)
            
            if self.debug_mode and method_name in ['physics', 'physics_noRandom', 'physics_noRandom_noTen']:
                self._save_weight_map_visualization(sequence_id, f"stage1_bg_light_{method_name}")
            elif self.debug_mode and method_name == 'physics_full':
                self._save_temporal_weight_maps_visualization(sequence_id, f"stage1_bg_light_{method_name}")
            
            if self.debug_mode:
                self._save_stage_events_visualization(s1_merged, sequence_id, 
                                                    f"{method_name}_stage1_bg_light",
                                                    f"Stage 1 ({method_name}): Background + Light")

            
            bg_weight = params.get('background_event_weight', 0.2)
            flare_weight = params.get('flare_intensity_multiplier', 1.0)
            s2_merged = self.merge_events(background_events_project, 
                                          flare_events_project,
                                          method=method_name,
                                          weight1=bg_weight, weight2=flare_weight)

            if self.debug_mode and method_name in ['physics', 'physics_noRandom', 'physics_noRandom_noTen']:
                self._save_weight_map_visualization(sequence_id, f"stage2_full_scene_{method_name}")
            elif self.debug_mode and method_name == 'physics_full':
                self._save_temporal_weight_maps_visualization(sequence_id, f"stage2_full_scene_{method_name}")
            
            if self.debug_mode:
                self._save_stage_events_visualization(s2_merged, sequence_id, 
                                                    f"{method_name}_stage2_bg_flare",
                                                    f"Stage 2 ({method_name}): Background + Flare")

            actual_composition_id = self.composition_start_id + sequence_id

            if self.test_mode:
                base_name = f"composed_{actual_composition_id:05d}_{descriptive_suffix}"
            else:
                base_name = f"composed_{actual_composition_id:05d}"

            s1_path = os.path.join(self.output_dirs[method_name]['stage1'], f"{base_name}_bg_light.h5")
            s2_path = os.path.join(self.output_dirs[method_name]['stage2'], f"{base_name}_bg_flare.h5")
            
            bg_light_metadata = {
                'event_type': 'background_with_light',
                'method': method_name,
                'background_events': len(background_events_project),
                'light_source_events': len(light_source_events_project),
                'stage1_merged_events': len(s1_merged),
                'source_light_file': os.path.basename(light_source_file_path)
            }
            self.save_events_dvs_format(s1_merged, s1_path, bg_light_metadata)
            
            full_scene_metadata = {
                'event_type': 'background_with_flare',
                'method': method_name,
                'background_events': len(background_events_project),
                'flare_events': len(flare_events_project),
                'total_events': len(s2_merged),
                'source_flare_file': os.path.basename(flare_file_path)
            }
            self.save_events_dvs_format(s2_merged, s2_path, full_scene_metadata)
            
            print(f"      Stage 1 ({method_name}): {len(s1_merged):,} events")
            print(f"      Stage 2 ({method_name}): {len(s2_merged):,} events")

            return None, None, s1_path, s2_path

        final_paths = ()
        if self.generate_both_methods:
            results = {}
            main_method_paths = None
            
            print(f"    6...")
            for method in self.all_methods:
                if method in self.output_dirs:
                    try:
                        _, _, s1_method, s2_method = _run_composition_for_method(method)
                        results[method] = (s1_method, s2_method)
                        
                        if method == self.merge_method:
                            main_method_paths = (s1_method, s2_method)
                            
                        print(f"      ✅ {method} ")
                    except Exception as e:
                        print(f"      ❌ {method} : {e}")
            
            final_paths = main_method_paths if main_method_paths else next(iter(results.values()), ())
                
        else:
            _, _, s1_p, s2_p = _run_composition_for_method(self.merge_method)
            final_paths = (s1_p, s2_p)
            
        composition_time = time.time() - start_time
        print(f"    Total composition time: {composition_time:.2f}s")

        if self.test_mode and self.background_only_dir is not None:
            actual_composition_id = self.composition_start_id + sequence_id

            bg_only_filename = f"composed_{actual_composition_id:05d}_{descriptive_suffix}_bg_only.h5"
            bg_only_path = os.path.join(self.background_only_dir, bg_only_filename)

            bg_only_metadata = {
                'event_type': 'background_only',
                'method': 'nolight',
                'background_events': len(background_events_project),
                'source_flare_file': os.path.basename(flare_file_path),
                'source_light_file': os.path.basename(light_source_file_path)
            }
            self.save_events_dvs_format(background_events_project, bg_only_path, bg_only_metadata)
            print(f"    🆕 Saved background-only events: {len(background_events_project):,} events")

        if self.debug_mode:
            debug_events = {
                "01_background_raw": background_events_project,
                "02_light_source_raw": light_source_events_project,
                "03_flare_raw": flare_events_project,
            }
            debug_metadata = {
                'flare_file': os.path.basename(flare_file_path),
                'light_source_file': os.path.basename(light_source_file_path),
            }
            self._save_debug_visualization(debug_events, sequence_id, debug_metadata)

        del background_events_project, flare_events_project, light_source_events_project
        del flare_events_dvs, light_source_events_dvs
        import gc
        gc.collect()

        return final_paths
    
    def _save_weight_map_visualization(self, sequence_id: int, stage_name: str):
        """Saves the last computed weight map A(x,y) as a heatmap."""

        debug_seq_dir = os.path.join(self.debug_dir, f"composition_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)

        if hasattr(self, '_last_weight_map') and self._last_weight_map is not None:
            import cv2
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            A = self._last_weight_map
            H, W = A.shape

            bg_color = self.visualization_background
            text_color = 'black' if bg_color == 'white' else 'white'

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=bg_color)
            ax1.set_facecolor(bg_color)
            ax2.set_facecolor(bg_color)
            
            im1 = ax1.imshow(A, cmap='viridis', vmin=0, vmax=1)
            ax1.set_title(f'Weight Map A(x,y) - {stage_name}', color=text_color)
            ax1.set_xlabel('X (pixels)', color=text_color)
            ax1.set_ylabel('Y (pixels)', color=text_color)
            ax1.tick_params(colors=text_color)
            cbar1 = plt.colorbar(im1, ax=ax1, label='Probability')
            cbar1.ax.yaxis.label.set_color(text_color)
            cbar1.ax.tick_params(colors=text_color)

            ax2.hist(A.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_title('Weight Distribution', color=text_color)
            ax2.set_xlabel('Weight Value', color=text_color)
            ax2.set_ylabel('Pixel Count', color=text_color)
            ax2.tick_params(colors=text_color)
            ax2.grid(True, alpha=0.3, color='gray')

            mean_weight = np.mean(A)
            std_weight = np.std(A)
            max_weight = np.max(A)
            min_weight = np.min(A)
            ax2.axvline(mean_weight, color='red', linestyle='--', label=f'Mean: {mean_weight:.3f}')
            legend = ax2.legend(facecolor=bg_color, edgecolor=text_color)
            plt.setp(legend.get_texts(), color=text_color)

            plt.tight_layout()

            vis_path = os.path.join(debug_seq_dir, f"weight_map_{stage_name}.png")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
            plt.close()
            
            heatmap_normalized = (A * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_VIRIDIS)
            heatmap_path = os.path.join(debug_seq_dir, f"weight_heatmap_{stage_name}.png")
            cv2.imwrite(heatmap_path, heatmap_colored)
            
            stats_path = os.path.join(debug_seq_dir, f"weight_stats_{stage_name}.txt")
            with open(stats_path, 'w') as f:
                f.write(f"Weight Map Statistics - {stage_name}\\n")
                f.write(f"=====================================\\n")
                f.write(f"Resolution: {W}x{H}\\n")
                f.write(f"Mean weight: {mean_weight:.6f}\\n")
                f.write(f"Std weight: {std_weight:.6f}\\n")
                f.write(f"Min weight: {min_weight:.6f}\\n")
                f.write(f"Max weight: {max_weight:.6f}\\n")
                f.write(f"Non-zero pixels: {np.count_nonzero(A)} ({np.count_nonzero(A)/(W*H)*100:.2f}%)\\n")
            
            print(f"      Weight map saved: {vis_path}")
            self._last_weight_map = None
    
    def _save_temporal_weight_maps_visualization(self, sequence_id: int, stage_name: str):

        debug_seq_dir = os.path.join(self.debug_dir, f"composition_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)

        if (hasattr(self, '_last_temporal_weight_maps') and
            hasattr(self, '_last_temporal_segments_info') and
            self._last_temporal_weight_maps is not None):

            import matplotlib.pyplot as plt
            import cv2
            import gc

            bg_color = self.visualization_background
            text_color = 'black' if bg_color == 'white' else 'white'

            weight_maps = self._last_temporal_weight_maps
            segments_info = self._last_temporal_segments_info
            num_segments = segments_info['num_segments']

            fig = plt.figure(figsize=(16, 12), facecolor=bg_color)
            
            cols = int(np.ceil(np.sqrt(num_segments)))
            rows = int(np.ceil(num_segments / cols))
            
            all_weights = []
            for weight_map in weight_maps:
                if weight_map.size > 0:
                    all_weights.append(weight_map.flatten())
            
            if all_weights:
                all_weights = np.concatenate(all_weights)
                vmin, vmax = np.min(all_weights), np.max(all_weights)
            else:
                vmin, vmax = 0, 1
            
            for i, weight_map in enumerate(weight_maps):
                ax = fig.add_subplot(rows, cols, i + 1)
                ax.set_facecolor(bg_color)

                if weight_map.size > 0 and np.any(weight_map > 0):
                    im = ax.imshow(weight_map, cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_title(f'Segment {i+1}\n({i*segments_info["segment_duration_us"]/1000:.1f}-{(i+1)*segments_info["segment_duration_us"]/1000:.1f}ms)',
                               color=text_color)
                else:
                    ax.imshow(np.zeros_like(weight_maps[0]), cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_title(f'Segment {i+1}\n(No events)', color=text_color)

                ax.set_xlabel('X (pixels)', color=text_color)
                ax.set_ylabel('Y (pixels)', color=text_color)
                ax.tick_params(labelsize=8, colors=text_color)

            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar_ax.set_facecolor(bg_color)
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Weight A(x,y,t)', rotation=270, labelpad=15, color=text_color)
            cbar.ax.tick_params(colors=text_color)

            plt.suptitle(f'Temporal Weight Maps A(x,y,t) - {stage_name}\n'
                        f'Physics2: {num_segments} segments, '
                        f'{segments_info["segment_duration_us"]/1000:.1f}ms each',
                        fontsize=14, color=text_color)
            plt.tight_layout()

            temporal_vis_path = os.path.join(debug_seq_dir, f"temporal_weight_maps_{stage_name}.png")
            plt.savefig(temporal_vis_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
            plt.close()
            
            stats_path = os.path.join(debug_seq_dir, f"temporal_weight_stats_{stage_name}.txt")
            with open(stats_path, 'w') as f:
                f.write(f"Temporal Weight Maps Statistics - {stage_name}\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Method: Physics2 (Time-varying)\n")
                f.write(f"Number of segments: {num_segments}\n")
                f.write(f"Segment duration: {segments_info['segment_duration_us']/1000:.2f}ms\n")
                f.write(f"Total duration: {(segments_info['t_max'] - segments_info['t_min'])/1000:.2f}ms\n\n")
                
                f.write("Per-segment statistics:\n")
                f.write("-" * 30 + "\n")
                
                for i, weight_map in enumerate(weight_maps):
                    if weight_map.size > 0:
                        mean_w = np.mean(weight_map)
                        std_w = np.std(weight_map)
                        max_w = np.max(weight_map)
                        min_w = np.min(weight_map)
                        nonzero_pixels = np.count_nonzero(weight_map)
                        total_pixels = weight_map.size
                        
                        f.write(f"Segment {i+1:2d} ({i*segments_info['segment_duration_us']/1000:5.1f}-{(i+1)*segments_info['segment_duration_us']/1000:5.1f}ms): "
                               f"mean={mean_w:.4f}, std={std_w:.4f}, "
                               f"range=[{min_w:.4f}, {max_w:.4f}], "
                               f"active={nonzero_pixels}/{total_pixels} ({nonzero_pixels/total_pixels*100:.1f}%)\n")
                    else:
                        f.write(f"Segment {i+1:2d}: No weight map data\n")
            
            if len(weight_maps) > 0:
                mid_idx = len(weight_maps) // 2
                mid_weight_map = weight_maps[mid_idx]
                
                if mid_weight_map.size > 0:
                    normalized = ((mid_weight_map - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
                    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
                    mid_heatmap_path = os.path.join(debug_seq_dir, f"temporal_weight_heatmap_mid_{stage_name}.png")
                    cv2.imwrite(mid_heatmap_path, colored)
                
                weight_variations = [np.std(wm) for wm in weight_maps if wm.size > 0]
                if weight_variations:
                    max_var_idx = np.argmax(weight_variations)
                    max_var_weight_map = weight_maps[max_var_idx]
                    
                    normalized = ((max_var_weight_map - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
                    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
                    max_var_heatmap_path = os.path.join(debug_seq_dir, f"temporal_weight_heatmap_maxvar_{stage_name}.png")
                    cv2.imwrite(max_var_heatmap_path, colored)
            
            print(f"      Temporal weight maps saved: {temporal_vis_path}")
            
            self._last_temporal_weight_maps = None
            self._last_temporal_segments_info = None
            
            del weight_maps, fig
            gc.collect()
    
    def _save_stage_events_visualization(self, events: np.ndarray, sequence_id: int, 
                                       stage_name: str, title: str):
        """
        stage
        Step1
        
        Args:
            events:  [N, 4]  [x, y, t, p]
            sequence_id: ID
            stage_name: stage， "simple_stage1_bg_light"
            title: 
        """
        if len(events) == 0:
            return
            
        debug_seq_dir = os.path.join(self.debug_dir, f"composition_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)
        
        stage_dir = os.path.join(debug_seq_dir, f"{stage_name}_events")
        os.makedirs(stage_dir, exist_ok=True)
        
        resolution_scales = [0.5, 1, 2, 4]
        resolution = (self.config['data']['resolution_w'], self.config['data']['resolution_h'])

        bg_color = self.visualization_background
        if bg_color == 'white':
            bg_value = 255
            on_color = (0, 0, 139)
            off_color = (139, 0, 0)
        else:
            bg_value = 0
            on_color = (0, 0, 255)
            off_color = (255, 0, 0)

        for scale in resolution_scales:
            scale_dir = os.path.join(stage_dir, f"temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)

            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            duration_ms = (t_max - t_min) / 1000.0

            base_window_ms = 10.0
            window_duration_ms = base_window_ms / scale
            window_duration_us = window_duration_ms * 1000

            num_frames = max(10, int(duration_ms / window_duration_ms))
            frame_step = (t_max - t_min) / num_frames if num_frames > 1 else 0

            for frame_idx in range(min(num_frames, 30)):
                frame_start = t_min + frame_idx * frame_step
                frame_end = frame_start + window_duration_us

                mask = (events[:, 2] >= frame_start) & (events[:, 2] < frame_end)
                frame_events = events[mask]

                frame = np.full((resolution[1], resolution[0], 3), bg_value, dtype=np.uint8)

                if len(frame_events) > 0:
                    for event in frame_events:
                        x, y, t, p = event
                        x, y = int(x), int(y)

                        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                            color = on_color if p > 0 else off_color
                            frame[y, x] = color

                import cv2
                frame_path = os.path.join(scale_dir, f"frame_{frame_idx:03d}.png")
                cv2.imwrite(frame_path, frame)
        
        metadata_path = os.path.join(stage_dir, "metadata.txt")
        with open(metadata_path, 'w') as f:
            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            duration_ms = (t_max - t_min) / 1000.0
            pos_events = np.sum(events[:, 3] > 0)
            neg_events = np.sum(events[:, 3] <= 0)
            
            f.write(f"{title} Visualization Metadata\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Event count: {len(events):,}\n")
            f.write(f"Duration: {duration_ms:.1f}ms\n")
            if duration_ms > 0:
                f.write(f"Event rate: {len(events) / (duration_ms / 1000):.1f} events/s\n")
            f.write(f"Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), ")
            f.write(f"{neg_events} OFF ({neg_events/len(events)*100:.1f}%)\n")
            f.write(f"Time range: {t_min:.0f} - {t_max:.0f} μs\n")
    
    def _save_debug_visualization(self, events_dict: Dict[str, np.ndarray], 
                                sequence_id: int, metadata: Dict):
        debug_seq_dir = os.path.join(self.debug_dir, f"composition_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)
        
        for event_name, events in events_dict.items():
            if len(events) > 0:
                title = self._get_event_title(event_name)
                self._create_event_visualization(events, debug_seq_dir, event_name, title)
        
        self._save_enhanced_composition_metadata(debug_seq_dir, events_dict, metadata)
    
    def _create_event_visualization(self, events: np.ndarray, output_dir: str, 
                                  event_type: str, title: str):
        if len(events) == 0:
            return
        
        type_dir = os.path.join(output_dir, f"{event_type}_events")
        os.makedirs(type_dir, exist_ok=True)
        
        resolution_scales = [0.5, 1, 2, 4]
        resolution = (self.config['data']['resolution_w'], self.config['data']['resolution_h'])

        bg_color = self.visualization_background
        if bg_color == 'white':
            bg_value = 255
            on_color = (0, 0, 139)
            off_color = (139, 0, 0)
        else:
            bg_value = 0
            on_color = (0, 0, 255)
            off_color = (255, 0, 0)

        for scale in resolution_scales:
            scale_dir = os.path.join(type_dir, f"temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)

            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            duration_ms = (t_max - t_min) / 1000.0

            base_window_ms = 10.0
            window_duration_ms = base_window_ms / scale
            window_duration_us = window_duration_ms * 1000

            num_frames = max(10, int(duration_ms / window_duration_ms))
            frame_step = (t_max - t_min) / num_frames if num_frames > 1 else 0

            for frame_idx in range(min(num_frames, 30)):
                frame_start = t_min + frame_idx * frame_step
                frame_end = frame_start + window_duration_us

                mask = (events[:, 2] >= frame_start) & (events[:, 2] < frame_end)
                frame_events = events[mask]

                frame = np.full((resolution[1], resolution[0], 3), bg_value, dtype=np.uint8)
                
                if len(frame_events) > 0:
                    for event in frame_events:
                        x, y, t, p = event
                        x, y = int(x), int(y)

                        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                            color = on_color if p > 0 else off_color
                            frame[y, x] = color
                
                import cv2
                frame_path = os.path.join(scale_dir, f"frame_{frame_idx:03d}.png")
                cv2.imwrite(frame_path, frame)
    
    def _get_event_title(self, event_name: str) -> str:
        title_map = {
            "01_background_raw": "Background Events (DSEC)",
            "02_light_source_raw": "Light Source Events (DVS)", 
            "03_flare_raw": "Flare Events (DVS)",
            "04_background_with_light": "Stage 1: Background + Light Source",
            "05_full_scene": "Stage 2: Full Scene (BG+Light+Flare)"
        }
        return title_map.get(event_name, event_name.replace("_", " ").title())
    
    def _save_enhanced_composition_metadata(self, output_dir: str, events_dict: Dict[str, np.ndarray], metadata: Dict):
        metadata_path = os.path.join(output_dir, "composition_metadata.txt")
        
        with open(metadata_path, 'w') as f:
            f.write("Event Composition Metadata (Three-Source Mode)\n")
            f.write("===============================================\n\n")
            
            f.write(f"Source Files:\n")
            f.write(f"  Flare: {metadata.get('flare_file', 'N/A')}\n")
            f.write(f"  Light Source: {metadata.get('light_source_file', 'N/A')}\n\n")
            
            for event_name, events in events_dict.items():
                if len(events) > 0:
                    title = self._get_event_title(event_name)
                    t_min, t_max = events[:, 2].min(), events[:, 2].max()
                    duration_ms = (t_max - t_min) / 1000.0
                    pos_events = np.sum(events[:, 3] > 0)
                    neg_events = np.sum(events[:, 3] <= 0)
                    
                    f.write(f"{title}:\n")
                    f.write(f"  Count: {len(events):,}\n")
                    f.write(f"  Duration: {duration_ms:.1f}ms\n")
                    if duration_ms > 0:
                        f.write(f"  Event rate: {len(events) / (duration_ms / 1000):.1f} events/s\n")
                    f.write(f"  Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), ")
                    f.write(f"{neg_events} OFF ({neg_events/len(events)*100:.1f}%)\n\n")
    
    def compose_batch(self, max_sequences: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        。flarelight_source，。
        
        Args:
            max_sequences: ，None
            
        Returns:
            (bg_light_files, full_scene_files) 
        """
        flare_files = {os.path.basename(p): p for p in glob.glob(os.path.join(self.flare_events_dir, "*.h5"))}
        light_source_files = {os.path.basename(p): p for p in glob.glob(os.path.join(self.light_source_events_dir, "*.h5"))}
        
        if not flare_files or not light_source_files:
            print(f"❌ Flare or light source event files not found.")
            print(f"   Flare dir: {self.flare_events_dir}")
            print(f"   Light source dir: {self.light_source_events_dir}")
            return [], []
        
        flare_bases = {f.replace('flare_', ''): f for f in flare_files.keys()}
        light_source_bases = {f.replace('light_source_', ''): f for f in light_source_files.keys()}
        
        common_bases = sorted(list(set(flare_bases.keys()) & set(light_source_bases.keys())))
        
        if not common_bases:
            print("❌ No matching flare and light source files found.")
            return [], []
        
        if max_sequences is not None:
            common_bases = common_bases[:max_sequences]
        
        print(f"\n🚀 Found {len(common_bases)} matched flare/light-source sequences. Composing...")
        print(f"📝 Composition numbering: {self.composition_start_id} to {self.composition_start_id + len(common_bases) - 1}")
        
        bg_light_files_out = []
        full_scene_files_out = []
        start_time = time.time()
        
        for i, base_name in enumerate(common_bases):
            actual_composition_id = self.composition_start_id + i
            print(f"\n--- Composing sequence {i+1}/{len(common_bases)} ({base_name}) (ID: {actual_composition_id}) ---")
            
            flare_filename = flare_bases[base_name]
            light_source_filename = light_source_bases[base_name]
            
            flare_path = flare_files[flare_filename]
            light_path = light_source_files[light_source_filename]
            
            try:
                bg_light_file, full_scene_file = self.compose_single_sequence(flare_path, light_path, i)
                bg_light_files_out.append(bg_light_file)
                full_scene_files_out.append(full_scene_file)
            except Exception as e:
                print(f"❌ Error composing sequence for {base_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        success_rate = len(bg_light_files_out) / len(common_bases) * 100
        
        print(f"\n✅ Event composition complete:")
        print(f"  Processed: {len(bg_light_files_out)}/{len(common_bases)} sequences ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average: {total_time/len(common_bases):.1f}s per sequence")
        for method_name, paths in self.output_dirs.items():
            print(f"  {method_name} method outputs:")
            print(f"    - Stage 1 (bg+light): {paths['stage1']}")
            print(f"    - Stage 2 (full scene): {paths['stage2']}")
        
        return bg_light_files_out, full_scene_files_out


def test_event_composer():
    import yaml
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['debug_mode'] = True
    
    composer = EventComposer(config)
    
    bg_files, merge_files = composer.compose_batch(max_sequences=3)
    
    print(f"Test complete! Generated {len(bg_files)} background files and {len(merge_files)} merge files.")
    return bg_files, merge_files


if __name__ == "__main__":
    test_event_composer()