"""
Independent Flare Event Generator for EventMamba-FX
===================================================

Step 1: Generate pure flare events using DVS simulator.
Output: Standard DVS H5 format in output/data/flare_events/

Key Features:
- Pure flare event generation without background mixing
- Standard DVS H5 format: /events/x, /events/y, /events/t, /events/p
- No feature extraction or post-processing
- Debug visualization support
"""

import os
import time
import random
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional

from src.dvs_flare_integration import create_flare_event_generator
from src.event_visualization_utils import EventVisualizer


class FlareEventGenerator:
    """
     - Step 1
    ，DVSH5
    """
    
    def __init__(self, config: Dict):
        """

        Args:
            config: 
        """
        self.config = config

        test_mode = config.get('test_mode', False)
        generation_config = config['generation']

        if test_mode:
            output_paths = generation_config['test_output_paths']
            debug_paths = generation_config.get('test_debug_paths', generation_config['debug_paths'])
        else:
            output_paths = generation_config['output_paths']
            debug_paths = generation_config['debug_paths']

        self.flare_output_dir = output_paths['flare_events']
        self.light_source_output_dir = output_paths['light_source_events']  # 🆕
        os.makedirs(self.flare_output_dir, exist_ok=True)
        os.makedirs(self.light_source_output_dir, exist_ok=True)  # 🆕

        self.debug_mode = config.get('debug_mode', False)
        if self.debug_mode:
            self.flare_debug_dir = debug_paths['flare_generation']
            self.light_source_debug_dir = debug_paths['light_source_generation']  # 🆕
            os.makedirs(self.flare_debug_dir, exist_ok=True)
            os.makedirs(self.light_source_debug_dir, exist_ok=True)  # 🆕

            bg_color = config.get('visualization_background', 'black')
            self.visualization_background = bg_color

            resolution = (config['data']['resolution_w'], config['data']['resolution_h'])
            self.flare_visualizer = EventVisualizer(self.flare_debug_dir, resolution, background_color=bg_color)
            self.light_source_visualizer = EventVisualizer(self.light_source_debug_dir, resolution, background_color=bg_color)  # 🆕
            print(f"🎯 FlareEventGenerator Debug Mode:")
            print(f"  Flare debug: {self.flare_debug_dir}")
            print(f"  Light source debug: {self.light_source_debug_dir}")  # 🆕
        else:
            self.visualization_background = 'black'
        
        self.dvs_generator = create_flare_event_generator(config)
        
        flare_config = config['data']['flare_synthesis']
        self.duration_range = flare_config['duration_range']
        
        self.sequence_start_id = self._get_next_sequence_id()
        
        mode_label = "TEST" if test_mode else "TRAIN/VAL"
        print(f"🚀 FlareEventGenerator initialized (Synced Generation - {mode_label} Mode):")
        print(f"  Flare events: {self.flare_output_dir}")
        print(f"  Light source events: {self.light_source_output_dir}")  # 🆕
        print(f"  Duration range: {self.duration_range[0]*1000:.0f}-{self.duration_range[1]*1000:.0f}ms")
        print(f"  Sequence start ID: {self.sequence_start_id} (continuing from existing files)")  # 🆕
        print(f"  Debug mode: {self.debug_mode}")
    
    def _get_next_sequence_id(self) -> int:
        """
        ID，
        
        Returns:
            ID
        """
        import glob
        
        flare_files = glob.glob(os.path.join(self.flare_output_dir, "*.h5"))
        light_source_files = glob.glob(os.path.join(self.light_source_output_dir, "*.h5"))
        
        max_existing = max(len(flare_files), len(light_source_files))
        
        if max_existing > 0:
            print(f"📁 Found existing files: {len(flare_files)} flare + {len(light_source_files)} light source")
            print(f"🔢 Starting sequence ID from: {max_existing}")
        
        return max_existing
    
    def generate_single_flare_sequence(self, sequence_id: int) -> Tuple[Optional[str], Optional[str]]:
        """
        🔄 : 
        
        Args:
            sequence_id: ID（）
            
        Returns:
            Tuple of (flare_output_path, light_source_output_path)
        """
        start_time = time.time()
        
        try:
            print(f"🎬 Generating synced sequence {sequence_id}...")
            flare_events, light_source_events, metadata, flare_frames, light_source_frames = self.dvs_generator.generate_synced_events(cleanup=True)
            
            if flare_events is None or light_source_events is None:
                print(f"❌ Failed to generate synced events for sequence {sequence_id}")
                return None, None
            
            if len(flare_events) == 0 and len(light_source_events) == 0:
                print(f"⚠️  Warning: No events generated for sequence {sequence_id}")
                return None, None

            actual_sequence_id = self.sequence_start_id + sequence_id
            base_filename = f"sequence_{actual_sequence_id:05d}.h5"
            
            flare_filename = f"flare_{base_filename}"
            flare_output_path = os.path.join(self.flare_output_dir, flare_filename)
            self._save_events_dvs_format(flare_events, flare_output_path, metadata)
            
            light_source_filename = f"light_source_{base_filename}"
            light_source_output_path = os.path.join(self.light_source_output_dir, light_source_filename)
            self._save_events_dvs_format(light_source_events, light_source_output_path, metadata)

            generation_time = time.time() - start_time
            duration_ms = metadata.get('duration_sec', 0) * 1000
            
            print(f"✅ Generated synced sequence {sequence_id}:")
            print(f"  Flare events: {len(flare_events):,}")
            print(f"  Light source events: {len(light_source_events):,}")
            print(f"  Duration: {duration_ms:.1f}ms")
            print(f"  Time: {generation_time:.2f}s")
            print(f"  Files: {flare_filename} + {light_source_filename}")
            
            if self.debug_mode:
                self._save_debug_visualization(flare_events, flare_frames, sequence_id, metadata, "flare")
                self._save_debug_visualization(light_source_events, light_source_frames, sequence_id, metadata, "light_source")

            del flare_frames, light_source_frames
            import gc
            gc.collect()

            return flare_output_path, light_source_output_path
            
        except Exception as e:
            print(f"❌ Error generating synced sequence {sequence_id}: {e}")
            return None, None
    
    def _save_events_dvs_format(self, events: np.ndarray, output_path: str, metadata: Dict):
        """
        DVSH5
        
        Args:
            events:  [N, 4]  [t, x, y, p] (DVS)
            output_path: 
            metadata: 
        """
        if len(events) > 0:
            events_normalized = events.copy()
            t_min = events_normalized[:, 0].min()
            events_normalized[:, 0] = events_normalized[:, 0] - t_min
            
            print(f"    Flare timing: starts at 0ms (duration: {metadata.get('duration_sec', 0)*1000:.1f}ms)")
        else:
            events_normalized = events
        
        with h5py.File(output_path, 'w') as f:
            events_group = f.create_group('events')
            
            events_group.create_dataset('t', data=events_normalized[:, 0].astype(np.int64), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=events_normalized[:, 1].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=events_normalized[:, 2].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=events_normalized[:, 3].astype(np.int8), 
                                      compression='gzip', compression_opts=9)
            
            events_group.attrs['num_events'] = len(events)
            events_group.attrs['duration_sec'] = metadata.get('duration_sec', 0)
            events_group.attrs['frequency_hz'] = metadata.get('frequency_hz', 0)
            events_group.attrs['resolution_height'] = self.config['data']['resolution_h']
            events_group.attrs['resolution_width'] = self.config['data']['resolution_w']
            events_group.attrs['simulator'] = 'dvs_voltmeter'
            events_group.attrs['generation_time'] = time.time()

            if 'flare_image_path' in metadata:
                events_group.attrs['flare_image_path'] = str(metadata['flare_image_path'])
            if 'light_source_image_path' in metadata:
                events_group.attrs['light_source_image_path'] = str(metadata['light_source_image_path'])
    
    def _save_debug_visualization(self, events: np.ndarray, frames: List[np.ndarray], 
                                sequence_id: int, metadata: Dict, event_type: str):
        """
        🔄 : debug ()
        
        Args:
            events:  [N, 4] DVS
            frames: 
            sequence_id: ID
            metadata: 
            event_type:  ("flare"  "light_source")
        """
        if event_type == "flare":
            base_debug_dir = self.flare_debug_dir
            sequence_name = f"flare_sequence_{sequence_id:03d}"
        elif event_type == "light_source":
            base_debug_dir = self.light_source_debug_dir
            sequence_name = f"light_source_sequence_{sequence_id:03d}"
        else:
            print(f"⚠️  Unknown event type: {event_type}")
            return
        
        debug_seq_dir = os.path.join(base_debug_dir, sequence_name)
        os.makedirs(debug_seq_dir, exist_ok=True)
        
        if len(events) > 0:
            vis_events = np.zeros_like(events)
            vis_events[:, 0] = events[:, 1]  # x
            vis_events[:, 1] = events[:, 2]  # y
            vis_events[:, 2] = events[:, 0]  # t
            vis_events[:, 3] = events[:, 3]  # p
            
            if len(vis_events) > 0:
                t_min = vis_events[:, 2].min()
                vis_events[:, 2] = vis_events[:, 2] - t_min
            
            self._create_event_visualization(vis_events, debug_seq_dir, metadata, event_type)
        
        if frames:
            self._save_source_frames(frames, debug_seq_dir, event_type)
        
        self._save_sequence_metadata(debug_seq_dir, events, metadata, event_type)
    
    def _create_event_visualization(self, events: np.ndarray, output_dir: str, metadata: Dict, event_type: str):
        if len(events) == 0:
            return
            
        original_fps = metadata.get('fps', 100)
        duration_sec = metadata.get('duration_sec', 0.1)
        total_frames = metadata.get('total_frames', int(original_fps * duration_sec))
        
        print(f"    Debug vis: {total_frames} frames, {original_fps}fps, {duration_sec*1000:.1f}ms")
        
        resolution_scales = [0.5]
        base_frame_interval_us = 1e6 / original_fps
        
        for scale in resolution_scales:
            scale_dir = os.path.join(output_dir, f"events_temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)
            
            accumulation_window_us = base_frame_interval_us / scale
            
            if scale == 1.0:
                vis_frames = max(1, total_frames - 1)
            else:
                vis_frames = max(1, int((total_frames - 1) / scale))
            
            print(f"      {scale}x: {vis_frames} frames, window={accumulation_window_us/1000:.1f}ms")
            
            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            time_step = (t_max - t_min) / vis_frames if vis_frames > 1 else 0
            
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

            for frame_idx in range(vis_frames):
                frame_center = t_min + frame_idx * time_step
                frame_start = frame_center
                frame_end = frame_start + accumulation_window_us

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
    
    def _save_source_frames(self, frames: List[np.ndarray], output_dir: str, event_type: str):
        frames_dir = os.path.join(output_dir, f"source_{event_type}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        import cv2
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
    
    def _save_sequence_metadata(self, output_dir: str, events: np.ndarray, metadata: Dict, event_type: str):
        metadata_path = os.path.join(output_dir, "metadata.txt")
        
        with open(metadata_path, 'w') as f:
            if event_type == "flare":
                f.write("Flare Event Generation Metadata\n")
                f.write("===============================\n\n")
            elif event_type == "light_source":
                f.write("Light Source Event Generation Metadata\n")
                f.write("======================================\n\n")
            else:
                f.write("Event Generation Metadata\n")
                f.write("=========================\n\n")
            
            f.write(f"Events: {len(events):,}\n")
            f.write(f"Duration: {metadata.get('duration_sec', 0)*1000:.1f}ms\n")
            f.write(f"Frequency: {metadata.get('frequency_hz', 0):.1f}Hz\n")
            f.write(f"FPS: {metadata.get('fps', 0):.0f}\n")
            f.write(f"Samples per cycle: {metadata.get('samples_per_cycle', 0):.1f}\n")
            
            if len(events) > 0:
                t_min, t_max = events[:, 0].min(), events[:, 0].max()
                pos_events = np.sum(events[:, 3] > 0)
                neg_events = np.sum(events[:, 3] <= 0)
                
                f.write(f"Time range: {t_min:.0f} - {t_max:.0f} μs\n")
                f.write(f"Event rate: {len(events) / (metadata.get('duration_sec', 1)):.1f} events/s\n")
                f.write(f"Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), ")
                f.write(f"{neg_events} OFF ({neg_events/len(events)*100:.1f}%)\n")
    
    def generate_batch(self, num_sequences: int) -> Tuple[List[str], List[str]]:
        """
        🔄 : 
        
        Args:
            num_sequences: 
            
        Returns:
            Tuple of (flare_file_paths, light_source_file_paths)
        """
        print(f"\n🚀 Generating {num_sequences} synced flare/light-source event sequences...")
        print(f"📝 Sequence numbering: {self.sequence_start_id} to {self.sequence_start_id + num_sequences - 1}")
        
        flare_files = []
        light_source_files = []
        start_time = time.time()
        
        for i in range(num_sequences):
            actual_id = self.sequence_start_id + i
            print(f"\n--- Generating synced sequence {i+1}/{num_sequences} (ID: {actual_id}) ---")
            
            flare_path, light_source_path = self.generate_single_flare_sequence(i)
            if flare_path and light_source_path:
                flare_files.append(flare_path)
                light_source_files.append(light_source_path)
            elif flare_path or light_source_path:
                print(f"⚠️  Sequence {i} partially failed - discarded for sync consistency")
        
        total_time = time.time() - start_time
        success_rate = len(flare_files) / num_sequences * 100
        
        print(f"\n✅ Synced event generation complete:")
        print(f"  Generated: {len(flare_files)}/{num_sequences} synced pairs ({success_rate:.1f}%)")
        print(f"  Flare files: {len(flare_files)}")
        print(f"  Light source files: {len(light_source_files)}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average: {total_time/num_sequences:.1f}s per synced pair")
        print(f"  Flare output: {self.flare_output_dir}")
        print(f"  Light source output: {self.light_source_output_dir}")
        
        return flare_files, light_source_files


def test_flare_generator():
    import yaml
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['debug_mode'] = True
    
    generator = FlareEventGenerator(config)
    
    files = generator.generate_batch(3)
    
    print(f"Test complete! Generated {len(files)} files.")
    return files


if __name__ == "__main__":
    test_flare_generator()