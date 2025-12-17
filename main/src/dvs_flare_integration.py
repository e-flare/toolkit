"""
Event Simulator Integration Module for EventMamba-FX

This module provides a unified interface for multiple event simulators to generate
realistic flare events that can be combined with background DSEC events.

Supported simulators:
- DVS-Voltmeter: Traditional physics-based simulator (high-framerate requirement)
- V2CE: Deep learning-based video-to-events simulator (low-framerate, high quality)

Key features:
- Generate flare video sequences and convert to events using selected simulator
- Handle temporary file management for simulator pipeline  
- Output events in format compatible with EventMamba-FX training
- Benchmark timing for each processing step
- Debug visualization with simulator-specific suffixes
"""

import os
import sys
import time
import tempfile
import shutil
import subprocess
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. V2CE simulator will not work.")

# Add simulator path to import DVS-Voltmeter modules
simulator_path = "/path/to/data/event_flick_flare/main/simulator/DVS-Voltmeter-main"
if simulator_path not in sys.path:
    sys.path.append(simulator_path)

# Try different import paths
try:
    from src.flare_synthesis import FlareFlickeringSynthesizer
except ImportError:
    try:
        from flare_synthesis import FlareFlickeringSynthesizer
    except ImportError:
        # Add current directory to path
        import sys
        sys.path.append(os.path.dirname(__file__))
        from flare_synthesis import FlareFlickeringSynthesizer


class V2CEFlareEventGenerator:
    """Generates flare events using V2CE (Video to Continuous Events) simulator."""
    
    def __init__(self, config: Dict):
        """Initialize the V2CE flare event generator.
        
        Args:
            config: Configuration dictionary
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for V2CE simulator but not available")
        
        self.config = config
        self.flare_synthesizer = FlareFlickeringSynthesizer(config)
        
        self.v2ce_config = config['data']['event_simulator']['v2ce']
        self.toolbox_path = self.v2ce_config['toolbox_path']
        self.model_path = self.v2ce_config['model_path']
        
        if not os.path.exists(self.toolbox_path):
            raise FileNotFoundError(f"V2CE toolbox not found at: {self.toolbox_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"V2CE model not found at: {self.model_path}")
        
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. V2CE may run slowly on CPU.")
        
        if self.toolbox_path not in sys.path:
            sys.path.append(self.toolbox_path)
        
        self.v2ce_resolution = (
            self.v2ce_config['width'],
            self.v2ce_config['height']
        )
        
        self.target_resolution = (
            config['data']['resolution_w'],  # 640
            config['data']['resolution_h']   # 480
        )
        print(f"V2CE input resolution: {self.v2ce_resolution[0]}x{self.v2ce_resolution[1]}")
        print(f"Target output resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
        
        # 🚨 DISABLE: Flare sequence debug disabled (using unified debug_epoch_000 instead)
        self.debug_mode = False  # config.get('debug_mode', False)
        self.debug_save_dir = config.get('debug_output_dir', './output/debug_visualizations_v2ce')
        self.debug_counter = 0
        if self.debug_mode:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            print(f"🚨 DEBUG MODE (V2CE): Will save flare sequences to {self.debug_save_dir}")
        
        self._v2ce_model = None
    
    def _get_v2ce_model(self):
        if self._v2ce_model is None:
            try:
                from v2ce import get_trained_mode
                self._v2ce_model = get_trained_mode(self.model_path)
                print("✅ V2CE model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load V2CE model: {e}")
        return self._v2ce_model
    
    def generate_flare_events(self, temp_dir: Optional[str] = None, 
                            cleanup: bool = True) -> Tuple[np.ndarray, Dict, List[np.ndarray]]:
        """Generate flare events using V2CE pipeline.
        
        Args:
            temp_dir: Optional temporary directory, creates one if None
            cleanup: Whether to cleanup temporary files
            
        Returns:
            Tuple of (events_array, timing_info, video_frames)
            Events format: [timestamp_us, x, y, polarity]
        """
        timing_info = {}
        total_start = time.time()
        
        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="v2ce_flare_events_")
            cleanup_temp = cleanup
        else:
            cleanup_temp = False
        
        try:
            # Step 1: Generate flickering flare video
            # print("Step 1: Generating flickering flare video for V2CE...")
            step1_start = time.time()
            
            video_frames, flare_metadata = self.flare_synthesizer.create_flare_event_sequence(
                target_resolution=self.v2ce_resolution
            )
            
            dynamic_fps = self._calculate_dynamic_fps(flare_metadata)
            # print(f"  Dynamic FPS for V2CE: {dynamic_fps} (flicker: {flare_metadata.get('frequency_hz', 'N/A')} Hz)")
            
            timing_info['flare_synthesis_sec'] = time.time() - step1_start
            # print(f"  Generated {len(video_frames)} frames in {timing_info['flare_synthesis_sec']:.3f}s")
            
            # Step 2: Save video frames as image sequence
            # print("Step 2: Saving image sequence for V2CE...")
            step2_start = time.time()
            
            image_dir = self._save_video_as_images(video_frames, temp_dir, flare_metadata)
            
            timing_info['image_saving_sec'] = time.time() - step2_start
            # print(f"  Saved {len(video_frames)} images in {timing_info['image_saving_sec']:.3f}s")
            
            # Step 3: Run V2CE inference
            # print("Step 3: Running V2CE inference...")
            step3_start = time.time()
            
            events_array = self._run_v2ce_inference(image_dir, dynamic_fps)
            
            timing_info['v2ce_inference_sec'] = time.time() - step3_start
            # print(f"  Generated {len(events_array)} events in {timing_info['v2ce_inference_sec']:.3f}s")
            
            # print("Step 4: Post-processing events...")
            step4_start = time.time()
            
            events_array = self._post_process_events(events_array)
            
            timing_info['post_processing_sec'] = time.time() - step4_start
            # print(f"  Post-processed events in {timing_info['post_processing_sec']:.3f}s")
            
            # Step 5: Debug visualization with V2CE suffix
            if self.debug_mode:
                # print("Step 5: Saving debug visualizations (V2CE)...")
                step5_start = time.time()
                
                self._save_debug_visualizations_v2ce(video_frames, events_array, flare_metadata)
                
                timing_info['debug_visualization_sec'] = time.time() - step5_start
                # print(f"  Saved debug visualizations in {timing_info['debug_visualization_sec']:.3f}s")
            
            # Combine metadata
            timing_info.update(flare_metadata)
            timing_info['total_pipeline_sec'] = time.time() - total_start
            timing_info['simulator_type'] = 'V2CE'
            
            return events_array, timing_info, video_frames
            
        finally:
            # Cleanup temporary directory if requested
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _save_video_as_images(self, video_frames: List[np.ndarray], 
                            temp_dir: str, metadata: Dict) -> str:
        """Save video frames as individual images for V2CE input.
        
        Args:
            video_frames: List of RGB video frames
            temp_dir: Temporary directory for storage
            metadata: Video metadata
            
        Returns:
            Path to the image directory
        """
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        for i, frame in enumerate(video_frames):
            # V2CE expects grayscale images
            if len(frame.shape) == 3:
                # Convert RGB to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = frame
            
            # Save as PNG (V2CE compatible format)
            frame_path = os.path.join(image_dir, f"{i:06d}.png")
            cv2.imwrite(frame_path, gray_frame)
        
        return image_dir
    
    def _run_v2ce_inference(self, image_dir: str, dynamic_fps: int = None) -> np.ndarray:
        """Run V2CE inference on the image sequence.
        
        Args:
            image_dir: Directory containing input images
            dynamic_fps: Dynamic frame rate for event generation
            
        Returns:
            Events array in format [timestamp_us, x, y, polarity]
        """
        # Use default fps if not provided
        if dynamic_fps is None:
            dynamic_fps = self.v2ce_config['base_fps']
        try:
            model = self._get_v2ce_model()
            
            from v2ce import video_to_voxels
            from scripts.LDATI import sample_voxel_statistical
            from functools import partial
            
            image_paths = sorted([
                os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                if f.endswith('.png')
            ])
            
            if not image_paths:
                raise ValueError(f"No PNG images found in {image_dir}")
            
            v2ce_config = self.v2ce_config
            
            pred_voxel = video_to_voxels(
                model=model,
                image_paths=image_paths,
                vidcap=None,
                infer_type=v2ce_config['infer_type'],
                seq_len=v2ce_config['seq_len'],
                width=v2ce_config['width'],
                height=v2ce_config['height'],
                batch_size=v2ce_config['batch_size']
            )
            
            L, _, _, H, W = pred_voxel.shape
            stage2_input = pred_voxel.reshape(L, 2, 10, H, W)
            stage2_input = torch.from_numpy(stage2_input).cuda()
            
            ldati = partial(
                sample_voxel_statistical, 
                fps=dynamic_fps,
                bidirectional=False, 
                additional_events_strategy='slope'
            )
            
            stage2_batch_size = 24
            event_stream_per_frame = []
            
            for i in range(0, stage2_input.shape[0], stage2_batch_size):
                batch_events = ldati(stage2_input[i:i+stage2_batch_size])
                event_stream_per_frame.extend(batch_events)
            
            event_stream = []
            for i in range(L):
                frame_events = event_stream_per_frame[i]
                frame_time_offset = int(i * 1 / dynamic_fps * 1e6)
                frame_events['timestamp'] += frame_time_offset
                event_stream.append(frame_events)
            
            if event_stream:
                combined_events = np.concatenate(event_stream)
                
                if len(combined_events) > 0:
                    events_array = np.column_stack([
                        combined_events['timestamp'].astype(np.int64),
                        combined_events['x'].astype(np.int16),
                        combined_events['y'].astype(np.int16),
                        combined_events['polarity'].astype(np.int8)
                    ])
                    
                    print(f"  V2CE generated {len(events_array)} events successfully")
                else:
                    print(f"  V2CE generated empty event stream")
                    events_array = np.empty((0, 4))
            else:
                print(f"  V2CE event_stream is empty")
                events_array = np.empty((0, 4))
            
            return events_array
            
        except Exception as e:
            print(f"V2CE inference failed: {e}")
            return np.empty((0, 4))
    
    def _post_process_events(self, events_array: np.ndarray) -> np.ndarray:
        """Post-process V2CE events (coordinate transformation, filtering).
        
        Args:
            events_array: Raw V2CE events [timestamp_us, x, y, polarity]
            
        Returns:
            Processed events array
        """
        if len(events_array) == 0:
            return events_array
        
        v2ce_w, v2ce_h = self.v2ce_resolution
        target_w, target_h = self.target_resolution
        
        scale_x = target_w / v2ce_w
        scale_y = target_h / v2ce_h
        
        events_scaled = events_array.copy().astype(np.float64)
        events_scaled[:, 1] = events_array[:, 1] * scale_x
        events_scaled[:, 2] = events_array[:, 2] * scale_y
        
        valid_mask = (
            (events_scaled[:, 1] >= 0) & (events_scaled[:, 1] < target_w) &
            (events_scaled[:, 2] >= 0) & (events_scaled[:, 2] < target_h)
        )
        events_scaled = events_scaled[valid_mask]
        
        events_scaled[:, 1] = np.round(events_scaled[:, 1]).astype(np.int16)
        events_scaled[:, 2] = np.round(events_scaled[:, 2]).astype(np.int16)
        events_scaled[:, 0] = events_scaled[:, 0].astype(np.int64)  # timestamp
        events_scaled[:, 3] = events_scaled[:, 3].astype(np.int8)   # polarity
        
        # print(f"  Coordinate transform: {v2ce_w}x{v2ce_h} → {target_w}x{target_h}")
        # print(f"  Events after filtering: {len(events_scaled)}/{len(events_array)}")
        
        return events_scaled
    
    def _save_debug_visualizations_v2ce(self, video_frames: List[np.ndarray], 
                                       events_array: np.ndarray, metadata: Dict):
        """Save debug visualizations with V2CE suffix and specific info.
        
        Args:
            video_frames: List of RGB video frames (V2CE input)
            events_array: Events array [timestamp_us, x, y, polarity] 
            metadata: V2CE metadata
        """
        # Create unique subdirectory with V2CE suffix
        sequence_id = f"flare_seq_v2ce_{self.debug_counter:03d}"
        sequence_debug_dir = os.path.join(self.debug_save_dir, sequence_id)
        os.makedirs(sequence_debug_dir, exist_ok=True)
        self.debug_counter += 1
        
        # print(f"  Saving V2CE debug to: {sequence_debug_dir}")
        
        # 1. Save original flare image sequence (V2CE input format)
        frames_dir = os.path.join(sequence_debug_dir, "v2ce_input_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(video_frames):
            # Save both RGB and grayscale versions
            frame_rgb_path = os.path.join(frames_dir, f"frame_{i:03d}_rgb.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_rgb_path, frame_bgr)
            
            # Grayscale version (actual V2CE input)
            frame_gray_path = os.path.join(frames_dir, f"frame_{i:03d}_gray.png")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(frame_gray_path, gray_frame)
        
        # 2. Save V2CE-specific metadata
        metadata_path = os.path.join(sequence_debug_dir, "v2ce_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"V2CE Flare Sequence Debug Info\n")
            f.write(f"===============================\n")
            f.write(f"Simulator: V2CE (Video to Continuous Events)\n")
            f.write(f"Input frames: {len(video_frames)}\n")
            f.write(f"Input resolution: {self.v2ce_resolution[0]}x{self.v2ce_resolution[1]}\n")
            f.write(f"Output resolution: {self.target_resolution[0]}x{self.target_resolution[1]}\n")
            f.write(f"Generated events: {len(events_array)}\n")
            
            if len(events_array) > 0:
                f.write(f"Time range: {events_array[0, 0]} - {events_array[-1, 0]} μs\n")
                f.write(f"Duration: {(events_array[-1, 0] - events_array[0, 0]) / 1000:.1f} ms\n")
                f.write(f"Event rate: {len(events_array) / ((events_array[-1, 0] - events_array[0, 0]) / 1e6):.1f} events/s\n")
                pos_events = np.sum(events_array[:, 3] == 1)
                neg_events = np.sum(events_array[:, 3] == 0) + np.sum(events_array[:, 3] == -1)
                f.write(f"Polarity: {pos_events} ON ({pos_events/len(events_array)*100:.1f}%), ")
                f.write(f"{neg_events} OFF ({neg_events/len(events_array)*100:.1f}%)\n")
                
                x_range = f"[{events_array[:, 1].min()}, {events_array[:, 1].max()}]"
                y_range = f"[{events_array[:, 2].min()}, {events_array[:, 2].max()}]"
                f.write(f"Spatial range: x={x_range}, y={y_range}\n")
            
            f.write(f"\nV2CE Configuration:\n")
            for key, value in self.v2ce_config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nOriginal Flare Metadata:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
        
        # 3. Create event visualizations matching DVS style (using target resolution)
        if len(events_array) > 0:
            vis_dir = os.path.join(sequence_debug_dir, "v2ce_event_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            if metadata and 'fps' in metadata:
                fps = metadata['fps']
            else:
                # 🚨 FIX: Use duration_range instead of deprecated duration_sec
                duration_range = self.config['data']['flare_synthesis'].get('duration_range', [0.05, 0.15])
                if isinstance(duration_range, list) and len(duration_range) == 2:
                    duration_sec = (duration_range[0] + duration_range[1]) / 2  # Use average
                else:
                    duration_sec = 0.1  # Fallback
                fps = len(video_frames) / duration_sec
            
            frame_duration_us = int(1e6 / fps)
            target_w, target_h = self.target_resolution
            
            subdivision_strategies = [0.5, 1, 2, 4]
            
            # print(f"  Creating V2CE multi-resolution event visualizations: {subdivision_strategies}x")
            # print(f"  Frame duration: {frame_duration_us}μs, FPS: {fps}")
            # print(f"  Event data format: [timestamp_us, x, y, polarity]")
            
            # Process each subdivision strategy
            for strategy in subdivision_strategies:
                strategy_dir = os.path.join(vis_dir, f"temporal_{strategy}x")
                os.makedirs(strategy_dir, exist_ok=True)
                
                if strategy == 0.5:
                    # 0.5x: Two frames share one event window (longer duration)
                    subdivision_duration_us = frame_duration_us * 2
                    # print(f"    {strategy}x: {subdivision_duration_us}μs per visualization (2 frames → 1 event window)")
                    
                    # Process pairs of frames
                    for frame_idx in range(0, len(video_frames), 2):
                        frame_start_time = events_array[0, 0] + frame_idx * frame_duration_us
                        frame_end_time = frame_start_time + subdivision_duration_us
                        
                        # Filter events in time window
                        mask = (events_array[:, 0] >= frame_start_time) & (events_array[:, 0] < frame_end_time)
                        frame_events = events_array[mask]
                        
                        # Use first frame as background
                        background = cv2.resize(video_frames[frame_idx], (target_w, target_h))
                        vis_canvas = (background * 0.3).astype(np.uint8)
                        if len(vis_canvas.shape) == 3:
                            pass  # Already RGB
                        else:
                            vis_canvas = cv2.cvtColor(vis_canvas, cv2.COLOR_GRAY2RGB)
                        
                        # Render events
                        event_count = 0
                        for event in frame_events:
                            t, x, y, p = event
                            x, y = int(x), int(y)
                            if 0 <= x < target_w and 0 <= y < target_h:
                                if p > 0:
                                    vis_canvas[y, x] = [255, 0, 0]  # Red
                                else:
                                    vis_canvas[y, x] = [0, 0, 255]  # Blue
                                event_count += 1
                        
                        vis_filename = f"v2ce_events_pair_{frame_idx:03d}_{event_count}events.png"
                        vis_path = os.path.join(strategy_dir, vis_filename)
                        vis_bgr = cv2.cvtColor(vis_canvas, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(vis_path, vis_bgr)
                
                elif strategy == 1:
                    # 1x: One frame per event window (standard)
                    subdivision_duration_us = frame_duration_us
                    # print(f"    {strategy}x: {subdivision_duration_us}μs per visualization (1 frame → 1 event window)")
                    
                    for frame_idx in range(len(video_frames)):
                        frame_start_time = events_array[0, 0] + frame_idx * frame_duration_us
                        frame_end_time = frame_start_time + subdivision_duration_us
                        
                        mask = (events_array[:, 0] >= frame_start_time) & (events_array[:, 0] < frame_end_time)
                        frame_events = events_array[mask]
                        
                        background = cv2.resize(video_frames[frame_idx], (target_w, target_h))
                        vis_canvas = (background * 0.3).astype(np.uint8)
                        if len(vis_canvas.shape) != 3:
                            vis_canvas = cv2.cvtColor(vis_canvas, cv2.COLOR_GRAY2RGB)
                        
                        event_count = 0
                        for event in frame_events:
                            t, x, y, p = event
                            x, y = int(x), int(y)
                            if 0 <= x < target_w and 0 <= y < target_h:
                                if p > 0:
                                    vis_canvas[y, x] = [255, 0, 0]
                                else:
                                    vis_canvas[y, x] = [0, 0, 255]
                                event_count += 1
                        
                        vis_filename = f"v2ce_events_frame_{frame_idx:03d}_{event_count}events.png"
                        vis_path = os.path.join(strategy_dir, vis_filename)
                        vis_bgr = cv2.cvtColor(vis_canvas, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(vis_path, vis_bgr)
                
                elif strategy == 2:
                    # 2x: Two subdivisions per frame (higher temporal resolution)
                    subdivision_duration_us = frame_duration_us // 2
                    # print(f"    {strategy}x: {subdivision_duration_us}μs per visualization (2 subdivisions per frame)")
                    
                    for frame_idx in range(len(video_frames)):
                        for sub_idx in range(2):
                            frame_start_time = events_array[0, 0] + frame_idx * frame_duration_us + sub_idx * subdivision_duration_us
                            frame_end_time = frame_start_time + subdivision_duration_us
                            
                            mask = (events_array[:, 0] >= frame_start_time) & (events_array[:, 0] < frame_end_time)
                            frame_events = events_array[mask]
                            
                            background = cv2.resize(video_frames[frame_idx], (target_w, target_h))
                            vis_canvas = (background * 0.3).astype(np.uint8)
                            if len(vis_canvas.shape) != 3:
                                vis_canvas = cv2.cvtColor(vis_canvas, cv2.COLOR_GRAY2RGB)
                            
                            event_count = 0
                            for event in frame_events:
                                t, x, y, p = event
                                x, y = int(x), int(y)
                                if 0 <= x < target_w and 0 <= y < target_h:
                                    if p > 0:
                                        vis_canvas[y, x] = [255, 0, 0]
                                    else:
                                        vis_canvas[y, x] = [0, 0, 255]
                                    event_count += 1
                            
                            vis_filename = f"v2ce_events_frame_{frame_idx:03d}_sub_{sub_idx:02d}_{event_count}events.png"
                            vis_path = os.path.join(strategy_dir, vis_filename)
                            vis_bgr = cv2.cvtColor(vis_canvas, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(vis_path, vis_bgr)
                
                elif strategy == 4:
                    # 4x: Four subdivisions per frame (highest temporal resolution)
                    subdivision_duration_us = frame_duration_us // 4
                    # print(f"    {strategy}x: {subdivision_duration_us}μs per visualization (4 subdivisions per frame)")
                    
                    for frame_idx in range(len(video_frames)):
                        for sub_idx in range(4):
                            frame_start_time = events_array[0, 0] + frame_idx * frame_duration_us + sub_idx * subdivision_duration_us
                            frame_end_time = frame_start_time + subdivision_duration_us
                            
                            mask = (events_array[:, 0] >= frame_start_time) & (events_array[:, 0] < frame_end_time)
                            frame_events = events_array[mask]
                            
                            background = cv2.resize(video_frames[frame_idx], (target_w, target_h))
                            vis_canvas = (background * 0.3).astype(np.uint8)
                            if len(vis_canvas.shape) != 3:
                                vis_canvas = cv2.cvtColor(vis_canvas, cv2.COLOR_GRAY2RGB)
                            
                            event_count = 0
                            for event in frame_events:
                                t, x, y, p = event
                                x, y = int(x), int(y)
                                if 0 <= x < target_w and 0 <= y < target_h:
                                    if p > 0:
                                        vis_canvas[y, x] = [255, 0, 0]
                                    else:
                                        vis_canvas[y, x] = [0, 0, 255]
                                    event_count += 1
                            
                            vis_filename = f"v2ce_events_frame_{frame_idx:03d}_sub_{sub_idx:02d}_{event_count}events.png"
                            vis_path = os.path.join(strategy_dir, vis_filename)
                            vis_bgr = cv2.cvtColor(vis_canvas, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(vis_path, vis_bgr)
        
        # print(f"  Saved V2CE debug: {len(video_frames)} input frames + event visualizations")
    
    def _calculate_dynamic_fps(self, flare_metadata: Dict) -> int:
        """Calculate correct FPS for V2CE based on input video's actual framerate.
        
        ：V2CEfps
        ，flarefps
        
        Args:
            flare_metadata: Flare metadata containing video generation info
            
        Returns:
            FPS matching the input video's actual framerate
        """
        actual_video_fps = flare_metadata.get('fps', 1600)
        
        
        v2ce_config = self.v2ce_config
        max_supported_fps = v2ce_config.get('max_fps', 800)
        min_supported_fps = 30
        
        video_duration_sec = flare_metadata.get('duration_sec', 0.05)
        total_frames = flare_metadata.get('total_frames', 80)
        
        ideal_fps = (total_frames - 1) / video_duration_sec
        
        optimal_fps = max(min_supported_fps, min(ideal_fps, max_supported_fps))
        
        if ideal_fps > max_supported_fps:
            compression_ratio = ideal_fps / max_supported_fps
            # print(f"  ⚠️ Ideal fps {ideal_fps:.0f} > max supported {max_supported_fps}")
            # print(f"  Time will be expanded by {compression_ratio:.1f}x (longer than expected)")
        elif ideal_fps < min_supported_fps:
            expansion_ratio = min_supported_fps / ideal_fps
            # print(f"  ⚠️ Ideal fps {ideal_fps:.0f} < min supported {min_supported_fps}")
            # print(f"  Time will be compressed by {expansion_ratio:.1f}x (shorter than expected)")
        
        video_duration_sec = flare_metadata.get('duration_sec', 0.05)
        total_frames = flare_metadata.get('total_frames', 80)
        expected_v2ce_duration = total_frames / optimal_fps * 1000  # ms
        
        # print(f"  V2CE FPS: {optimal_fps} (video fps: {actual_video_fps})")
        # print(f"  Expected time span: {expected_v2ce_duration:.1f}ms (video: {video_duration_sec*1000:.0f}ms)")
        
        return int(optimal_fps)


class IEBCSFlareEventGenerator:
    """Generates flare events using IEBCS (ICNS Event Based Camera Simulator)."""
    
    def __init__(self, config: Dict):
        """Initialize the IEBCS flare event generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.flare_synthesizer = FlareFlickeringSynthesizer(config)
        
        iebcs_config = config['data']['event_simulator']['iebcs']
        self.simulator_path = iebcs_config['simulator_path']
        self.sensor_params = iebcs_config['sensor_parameters']
        self.noise_config = iebcs_config['noise_model']
        self.frame_config = iebcs_config['frame_generation']
        self.debug_viz_config = iebcs_config['debug_visualization']
        
        if not os.path.exists(self.simulator_path):
            raise FileNotFoundError(f"IEBCS simulator not found at: {self.simulator_path}")
        
        iebcs_src_path = os.path.join(self.simulator_path, 'src')
        if iebcs_src_path not in sys.path:
            sys.path.append(iebcs_src_path)
        
        self.iebcs_resolution = (
            config['data']['resolution_w'],  # 640
            config['data']['resolution_h']   # 480
        )
        print(f"IEBCS resolution: {self.iebcs_resolution[0]}x{self.iebcs_resolution[1]}")
        
        # 🚨 DISABLE: Flare sequence debug disabled (using unified debug_epoch_000 instead)
        self.debug_mode = False  # config.get('debug_mode', False)
        self.debug_save_dir = config.get('debug_output_dir', './output/debug_visualizations_iebcs')
        self.debug_counter = 0
        if self.debug_mode:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            print(f"🚨 DEBUG MODE (IEBCS): Will save flare sequences to {self.debug_save_dir}")
        
        self._iebcs_modules = None
    
    def _import_iebcs_modules(self):
        if self._iebcs_modules is None:
            try:
                from event_buffer import EventBuffer
                from dvs_sensor import DvsSensor
                self._iebcs_modules = {
                    'EventBuffer': EventBuffer,
                    'DvsSensor': DvsSensor
                }
                print("✅ IEBCS modules imported successfully")
            except ImportError as e:
                raise RuntimeError(f"Failed to import IEBCS modules: {e}")
        return self._iebcs_modules
    
    def generate_flare_events(self, temp_dir: Optional[str] = None, 
                            cleanup: bool = True) -> Tuple[np.ndarray, Dict, List[np.ndarray]]:
        """Generate flare events using IEBCS pipeline.
        
        Args:
            temp_dir: Optional temporary directory, creates one if None
            cleanup: Whether to cleanup temporary files
            
        Returns:
            Tuple of (events_array, timing_info, video_frames)
            Events format: [timestamp_us, x, y, polarity]
        """
        timing_info = {}
        total_start = time.time()
        
        # Import IEBCS modules
        iebcs_modules = self._import_iebcs_modules()
        EventBuffer = iebcs_modules['EventBuffer']
        DvsSensor = iebcs_modules['DvsSensor']
        
        try:
            # Step 1: Generate flickering flare video
            # print("Step 1: Generating flickering flare video for IEBCS...")
            step1_start = time.time()
            
            original_fps = self.config['data']['flare_synthesis']['max_fps']
            dynamic_fps = self._calculate_dynamic_fps_for_iebcs()
            # print(f"  Using dynamic FPS: {dynamic_fps} (original: {original_fps}) for better flicker sampling")
            
            self.config['data']['flare_synthesis']['max_fps'] = dynamic_fps
            
            video_frames, flare_metadata = self.flare_synthesizer.create_flare_event_sequence(
                target_resolution=self.iebcs_resolution
            )
            
            self.config['data']['flare_synthesis']['max_fps'] = original_fps
            
            timing_info['flare_synthesis_sec'] = time.time() - step1_start
            # print(f"  Generated {len(video_frames)} frames in {timing_info['flare_synthesis_sec']:.3f}s")
            
            # Step 2: Initialize IEBCS sensor
            # print("Step 2: Initializing IEBCS sensor...")
            step2_start = time.time()
            
            sensor = DvsSensor("IEBCSSensor")
            sensor.initCamera(
                x=self.iebcs_resolution[0],
                y=self.iebcs_resolution[1],
                lat=self.sensor_params['latency'],
                jit=self.sensor_params['jitter'],
                ref=self.sensor_params['refractory'],
                tau=self.sensor_params['tau'],
                th_pos=self.sensor_params['th_pos'],
                th_neg=self.sensor_params['th_neg'],
                th_noise=self.sensor_params['th_noise'],
                bgnp=self.sensor_params['bgnp'],
                bgnn=self.sensor_params['bgnn']
            )
            
            # Optional: Use measured noise distributions
            if self.noise_config['enable_measured_noise']:
                noise_pos_path = os.path.join(self.simulator_path, self.noise_config['noise_pos_file'])
                noise_neg_path = os.path.join(self.simulator_path, self.noise_config['noise_neg_file'])
                if os.path.exists(noise_pos_path) and os.path.exists(noise_neg_path):
                    sensor.init_bgn_hist(noise_pos_path, noise_neg_path)
                    # print("  Using measured noise distributions")
                else:
                    # print("  Warning: Measured noise files not found, using default noise model")
                    pass
            
            timing_info['sensor_init_sec'] = time.time() - step2_start
            # print(f"  IEBCS sensor initialized in {timing_info['sensor_init_sec']:.3f}s")
            
            # Step 3: Process video frames through IEBCS
            # print("Step 3: Processing frames through IEBCS...")
            step3_start = time.time()
            
            events_list = []
            dt = self.sensor_params['dt']  # microseconds
            
            # Initialize sensor with first frame
            first_frame = self._prepare_frame_for_iebcs(video_frames[0])
            sensor.init_image(first_frame)
            
            # Process subsequent frames
            for i, frame in enumerate(video_frames[1:], 1):
                img = self._prepare_frame_for_iebcs(frame)
                event_buffer = sensor.update(img, dt)
                
                # Extract events from buffer
                if event_buffer.i > 0:  # Check if any events were generated
                    frame_events = self._extract_events_from_buffer(event_buffer, i * dt)
                    events_list.extend(frame_events)
            
            timing_info['iebcs_processing_sec'] = time.time() - step3_start
            # print(f"  Processed {len(video_frames)} frames in {timing_info['iebcs_processing_sec']:.3f}s")
            
            # Step 4: Convert to numpy array
            # print("Step 4: Converting events to array...")
            step4_start = time.time()
            
            if events_list:
                events_array = np.array(events_list, dtype=np.float64)
                # Sort by timestamp
                events_array = events_array[np.argsort(events_array[:, 0])]
            else:
                # No events generated
                events_array = np.empty((0, 4), dtype=np.float64)
                print("  Warning: No events generated!")
            
            timing_info['array_conversion_sec'] = time.time() - step4_start
            # print(f"  Generated {len(events_array)} events in {timing_info['array_conversion_sec']:.3f}s")
            
            # Step 5: Debug visualization with IEBCS suffix (only in debug mode)
            if self.debug_mode:
                # print("Step 5: Saving debug visualizations (IEBCS)...")
                step5_start = time.time()
                
                self._save_debug_visualizations_iebcs(video_frames, events_array, flare_metadata)
                
                timing_info['debug_visualization_sec'] = time.time() - step5_start
                # print(f"  Debug visualization saved in {timing_info['debug_visualization_sec']:.3f}s")
            else:
                # print("Step 5: Skipping debug visualization (debug mode disabled)")
                timing_info['debug_visualization_sec'] = 0.0
            
            timing_info['total_pipeline_sec'] = time.time() - total_start
            
            print(f"🎯 IEBCS Pipeline completed in {timing_info['total_pipeline_sec']:.3f}s")
            print(f"   Final events: {len(events_array)} events")
            if len(events_array) > 0:
                event_density = len(events_array) / (timing_info['total_pipeline_sec'] * 1000)
                print(f"   Event density: {event_density:.0f} events/ms")
            
            return events_array, timing_info, video_frames
            
        except Exception as e:
            print(f"❌ IEBCS pipeline failed: {e}")
            raise
    
    def _prepare_frame_for_iebcs(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for IEBCS processing.
        
        Args:
            frame: Input frame (0-255 uint8)
            
        Returns:
            Prepared frame for IEBCS (radiometric values)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Convert to radiometric values: 255 = 10 klux
        # IEBCS expects logarithmic domain input
        frame_float = frame.astype(np.float64) / 255.0 * 1e4
        
        return frame_float
    
    def _extract_events_from_buffer(self, event_buffer, base_timestamp_us: int) -> List[List[float]]:
        """Extract events from IEBCS EventBuffer.
        
        Args:
            event_buffer: IEBCS EventBuffer object
            base_timestamp_us: Base timestamp for this frame
            
        Returns:
            List of events [timestamp_us, x, y, polarity]
        """
        events = []
        
        for i in range(event_buffer.i):
            timestamp_us = float(base_timestamp_us + event_buffer.ts[i])
            x = float(event_buffer.x[i])
            y = float(event_buffer.y[i])
            polarity = float(event_buffer.p[i])  # 0 or 1
            
            events.append([timestamp_us, x, y, polarity])
        
        return events
    
    def _calculate_dynamic_fps_for_iebcs(self) -> int:
        typical_frequency = 110.0  # Hz
        
        min_samples_per_cycle = self.frame_config['min_samples_per_cycle']
        max_fps = self.frame_config['max_fps']
        base_fps = self.frame_config['base_fps']
        
        ideal_fps = typical_frequency * min_samples_per_cycle
        
        if ideal_fps > max_fps:
            optimal_fps = max_fps
        elif ideal_fps < base_fps:
            optimal_fps = base_fps
        else:
            optimal_fps = ideal_fps
        
        # print(f"  IEBCS FPS calculation: ideal={ideal_fps:.0f}, final={optimal_fps:.0f}")
        return int(optimal_fps)
    
    def _save_debug_visualizations_iebcs(self, video_frames: List[np.ndarray], 
                                       events_array: np.ndarray, 
                                       flare_metadata: Dict):
        """Save debug visualizations with IEBCS suffix and multi-resolution support."""
        debug_sequence_dir = os.path.join(self.debug_save_dir, f"flare_seq_{self.debug_counter:03d}_iebcs")
        os.makedirs(debug_sequence_dir, exist_ok=True)
        
        # Save original frames
        frames_dir = os.path.join(debug_sequence_dir, "original_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(video_frames):
            frame_filename = os.path.join(frames_dir, f"frame_{i:03d}.png")
            cv2.imwrite(frame_filename, frame)
        
        # Save event visualizations with multi-resolution support
        events_dir = os.path.join(debug_sequence_dir, "event_visualizations")
        os.makedirs(events_dir, exist_ok=True)
        
        if len(events_array) > 0 and self.debug_viz_config.get('enable_multi_timewindow', False):
            self._create_multi_timewindow_visualizations_iebcs(
                events_array, events_dir, len(video_frames)
            )
        elif len(events_array) > 0:
            self._create_simple_event_visualizations_iebcs(
                events_array, events_dir, len(video_frames)
            )
        
        # Save metadata
        metadata_path = os.path.join(debug_sequence_dir, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"IEBCS Debug Sequence {self.debug_counter}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total frames: {len(video_frames)}\n")
            f.write(f"Total events: {len(events_array)}\n")
            f.write(f"Resolution: {self.iebcs_resolution[0]}x{self.iebcs_resolution[1]}\n")
            f.write(f"Frame interval: {self.sensor_params['dt']} μs\n")
            f.write("\nSensor Parameters:\n")
            for key, value in self.sensor_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nFlare Metadata:\n")
            for key, value in flare_metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nVisualization Config:\n")
            if self.debug_viz_config.get('enable_multi_timewindow', False):
                f.write(f"  Multi-timewindow: {self.debug_viz_config['time_window_scales']}\n")
                f.write(f"  Temporal subdivisions: {self.debug_viz_config['temporal_subdivisions']}\n")
            else:
                f.write("  Single timewindow mode\n")
        
        self.debug_counter += 1
        # print(f"  💾 Debug data saved to: {debug_sequence_dir}")
    
    def _create_multi_timewindow_visualizations_iebcs(self, events_array: np.ndarray, 
                                                     events_dir: str, num_frames: int):
        time_window_scales = self.debug_viz_config['time_window_scales']
        temporal_subdivisions = self.debug_viz_config['temporal_subdivisions']
        
        # print(f"  Creating multi-timewindow visualizations: {time_window_scales}x time windows")
        
        for scale in time_window_scales:
            scale_dir = os.path.join(events_dir, f"temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)
        
        frame_duration = self.sensor_params['dt']  # microseconds per frame
        
        for frame_idx in range(num_frames):
            frame_start_time = frame_idx * frame_duration
            frame_end_time = (frame_idx + 1) * frame_duration
            
            for scale in time_window_scales:
                if scale == 0.5:
                    if frame_idx % 2 == 0:
                        window_duration = frame_duration * scale
                        window_start = max(0, frame_end_time - window_duration)
                        window_end = frame_end_time
                        
                        window_events = events_array[
                            (events_array[:, 0] >= window_start) & 
                            (events_array[:, 0] < window_end)
                        ]
                        
                        if len(window_events) > 0:
                            viz_img = self._create_event_visualization_iebcs(window_events)
                            scale_dir = os.path.join(events_dir, f"temporal_{scale}x")
                            viz_filename = os.path.join(
                                scale_dir, 
                                f"frame_{frame_idx:03d}_{scale}x_events.png"
                            )
                            cv2.imwrite(viz_filename, viz_img)
                
                elif scale == 1:
                    window_duration = frame_duration * scale
                    window_start = max(0, frame_end_time - window_duration)
                    window_end = frame_end_time
                    
                    window_events = events_array[
                        (events_array[:, 0] >= window_start) & 
                        (events_array[:, 0] < window_end)
                    ]
                    
                    if len(window_events) > 0:
                        viz_img = self._create_event_visualization_iebcs(window_events)
                        scale_dir = os.path.join(events_dir, f"temporal_{scale}x")
                        viz_filename = os.path.join(
                            scale_dir, 
                            f"frame_{frame_idx:03d}_{scale}x_events.png"
                        )
                        cv2.imwrite(viz_filename, viz_img)
                
                else:  # scale == 2 or 4
                    subdivision_duration = frame_duration / temporal_subdivisions
                    
                    for subdivision in range(temporal_subdivisions):
                        sub_start = frame_start_time + subdivision * subdivision_duration
                        sub_end = sub_start + subdivision_duration
                        
                        window_duration = subdivision_duration * scale
                        window_start = max(0, sub_end - window_duration)
                        window_end = sub_end
                        
                        window_events = events_array[
                            (events_array[:, 0] >= window_start) & 
                            (events_array[:, 0] < window_end)
                        ]
                        
                        if len(window_events) > 0:
                            viz_img = self._create_event_visualization_iebcs(window_events)
                            scale_dir = os.path.join(events_dir, f"temporal_{scale}x")
                            viz_filename = os.path.join(
                                scale_dir, 
                                f"frame_{frame_idx:03d}_sub_{subdivision:02d}_{scale}x_events.png"
                            )
                            cv2.imwrite(viz_filename, viz_img)
    
    def _create_simple_event_visualizations_iebcs(self, events_array: np.ndarray, 
                                                events_dir: str, num_frames: int):
        for frame_idx in range(num_frames):
            frame_start_time = frame_idx * self.sensor_params['dt']
            frame_end_time = (frame_idx + 1) * self.sensor_params['dt']
            
            # Filter events for this time window
            frame_events = events_array[
                (events_array[:, 0] >= frame_start_time) & 
                (events_array[:, 0] < frame_end_time)
            ]
            
            if len(frame_events) > 0:
                # Create event visualization
                event_img = np.zeros((*self.iebcs_resolution[::-1], 3), dtype=np.uint8)
                
                for event in frame_events:
                    x, y, pol = int(event[1]), int(event[2]), int(event[3])
                    if 0 <= x < self.iebcs_resolution[0] and 0 <= y < self.iebcs_resolution[1]:
                        if pol > 0.5:  # Positive event
                            event_img[y, x] = [0, 0, 255]  # Red
                        else:  # Negative event
                            event_img[y, x] = [255, 0, 0]  # Blue
                
                event_filename = os.path.join(events_dir, f"frame_{frame_idx:03d}_events.png")
                cv2.imwrite(event_filename, event_img)
    
    def _create_event_visualization_iebcs(self, events: np.ndarray) -> np.ndarray:
        event_img = np.zeros((*self.iebcs_resolution[::-1], 3), dtype=np.uint8)
        
        for event in events:
            x, y, pol = int(event[1]), int(event[2]), int(event[3])
            if 0 <= x < self.iebcs_resolution[0] and 0 <= y < self.iebcs_resolution[1]:
                if pol > 0.5:  # Positive event
                    event_img[y, x] = [0, 0, 255]  # Red
                else:  # Negative event
                    event_img[y, x] = [255, 0, 0]  # Blue
        
        return event_img


class DVSFlareEventGenerator:
    """Generates flare events using DVS simulator integration."""
    
    def __init__(self, config: Dict):
        """Initialize the DVS flare event generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.flare_synthesizer = FlareFlickeringSynthesizer(config)
        
        dvs_config = config['data']['event_simulator']['dvs_voltmeter']
        self.simulator_path = dvs_config['simulator_path']
        
        self.dvs_resolution = (
            config['data']['resolution_w'],  # 640
            config['data']['resolution_h']   # 480
        )
        print(f"DVS simulator resolution set to: {self.dvs_resolution[0]}x{self.dvs_resolution[1]}")
        
        # 🚨 DISABLE: Flare sequence debug disabled (using unified debug_epoch_000 instead)
        self.debug_mode = False  # config.get('debug_mode', False)
        self.debug_save_dir = config.get('debug_output_dir', './output/debug_visualizations_dvs')
        self.debug_counter = 0  # Counter for unique debug filenames
        if self.debug_mode:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            print(f"🚨 DEBUG MODE (DVS): Will save flare sequences to {self.debug_save_dir}")
    
    def generate_synced_events(self, cleanup: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict, List, List]:
        """
        🔄 : 
        
        Args:
            cleanup: Whether to cleanup temporary files
            
        Returns:
            Tuple of (flare_events, light_source_events, metadata, flare_frames, light_source_frames)
        """
        timing_info = {}
        total_start = time.time()
        
        try:
            print("🎬 Generating synced flare and light source video frames...")
            video_start = time.time()
            
            flare_frames, light_source_frames, metadata = self.flare_synthesizer.create_synced_flare_and_light_source_sequences()
            
            if not flare_frames or not light_source_frames:
                print("❌ Failed to generate synced video frames")
                return None, None, {}, [], []
            
            timing_info['video_generation'] = time.time() - video_start
            print(f"  ✅ Generated {len(flare_frames)} flare frames, {len(light_source_frames)} light source frames ({timing_info['video_generation']:.2f}s)")

            flare_sim_start = time.time()
            print("🎯 Simulating flare events (with reflection)...")
            flare_events = self._simulate_events_from_frames(flare_frames, metadata, "flare", cleanup)
            timing_info['flare_simulation'] = time.time() - flare_sim_start
            
            if flare_events is None or len(flare_events) == 0:
                print("⚠️  Warning: No flare events generated")
                flare_events = np.array([]).reshape(0, 4)
            
            print(f"  ✅ Flare events: {len(flare_events):,} ({timing_info['flare_simulation']:.2f}s)")
            
            light_sim_start = time.time()
            print("💡 Simulating light source events (no reflection)...")
            light_source_events = self._simulate_events_from_frames(light_source_frames, metadata, "light_source", cleanup)
            timing_info['light_source_simulation'] = time.time() - light_sim_start
            
            if light_source_events is None or len(light_source_events) == 0:
                print("⚠️  Warning: No light source events generated")
                light_source_events = np.array([]).reshape(0, 4)
            
            print(f"  ✅ Light source events: {len(light_source_events):,} ({timing_info['light_source_simulation']:.2f}s)")
            
            timing_info['total_time'] = time.time() - total_start
            metadata.update(timing_info)
            metadata.update({
                'flare_event_count': len(flare_events),
                'light_source_event_count': len(light_source_events),
                'sync_generation': True,
                'simulator_type': 'dvs_voltmeter'
            })
            
            print(f"🎉 Synced event generation complete: {timing_info['total_time']:.2f}s total")
            
            return flare_events, light_source_events, metadata, flare_frames, light_source_frames
            
        except Exception as e:
            print(f"❌ Error in synced event generation: {e}")
            return None, None, {}, [], []
    
    def _simulate_events_from_frames(self, frames: List[np.ndarray], metadata: Dict, 
                                   event_type: str, cleanup: bool) -> Optional[np.ndarray]:
        """
        🆕 DVS
        
        Args:
            frames: 
            metadata:  (DVS)
            event_type:  ("flare"  "light_source")
            cleanup: 
            
        Returns:
             [timestamp_us, x, y, polarity]  None
        """
        temp_dir = tempfile.mkdtemp(prefix=f"{event_type}_events_")
        cleanup_temp = cleanup
        
        try:
            sequence_dir = self._save_video_for_dvs_simulator(frames, temp_dir, metadata)
            
            dvs_k1 = metadata.get('dvs_k1', None)
            if dvs_k1:
                print(f"  Using fixed k1 from script: {dvs_k1:.3f}")
                events_array = self._run_dvs_simulator_with_k1(temp_dir, dvs_k1)
            else:
                print(f"  Warning: No k1 in script, using random k1")
                events_array = self._run_dvs_simulator(temp_dir)
            
            return events_array
            
        except Exception as e:
            print(f"⚠️  Error simulating {event_type} events: {e}")
            return None
            
        finally:
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _run_dvs_simulator_with_k1(self, input_dir: str, fixed_k1: float) -> np.ndarray:
        """
        🆕 k1DVS-Voltmeter
        
        Args:
            input_dir: info.txt
            fixed_k1: k1
            
        Returns:
             [timestamp_us, x, y, polarity]
        """
        original_cwd = os.getcwd()
        
        try:
            os.chdir(self.simulator_path)
            
            self._prepare_simulator_config_with_k1(input_dir, fixed_k1)
            
            result = subprocess.run([
                sys.executable, "main.py"
            ], capture_output=True, text=True, timeout=420)
            
            if result.returncode != 0:
                raise RuntimeError(f"DVS simulator failed: {result.stderr}")
            
            return self._load_dvs_results()
            
        finally:
            os.chdir(original_cwd)
    
    def _prepare_simulator_config_with_k1(self, input_dir: str, fixed_k1: float):
        """
        🆕 k1DVS
        
        Args:
            input_dir: 
            fixed_k1: k1
        """
        config_path = os.path.join(self.simulator_path, "src/config.py")
        
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        backup_path = config_path + ".backup"
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(config_content)
        
        import re
        modified_content = re.sub(
            r"__C\.INPUT\.INTERP_PATH = ['\"].*?['\"]",
            f'__C.INPUT.INTERP_PATH = "{input_dir}"',
            config_content
        )
        
        dvs_config = self.config['data']['event_simulator']['dvs_voltmeter']
        if 'dvs346' in modified_content.lower():
            k_values = list(dvs_config['parameters']['dvs346_k'])
            k_values[0] = fixed_k1
            
            print(f"  Using fixed k1: {k_values[0]:.3f} (range: 5.0-16.0)")
            
            k_str = f"[{', '.join(map(str, k_values))}]"
            modified_content = re.sub(
                r"__C\.SENSOR\.K = \[.*?\]",
                f"__C.SENSOR.K = {k_str}",
                modified_content
            )
            
        elif 'DVS240' in modified_content:
            k_values = list(dvs_config['parameters']['dvs240_k'])
            k_values[0] = fixed_k1
            
            print(f"  Using fixed k1: {k_values[0]:.3f} (DVS240)")
            
            k_str = f"[{', '.join(map(str, k_values))}]"
            modified_content = re.sub(
                r"__C\.SENSOR\.K = \[.*?\]",
                f"__C.SENSOR.K = {k_str}",
                modified_content
            )
        
        with open(config_path, 'w') as f:
            f.write(modified_content)
        
    def generate_flare_events(self, temp_dir: Optional[str] = None, 
                            cleanup: bool = True) -> Tuple[np.ndarray, Dict, List[np.ndarray]]:
        """Generate flare events using the complete pipeline.
        
        Args:
            temp_dir: Optional temporary directory, creates one if None
            cleanup: Whether to cleanup temporary files
            
        Returns:
            Tuple of (events_array, timing_info, video_frames)
            Events format: [timestamp_us, x, y, polarity]
            video_frames: List of original flare sequence frames for debug visualization
        """
        timing_info = {}
        total_start = time.time()
        
        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="flare_events_")
            cleanup_temp = cleanup
        else:
            cleanup_temp = False
            
        try:
            # Step 1: Generate flickering flare video
            # print("Step 1: Generating flickering flare video...")
            step1_start = time.time()
            
            video_frames, flare_metadata = self.flare_synthesizer.create_flare_event_sequence(
                target_resolution=self.dvs_resolution
            )
            
            timing_info['flare_synthesis_sec'] = time.time() - step1_start
            # print(f"  Generated {len(video_frames)} frames in {timing_info['flare_synthesis_sec']:.3f}s")
            
            # Step 2: Save video frames for DVS simulator
            # print("Step 2: Saving video frames for DVS simulator...")
            step2_start = time.time()
            
            sequence_dir = self._save_video_for_dvs_simulator(video_frames, temp_dir, flare_metadata)
            
            timing_info['frame_saving_sec'] = time.time() - step2_start
            # print(f"  Saved frames in {timing_info['frame_saving_sec']:.3f}s")
            
            # Step 3: Run DVS simulator
            # print("Step 3: Running DVS simulator...")
            step3_start = time.time()
            
            events_array = self._run_dvs_simulator(temp_dir)
            
            timing_info['dvs_simulation_sec'] = time.time() - step3_start
            # print(f"  Generated {len(events_array)} events in {timing_info['dvs_simulation_sec']:.3f}s")
            
            # Step 4: Debug mode - save image sequences and event visualizations
            if self.debug_mode:
                # print("Step 4: Saving debug visualizations...")
                step4_start = time.time()
                
                self._save_debug_visualizations(video_frames, events_array, flare_metadata)
                
                timing_info['debug_visualization_sec'] = time.time() - step4_start
                # print(f"  Saved debug visualizations in {timing_info['debug_visualization_sec']:.3f}s")
            
            # Combine metadata
            timing_info.update(flare_metadata)
            timing_info['total_pipeline_sec'] = time.time() - total_start
            
            return events_array, timing_info, video_frames
            
        finally:
            # Cleanup temporary directory if requested
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    def _run_dvs_simulator(self, input_dir: str) -> np.ndarray:
        """Run the DVS-Voltmeter simulator on the prepared video frames.
        
        Args:
            input_dir: Directory containing video frames and info.txt
            
        Returns:
            Events array in format [timestamp_us, x, y, polarity]
        """
        # Change to simulator directory
        original_cwd = os.getcwd()
        
        try:
            os.chdir(self.simulator_path)
            
            # Prepare simulator config (modify paths dynamically)
            self._prepare_simulator_config(input_dir)
            
            # Run simulator
            result = subprocess.run([
                sys.executable, "main.py"
            ], capture_output=True, text=True, timeout=420)
            
            if result.returncode != 0:
                raise RuntimeError(f"DVS simulator failed: {result.stderr}")
            
            # Load generated events (DVS simulator outputs to OUT_PATH/video_name.txt)
            output_file = os.path.join(input_dir, "flare_sequence.txt")  
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"DVS simulator output not found: {output_file}")
            
            # Parse events
            events = self._parse_event_file(output_file)
            
            return events
            
        finally:
            os.chdir(original_cwd)
            
    def _save_video_for_dvs_simulator(self, video_frames: List[np.ndarray], 
                                    temp_dir: str, metadata: Dict = None) -> str:
        """Save video frames in the structure expected by DVS simulator.
        
        The DVS simulator expects:
        - temp_dir/flare_sequence/info.txt  (frame list with timestamps)
        - temp_dir/flare_sequence/*.png     (frame images)
        
        Args:
            video_frames: List of RGB video frames
            temp_dir: Temporary directory for storage
            
        Returns:
            Path to the sequence directory
        """
        sequence_name = "flare_sequence"
        sequence_dir = os.path.join(temp_dir, sequence_name)
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Calculate timestamps based on video metadata (dynamic FPS)
        if metadata and 'fps' in metadata:
            fps = metadata['fps']
        else:
            # 🚨 FIX: Use duration_range instead of deprecated duration_sec
            duration_range = self.config['data']['flare_synthesis'].get('duration_range', [0.05, 0.15])
            if isinstance(duration_range, list) and len(duration_range) == 2:
                duration_sec = (duration_range[0] + duration_range[1]) / 2  # Use average
            else:
                duration_sec = 0.1  # Fallback
            fps = len(video_frames) / duration_sec
        
        frame_duration_us = int(1e6 / fps)  # microseconds per frame
        
        frame_paths = []
        
        # Save frames and collect info
        for i, frame in enumerate(video_frames):
            frame_filename = f"{i:06d}.png"
            frame_path = os.path.join(sequence_dir, frame_filename)
            
            # Convert RGB to BGR for OpenCV and save
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
            
            # DVS simulator expects absolute paths or paths relative to IN_PATH
            relative_path = frame_path  # Use absolute path for now
            timestamp_us = i * frame_duration_us
            frame_paths.append((relative_path, timestamp_us))
        
        # Create info.txt in the sequence directory
        info_path = os.path.join(sequence_dir, "info.txt")
        with open(info_path, 'w') as f:
            for frame_path, timestamp in frame_paths:
                f.write(f"{frame_path} {timestamp:012d}\n")
        
        return sequence_dir
            
    def _prepare_simulator_config(self, input_dir: str):
        """Prepare DVS simulator configuration for the input directory.
        
        Args:
            input_dir: Directory containing video frames
        """
        config_path = os.path.join(self.simulator_path, "src/config.py")
        
        # Read current config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Create backup
        backup_path = config_path + ".backup"
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(config_content)
        
        import re
        
        modified_content = re.sub(
            r"__C\.DIR\.IN_PATH = '[^']*'",
            f"__C.DIR.IN_PATH = '{input_dir}/'",
            config_content
        )
        modified_content = re.sub(
            r"__C\.DIR\.OUT_PATH = '[^']*'",
            f"__C.DIR.OUT_PATH = '{input_dir}/'",
            modified_content
        )
        
        dvs_params = self.config['data']['event_simulator']['dvs_voltmeter'].get('parameters', {})
        if dvs_params:
            # print(f"  Applying custom DVS parameters: {dvs_params}")
            
            if 'DVS346' in modified_content:
                if 'dvs346_k' in dvs_params:
                    k_values = dvs_params['dvs346_k'].copy()
                    
                    import random
                    k1_min = 5.0
                    k1_max = 16.0
                    
                    if random.random() < 0.5:
                        k_values[0] = random.uniform(5.0, 7.0)
                    else:
                        k_values[0] = random.uniform(7.0, 16.0)
                    
                    print(f"  Random k1: {k_values[0]:.3f} (range: {k1_min}-{k1_max})")
                    
                    k_str = f"[{', '.join(map(str, k_values))}]"
                    modified_content = re.sub(
                        r"__C\.SENSOR\.K = \[.*?\]",
                        f"__C.SENSOR.K = {k_str}",
                        modified_content
                    )
                    # print(f"  DVS346 K parameters (k1 randomized): {k_values}")
            
            elif 'DVS240' in modified_content:
                if 'dvs240_k' in dvs_params:
                    k_values = dvs_params['dvs240_k']
                    k_str = f"[{', '.join(map(str, k_values))}]"
                    modified_content = re.sub(
                        r"__C\.SENSOR\.K = \[.*?\]",
                        f"__C.SENSOR.K = {k_str}",
                        modified_content
                    )  
                    # print(f"  DVS240 K parameters: {k_values}")
        else:
            # print("  Using default DVS parameters")
            pass
        
        # Write modified config
        with open(config_path, 'w') as f:
            f.write(modified_content)
            
    def _parse_event_file(self, event_file_path: str) -> np.ndarray:
        """Parse DVS simulator output file to events array.
        
        Args:
            event_file_path: Path to events text file
            
        Returns:
            Events array [timestamp_us, x, y, polarity]
        """
        events = []
        
        with open(event_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) >= 4:
                    # Parse: timestamp_us, x, y, polarity
                    timestamp = int(float(parts[0]))
                    x = int(parts[1])
                    y = int(parts[2])
                    polarity = int(parts[3])
                    
                    events.append([timestamp, x, y, polarity])
        
        return np.array(events, dtype=np.float64) if events else np.empty((0, 4), dtype=np.float64)
    
    def _save_debug_visualizations(self, video_frames: List[np.ndarray], 
                                 events_array: np.ndarray, metadata: Dict):
        """Save debug visualizations of flare sequence and events.
        
        Args:
            video_frames: List of RGB video frames
            events_array: Events array [timestamp_us, x, y, polarity]
            metadata: Flare metadata
        """
        # Create unique subdirectory for this flare sequence
        sequence_id = f"flare_seq_{self.debug_counter:03d}"
        sequence_debug_dir = os.path.join(self.debug_save_dir, sequence_id)
        os.makedirs(sequence_debug_dir, exist_ok=True)
        self.debug_counter += 1
        
        # print(f"  Saving to: {sequence_debug_dir}")
        
        # 1. Save original flare image sequence
        frames_dir = os.path.join(sequence_debug_dir, "original_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(video_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
        
        # 2. Save metadata
        metadata_path = os.path.join(sequence_debug_dir, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Flare Sequence Debug Info\n")
            f.write(f"========================\n")
            f.write(f"Total frames: {len(video_frames)}\n")
            f.write(f"Total events: {len(events_array)}\n")
            if len(events_array) > 0:
                f.write(f"Time range: {events_array[0, 0]} - {events_array[-1, 0]} μs\n")
                f.write(f"Duration: {(events_array[-1, 0] - events_array[0, 0]) / 1000:.1f} ms\n")
                f.write(f"Event rate: {len(events_array) / ((events_array[-1, 0] - events_array[0, 0]) / 1e6):.1f} events/s\n")
                pos_events = np.sum(events_array[:, 3] == 1)
                neg_events = np.sum(events_array[:, 3] == 0)
                f.write(f"Polarity: {pos_events} ON ({pos_events/len(events_array)*100:.1f}%), ")
                f.write(f"{neg_events} OFF ({neg_events/len(events_array)*100:.1f}%)\n")
            f.write(f"\nFlare Metadata:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
            
            if 'movement_distance_pixels' in metadata:
                f.write(f"\nMovement Analysis:\n")
                f.write(f"  total_distance_pixels: {metadata['movement_distance_pixels']:.1f}\n")
                f.write(f"  movement_speed_pixels_per_sec: {metadata['movement_speed_pixels_per_sec']:.1f}\n")
        
        # 3. Create multi-resolution event visualizations overlaid on frames
        if len(events_array) > 0:
            vis_dir = os.path.join(sequence_debug_dir, "event_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Calculate frame duration for event-to-frame mapping
            if metadata and 'fps' in metadata:
                fps = metadata['fps']
            else:
                # 🚨 FIX: Use duration_range instead of deprecated duration_sec
                duration_range = self.config['data']['flare_synthesis'].get('duration_range', [0.05, 0.15])
                if isinstance(duration_range, list) and len(duration_range) == 2:
                    duration_sec = (duration_range[0] + duration_range[1]) / 2  # Use average
                else:
                    duration_sec = 0.1  # Fallback
                fps = len(video_frames) / duration_sec
            
            frame_duration_us = int(1e6 / fps)
            
            # Get subdivision strategies (multiple temporal resolutions)
            subdivision_strategies = self.config.get('debug_event_subdivisions', [0.5, 1, 2, 4])
            if not isinstance(subdivision_strategies, list):
                subdivision_strategies = [subdivision_strategies]  # Backward compatibility
            
            # print(f"  Creating multi-resolution event visualizations: {subdivision_strategies}x")
            # print(f"  Frame duration: {frame_duration_us}μs")
            
            # Process each subdivision strategy
            for strategy in subdivision_strategies:
                strategy_dir = os.path.join(vis_dir, f"temporal_{strategy}x")
                os.makedirs(strategy_dir, exist_ok=True)
                
                if strategy == 0.5:
                    # 0.5x: Two frames share one event window (longer duration)
                    subdivision_duration_us = frame_duration_us * 2
                    # print(f"    {strategy}x: {subdivision_duration_us}μs per visualization (2 frames → 1 event window)")
                    
                    # Process pairs of frames
                    for frame_idx in range(0, len(video_frames), 2):
                        pair_start_time = frame_idx * frame_duration_us
                        pair_end_time = min((frame_idx + 2) * frame_duration_us, len(video_frames) * frame_duration_us)
                        
                        # Find events in this extended window
                        mask = (events_array[:, 0] >= pair_start_time) & (events_array[:, 0] < pair_end_time)
                        pair_events = events_array[mask]
                        
                        # Use first frame of the pair as base
                        if frame_idx < len(video_frames):
                            vis_frame = video_frames[frame_idx].copy()
                            if len(pair_events) > 0:
                                vis_frame = self._render_events_on_image(
                                    vis_frame, 
                                    pair_events[:, 1].astype(int),
                                    pair_events[:, 2].astype(int),
                                    pair_events[:, 3].astype(int)
                                )
                            
                            vis_filename = f"frame_{frame_idx:03d}_0.5x_events.png"
                            vis_path = os.path.join(strategy_dir, vis_filename)
                            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(vis_path, vis_frame_bgr)
                
                elif strategy == 1:
                    # 1x: One frame = one event window
                    subdivision_duration_us = frame_duration_us
                    # print(f"    {strategy}x: {subdivision_duration_us}μs per visualization (1 frame → 1 event window)")
                    
                    for frame_idx in range(len(video_frames)):
                        frame_start_time = frame_idx * frame_duration_us
                        frame_end_time = (frame_idx + 1) * frame_duration_us
                        
                        mask = (events_array[:, 0] >= frame_start_time) & (events_array[:, 0] < frame_end_time)
                        frame_events = events_array[mask]
                        
                        vis_frame = video_frames[frame_idx].copy()
                        if len(frame_events) > 0:
                            vis_frame = self._render_events_on_image(
                                vis_frame, 
                                frame_events[:, 1].astype(int),
                                frame_events[:, 2].astype(int),
                                frame_events[:, 3].astype(int)
                            )
                        
                        vis_filename = f"frame_{frame_idx:03d}_1x_events.png"
                        vis_path = os.path.join(strategy_dir, vis_filename)
                        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(vis_path, vis_frame_bgr)
                
                else:
                    # 2x, 4x: Multiple subdivisions per frame
                    num_subdivisions = int(strategy)
                    subdivision_duration_us = frame_duration_us // num_subdivisions
                    # print(f"    {strategy}x: {subdivision_duration_us}μs per visualization ({num_subdivisions} subdivisions per frame)")
                    
                    for frame_idx in range(len(video_frames)):
                        frame_start_time = frame_idx * frame_duration_us
                        frame_end_time = (frame_idx + 1) * frame_duration_us
                        
                        for sub_idx in range(num_subdivisions):
                            sub_start_time = frame_start_time + (sub_idx * subdivision_duration_us)
                            sub_end_time = min(sub_start_time + subdivision_duration_us, frame_end_time)
                            
                            mask = (events_array[:, 0] >= sub_start_time) & (events_array[:, 0] < sub_end_time)
                            sub_events = events_array[mask]
                            
                            vis_frame = video_frames[frame_idx].copy()
                            if len(sub_events) > 0:
                                vis_frame = self._render_events_on_image(
                                    vis_frame, 
                                    sub_events[:, 1].astype(int),
                                    sub_events[:, 2].astype(int),
                                    sub_events[:, 3].astype(int)
                                )
                            
                            vis_filename = f"frame_{frame_idx:03d}_sub_{sub_idx:02d}_{strategy}x_events.png"
                            vis_path = os.path.join(strategy_dir, vis_filename)
                            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(vis_path, vis_frame_bgr)
        
        # Calculate total visualizations across all strategies
        if len(events_array) > 0:
            total_visualizations = 0
            for strategy in subdivision_strategies:
                if strategy == 0.5:
                    total_visualizations += (len(video_frames) + 1) // 2  # Pairs of frames
                elif strategy == 1:
                    total_visualizations += len(video_frames)  # One per frame
                else:
                    total_visualizations += len(video_frames) * int(strategy)  # Multiple per frame
        else:
            total_visualizations = 0
            
        if 'movement_distance_pixels' in metadata and metadata['movement_distance_pixels'] > 1.0:
            self._save_movement_trajectory_visualization(sequence_debug_dir, video_frames, metadata)
        
        # print(f"  Saved {len(video_frames)} original frames and {total_visualizations} multi-resolution event visualizations")
    
    def _render_events_on_image(self, image: np.ndarray, x: np.ndarray, 
                              y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Render events on image with different colors for polarities.
        
        Args:
            image: RGB image array
            x: X coordinates of events
            y: Y coordinates of events
            p: Polarity values (0/1)
            
        Returns:
            Image with events rendered
        """
        image_vis = image.copy()
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        
        for x_, y_, p_ in zip(x, y, p):
            if 0 <= x_ < width and 0 <= y_ < height:
                if p_ == 0:  # Negative polarity - Blue
                    image_vis[y_, x_] = np.array([0, 0, 255])
                else:  # Positive polarity - Red
                    image_vis[y_, x_] = np.array([255, 0, 0])
        
        return image_vis
    
    def _save_movement_trajectory_visualization(self, sequence_debug_dir: str, 
                                              video_frames: List[np.ndarray], metadata: Dict):
        """Create movement trajectory visualization overlay.
        
        Args:
            sequence_debug_dir: Debug output directory for this sequence
            video_frames: List of RGB video frames
            metadata: Flare metadata containing movement information
        """
        trajectory_dir = os.path.join(sequence_debug_dir, "movement_trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        num_frames = len(video_frames)
        duration = metadata.get('duration_sec', 0.1)
        distance = metadata.get('movement_distance_pixels', 0)
        speed = metadata.get('movement_speed_pixels_per_sec', 0)
        
        height, width = video_frames[0].shape[:2]
        
        trajectory_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        for frame_idx, frame in enumerate(video_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.max(gray) > 50:
                _, _, _, max_loc = cv2.minMaxLoc(gray)
                center_x, center_y = max_loc
                
                color_ratio = frame_idx / max(1, num_frames - 1)
                color = (
                    int(255 * color_ratio),
                    int(255 * (1 - color_ratio)),
                    0
                )
                
                cv2.circle(trajectory_overlay, (center_x, center_y), 3, color, -1)
                
                if frame_idx == 0:
                    cv2.circle(trajectory_overlay, (center_x, center_y), 8, (0, 255, 0), 2)
                    cv2.putText(trajectory_overlay, "START", (center_x + 10, center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif frame_idx == num_frames - 1:
                    cv2.circle(trajectory_overlay, (center_x, center_y), 8, (0, 0, 255), 2)
                    cv2.putText(trajectory_overlay, "END", (center_x + 10, center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        trajectory_path = os.path.join(trajectory_dir, "movement_trajectory.png")
        cv2.imwrite(trajectory_path, trajectory_overlay)
        
        if len(video_frames) > 0:
            combined_vis = video_frames[0].copy()
            overlay_mask = np.any(trajectory_overlay > 0, axis=2)
            combined_vis[overlay_mask] = (combined_vis[overlay_mask] * 0.6 + 
                                        trajectory_overlay[overlay_mask] * 0.4).astype(np.uint8)
            
            combined_path = os.path.join(trajectory_dir, "trajectory_overlay_on_frame.png")
            cv2.imwrite(combined_path, combined_vis)
        
        trajectory_info_path = os.path.join(trajectory_dir, "trajectory_info.txt")
        with open(trajectory_info_path, 'w') as f:
            f.write(f"Movement Trajectory Analysis\n")
            f.write(f"===========================\n")
            f.write(f"Duration: {duration:.3f}s\n")
            f.write(f"Total distance: {distance:.1f} pixels\n")
            f.write(f"Average speed: {speed:.1f} pixels/s\n")
            f.write(f"Frames analyzed: {num_frames}\n")
            f.write(f"Movement per frame: {distance/max(1, num_frames-1):.2f} pixels\n")
        
        # print(f"    Saved movement trajectory visualization: {distance:.1f}px in {duration:.3f}s")
    
    def restore_simulator_config(self):
        """Restore original DVS simulator configuration."""
        config_path = os.path.join(self.simulator_path, "src/config.py")
        backup_path = config_path + ".backup"
        
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, config_path)
            os.remove(backup_path)


def test_dvs_flare_integration(config_path: str = "configs/config.yaml"):
    """Test the complete DVS-flare integration pipeline."""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize generator
    generator = DVSFlareEventGenerator(config)
    
    print("Testing DVS-Flare integration...")
    print("=" * 50)
    
    try:
        # Generate flare events
        events, timing_info, video_frames = generator.generate_flare_events()
        
        print(f"Results:")
        print(f"  Generated events: {len(events)}")
        if len(events) > 0:
            # print(f"  Time range: {events[0, 0]} - {events[-1, 0]} μs")
            # print(f"  Duration: {(events[-1, 0] - events[0, 0]) / 1000:.1f} ms")
            # print(f"  Event rate: {len(events) / ((events[-1, 0] - events[0, 0]) / 1e6):.1f} events/s")
            
            # Analyze polarity distribution
            pos_events = np.sum(events[:, 3] == 1)
            neg_events = np.sum(events[:, 3] == 0)
            # print(f"  Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), {neg_events} OFF ({neg_events/len(events)*100:.1f}%)")
        
        # print(f"\nTiming breakdown:")
        for key, value in timing_info.items():
            if key.endswith('_sec'):
                pass
                # print(f"  {key}: {value:.3f}s")
        
        return events, timing_info
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return None, None
    
    finally:
        if hasattr(generator, 'restore_simulator_config'):
            generator.restore_simulator_config()


def create_flare_event_generator(config: Dict):
    """Factory function to create the appropriate flare event generator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FlareEventGenerator instance (V2CE or DVS-Voltmeter)
        
    Raises:
        ValueError: If simulator type is not supported
    """
    simulator_type = config['data']['event_simulator']['type'].lower()
    
    if simulator_type == 'v2ce':
        print(f"🚀 Initializing V2CE flare event generator...")
        return V2CEFlareEventGenerator(config)
    elif simulator_type == 'dvs_voltmeter':
        print(f"🚀 Initializing DVS-Voltmeter flare event generator...")
        return DVSFlareEventGenerator(config)
    elif simulator_type == 'iebcs':
        print(f"🚀 Initializing IEBCS flare event generator...")
        return IEBCSFlareEventGenerator(config)
    else:
        raise ValueError(f"Unsupported simulator type: {simulator_type}. "
                        f"Supported types: 'v2ce', 'dvs_voltmeter', 'iebcs'")


def test_event_simulator_integration(config_path: str = "configs/config.yaml"):
    """Test the complete event simulator integration pipeline with selected simulator."""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get simulator type 
    simulator_type = config['data']['event_simulator']['type']
    
    # Initialize generator using factory function
    generator = create_flare_event_generator(config)
    
    print("Testing Event Simulator Integration...")
    print("=" * 50)
    print(f"Selected simulator: {simulator_type.upper()}")
    
    try:
        # Generate flare events
        events, timing_info, video_frames = generator.generate_flare_events()
        
        print(f"\nResults:")
        print(f"  Simulator: {timing_info.get('simulator_type', simulator_type)}")
        print(f"  Generated events: {len(events)}")
        if len(events) > 0:
            # print(f"  Time range: {events[0, 0]} - {events[-1, 0]} μs")
            # print(f"  Duration: {(events[-1, 0] - events[0, 0]) / 1000:.1f} ms")
            # print(f"  Event rate: {len(events) / ((events[-1, 0] - events[0, 0]) / 1e6):.1f} events/s")
            
            # Analyze polarity distribution
            pos_events = np.sum(events[:, 3] == 1) + np.sum(events[:, 3] > 0)
            neg_events = len(events) - pos_events
            # print(f"  Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), {neg_events} OFF ({neg_events/len(events)*100:.1f}%)")
            
            # Spatial analysis
            x_range = f"[{events[:, 1].min()}, {events[:, 1].max()}]"
            y_range = f"[{events[:, 2].min()}, {events[:, 2].max()}]"
            # print(f"  Spatial range: x={x_range}, y={y_range}")
        
        # print(f"\nTiming breakdown:")
        for key, value in timing_info.items():
            if key.endswith('_sec'):
                pass
                # print(f"  {key}: {value:.3f}s")
        
        return events, timing_info
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    finally:
        if hasattr(generator, 'restore_simulator_config'):
            generator.restore_simulator_config()


def test_dvs_flare_integration(config_path: str = "configs/config.yaml"):
    return test_event_simulator_integration(config_path)


if __name__ == "__main__":
    # Run test with selected simulator
    events, timing = test_event_simulator_integration()
    if events is not None:
        print(f"\nEvent simulator integration test completed successfully!")
    else:
        print(f"\nEvent simulator integration test failed!")