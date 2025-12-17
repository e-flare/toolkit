"""
Custom Training System for Event-Voxel Denoising
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional
import json
from tqdm import tqdm

class EventVoxelTrainer:
    """
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        
        Args:
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        

        self.logger = logging.getLogger(__name__)
        

        self._setup_optimizer_and_loss()
        

        self._setup_scheduler()
        

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_val_loss = float('inf')
        

        self.checkpoint_dir = Path(config['trainer']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        

        if config.get('logger', {}).get('name') == 'TensorBoardLogger':
            log_dir = config['logger']['log_dir']
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        self.logger.info(f"EventVoxelTrainer initialized:")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Model: {model.__class__.__name__}")
        self.logger.info(f"  Train samples: {len(train_loader.dataset)}")
        self.logger.info(f"  Val samples: {len(val_loader.dataset)}")
        self.logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        

        self._debug_model_architecture()
    
    def _debug_model_architecture(self):
        self.logger.info("=== Model Architecture Debug ===")
        

        if hasattr(self.model, 'final_sigmoid'):
            self.logger.info(f"Model final_sigmoid: {self.model.final_sigmoid}")
        

        model_children = list(self.model.children())
        if model_children:
            last_layer = model_children[-1]
            self.logger.info(f"Last layer type: {type(last_layer)}")
            

        self.model.eval()
        with torch.no_grad():

            test_input_zeros = torch.zeros(1, 1, 8, 480, 640).to(self.device)
            test_output_zeros = self.model(test_input_zeros)
            self.logger.info(f"Test with zeros input: output_mean={test_output_zeros.mean():.6f}, output_std={test_output_zeros.std():.6f}")
            

            test_input_ones = torch.ones(1, 1, 8, 480, 640).to(self.device)
            test_output_ones = self.model(test_input_ones)
            self.logger.info(f"Test with ones input: output_mean={test_output_ones.mean():.6f}, output_std={test_output_ones.std():.6f}")
            

            test_input_random = torch.randn(1, 1, 8, 480, 640).to(self.device)
            test_output_random = self.model(test_input_random)
            self.logger.info(f"Test with random input: output_mean={test_output_random.mean():.6f}, output_std={test_output_random.std():.6f}")
        
        self.model.train()
        self.logger.info("=== End Model Debug ===")
    
    def _setup_optimizer_and_loss(self):
        opt_config = self.config['optimizer']
        
        if opt_config['name'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['name'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['name']}")
        

        loss_config = self.config['loss']
        if loss_config['name'] == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_config['name'] == 'L1Loss':
            self.criterion = nn.L1Loss()
        elif loss_config['name'] == 'SmoothL1Loss':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss: {loss_config['name']}")
        
        self.logger.info(f"Optimizer: {opt_config['name']} (LR: {opt_config['learning_rate']})")
        self.logger.info(f"Loss function: {loss_config['name']}")
    
    def _setup_scheduler(self):
        if 'lr_scheduler' not in self.config:
            self.scheduler = None
            return
        
        sched_config = self.config['lr_scheduler']
        
        if sched_config['name'] == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=sched_config['milestones'],
                gamma=sched_config['gamma']
            )
        elif sched_config['name'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            self.scheduler = None
            self.logger.warning(f"Unsupported scheduler: {sched_config['name']}")
        
        if self.scheduler:
            self.logger.info(f"Scheduler: {sched_config['name']}")
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        

        progress_bar = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
            ncols=100
        )
        
        for batch_idx, batch in progress_bar:

            debug_config = self.config.get('debug', {})
            if debug_config.get('enabled', False):
                max_iterations = debug_config.get('max_iterations', 2)
                if batch_idx >= max_iterations:
                    self.logger.info(f"🐛 DEBUG MODE: Stopping after {max_iterations} iterations")
                    break
            

            inputs = batch['raw'].to(self.device)      # (B, 1, 8, H, W) 
            targets = batch['label'].to(self.device)   # (B, 1, 8, H, W)
            

            if debug_config.get('enabled', False) and batch_idx < 2:
                self._trigger_debug_visualization(
                    batch_idx, inputs, targets, batch, 
                    debug_config['debug_dir'], self.current_epoch
                )
            

            self.optimizer.zero_grad()
            outputs = self.model(inputs)               # (B, 1, 8, H, W)
            

            if debug_config.get('enabled', False) and batch_idx < 2:
                self._trigger_model_output_visualization(
                    batch_idx, inputs, outputs, debug_config['debug_dir'], self.current_epoch
                )
            

            if batch_idx % 100 == 0:
                print(f"\n[DEBUG-TRAIN] Batch {batch_idx}: Input mean={inputs.mean():.4f}, Output mean={outputs.mean():.4f}")
                print(f"[DEBUG-TRAIN] Are outputs identical to inputs? {torch.equal(outputs, inputs)}")
                print(f"[DEBUG-TRAIN] Model in training mode? {self.model.training}")
            

            loss = self.criterion(outputs, targets)
            

            loss.backward()
            self.optimizer.step()
            

            total_loss += loss.item()
            num_batches += 1
            self.current_iteration += 1
            

            avg_loss_so_far = total_loss / num_batches
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{avg_loss_so_far:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            

            if self.writer:
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), self.current_iteration)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_iteration)
            

            trainer_config = self.config['trainer']
            validate_after_iters = trainer_config.get('validate_after_iters', 100)
            
            if self.current_iteration % validate_after_iters == 0:

                print(f"\n[DEBUG] Validation triggered: iter={self.current_iteration}, validate_after_iters={validate_after_iters}", flush=True)
                

                progress_bar.set_description(f"Epoch {self.current_epoch + 1} (Validating...)")
                

                model_params_sum = sum([p.sum().item() for p in self.model.parameters()])
                print(f"[DEBUG] Model params sum: {model_params_sum:.6f}", flush=True)
                

                val_metrics = self.validate_epoch()
                

                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                

                try:
                    self.save_checkpoint(is_best=is_best)
                    checkpoint_name = f"epoch_{self.current_epoch:04d}_iter_{self.current_iteration:06d}"
                    checkpoint_status = f"✅({checkpoint_name})" 
                except Exception as e:
                    checkpoint_status = f"❌({e})"
                

                best_indicator = " 🎯" if is_best else ""
                result_msg = f"\n💯 Iter {self.current_iteration:4d}: Val={val_metrics['loss']:.4f}{best_indicator} {checkpoint_status}"
                print(result_msg, flush=True)
                

                progress_bar.set_description(f"Epoch {self.current_epoch + 1}")
            

            max_iters = trainer_config.get('max_num_iterations')
            if max_iters and self.current_iteration >= max_iters:
                progress_bar.set_description(f"Epoch {self.current_epoch + 1} (Max iters reached)")
                break
        
        progress_bar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        



        max_val_batches = 10
        

        batch_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_val_batches:
                    break
                inputs = batch['raw'].to(self.device)
                targets = batch['label'].to(self.device)
                

                if batch_idx == 0:
                    print(f"[DEBUG] Input stats: min={inputs.min():.4f}, max={inputs.max():.4f}, mean={inputs.mean():.4f}")
                    print(f"[DEBUG] Target stats: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")

                    unique_values_input = torch.unique(inputs).cpu().numpy()[:10]
                    unique_values_target = torch.unique(targets).cpu().numpy()[:10]
                    print(f"[DEBUG] Input unique values (first 10): {unique_values_input}")
                    print(f"[DEBUG] Target unique values (first 10): {unique_values_target}")
                
                outputs = self.model(inputs)
                
                if batch_idx == 0:
                    print(f"[DEBUG] Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
                    print(f"[DEBUG] Output shape: {outputs.shape}")
                    print(f"[DEBUG] Are outputs identical to targets? {torch.equal(outputs, targets)}")
                    print(f"[DEBUG] Output dtype: {outputs.dtype}, device: {outputs.device}")
                    print(f"[DEBUG] Output requires_grad: {outputs.requires_grad}")
                    

                    print(f"[DEBUG] Checking model architecture...")
                
                loss = self.criterion(outputs, targets)
                
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                total_loss += batch_loss
                num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        

        batch_losses_str = ", ".join([f"{loss:.4f}" for loss in batch_losses])
        print(f"[DEBUG] Validation completed: {num_batches} batches, avg_loss={avg_loss:.6f}", flush=True)
        print(f"[DEBUG] Individual batch losses: [{batch_losses_str}]", flush=True)
        print(f"[DEBUG] Input shape: {inputs.shape}, Target shape: {targets.shape}", flush=True)
        

        self.model.train()
        
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        

        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        

        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            # self.logger.info(f"Saved best checkpoint: {best_path}")  # Reduce verbosity
        

        iter_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:04d}_iter_{self.current_iteration:06d}.pth'
        torch.save(checkpoint, iter_path)
        
        # self.logger.info(f"Saved checkpoint: {latest_path}")  # Reduce verbosity
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.best_val_loss = checkpoint['best_val_loss']
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def train(self):
        self.logger.info("=== Starting Event-Voxel Denoising Training ===")
        
        trainer_config = self.config['trainer']
        max_epochs = trainer_config.get('max_num_epochs', 100)
        max_iters = trainer_config.get('max_num_iterations', None)
        validate_after_iters = trainer_config.get('validate_after_iters', 100)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            

            print(f"\n📊 Epoch {epoch + 1}/{max_epochs}")
            

            train_metrics = self.train_epoch()
            

            print(f"✅ Epoch {epoch + 1:3d}: Train={train_metrics['loss']:.4f}")
            

            if self.writer:
                self.writer.add_scalar('Loss/Train_Epoch', train_metrics['loss'], epoch)
            

            self.save_checkpoint(is_best=False)
            

            if self.scheduler:
                self.scheduler.step()
            

            if max_iters and self.current_iteration >= max_iters:
                print(f"🛑 Training stopped: reached max iterations {max_iters}")

                val_metrics = self.validate_epoch()
                is_best = val_metrics['loss'] < self.best_val_loss
                best_indicator = " 🎯" if is_best else ""
                print(f"🏁 Final: Train={train_metrics['loss']:.4f}, Val={val_metrics['loss']:.4f}{best_indicator}")
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=is_best)
                break
        

        total_time = time.time() - start_time
        print(f"\n🎉 Training Completed!")
        print(f"⏱️  Total time: {total_time/3600:.2f}h")
        print(f"🎯 Best val loss: {self.best_val_loss:.4f}")
        print(f"💾 Checkpoint: {self.checkpoint_dir}/best_checkpoint.pth")
        

        summary = {
            'total_epochs': self.current_epoch + 1,
            'total_iterations': self.current_iteration,
            'best_val_loss': self.best_val_loss,
            'total_time_hours': total_time / 3600,
            'final_lr': self.optimizer.param_groups[0]['lr']
        }
        
        summary_path = self.checkpoint_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        return self.best_val_loss
    
    def _trigger_debug_visualization(self, batch_idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                                   batch: dict, debug_dir: str, epoch: int):
        """
        """
        try:
            import os
            from pathlib import Path
            

            iteration_dir = Path(debug_dir) / f"epoch_{epoch:03d}_iter_{batch_idx:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🐛 Generating 9 debug visualizations for Epoch {epoch}, Batch {batch_idx}")
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
                           name="input_events", num_time_slices=8)
            

            input_voxel_dir = iteration_dir / "3_input_voxel"
            input_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(input_voxel, sensor_size=(480, 640), output_dir=str(input_voxel_dir), 
                          name="input_voxel", duration_ms=20)
            


            target_events_dir = iteration_dir / "4_target_events"
            target_events_dir.mkdir(exist_ok=True)
            visualize_events(target_events_np, sensor_size=(480, 640), output_dir=str(target_events_dir), 
                           name="target_events", num_time_slices=8)
            

            target_voxel_dir = iteration_dir / "6_target_voxel"
            target_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(target_voxel, sensor_size=(480, 640), output_dir=str(target_voxel_dir), 
                          name="target_voxel", duration_ms=20)
            
            self.logger.info(f"🐛 Generated input and target visualizations (1,3,4,6/9) in {iteration_dir}")
            
        except Exception as e:
            self.logger.warning(f"🐛 Debug visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _trigger_model_output_visualization(self, batch_idx: int, inputs: torch.Tensor, outputs: torch.Tensor, 
                                          debug_dir: str, epoch: int):
        """
        """
        try:
            from pathlib import Path
            
            iteration_dir = Path(debug_dir) / f"epoch_{epoch:03d}_iter_{batch_idx:03d}"
            

            input_voxel = inputs[0, 0].cpu()   # (8, H, W)
            output_voxel = outputs[0, 0].cpu()
            

            is_true_residual = hasattr(self.model, 'get_residual')
            
            if is_true_residual:

                with torch.no_grad():
                    residual_voxel = self.model.get_residual(inputs)[0, 0].cpu()  # (8, H, W)
                
                self.logger.info(f"🐛 True residual learning detected:")
                self.logger.info(f"🐛   Input mean: {input_voxel.mean():.4f}, std: {input_voxel.std():.4f}")
                self.logger.info(f"🐛   Residual mean: {residual_voxel.mean():.4f}, std: {residual_voxel.std():.4f}")
                self.logger.info(f"🐛   Output mean: {output_voxel.mean():.4f}, std: {output_voxel.std():.4f}")
                self.logger.info(f"🐛   Identity check: output ≈ input + residual = {torch.allclose(output_voxel, input_voxel + residual_voxel, atol=1e-6)}")
                

                residual_voxel_dir = iteration_dir / "8_residual_voxel"
                residual_voxel_dir.mkdir(exist_ok=True)
                
                from src.data_processing.professional_visualizer import visualize_voxel
                visualize_voxel(residual_voxel, sensor_size=(480, 640), output_dir=str(residual_voxel_dir), 
                              name="residual_voxel", duration_ms=20)
                

                if residual_voxel.abs().sum() > 1e-6:
                    from src.data_processing.decode import voxel_to_events
                    residual_events_np = voxel_to_events(residual_voxel, total_duration=20000, sensor_size=(480, 640))
                    
                    residual_events_dir = iteration_dir / "8_residual_events"
                    residual_events_dir.mkdir(exist_ok=True)
                    
                    from src.data_processing.professional_visualizer import visualize_events
                    visualize_events(residual_events_np, sensor_size=(480, 640), output_dir=str(residual_events_dir), 
                                   name="residual_events", num_time_slices=8)
            

            from src.data_processing.decode import voxel_to_events
            output_events_np = voxel_to_events(output_voxel, total_duration=20000, sensor_size=(480, 640))
            
            from src.data_processing.professional_visualizer import visualize_events, visualize_voxel
            


            output_events_dir = iteration_dir / "7_output_events"
            output_events_dir.mkdir(exist_ok=True)
            visualize_events(output_events_np, sensor_size=(480, 640), output_dir=str(output_events_dir), 
                           name="final_output_events", num_time_slices=8)
            

            output_voxel_dir = iteration_dir / "9_output_voxel"
            output_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(output_voxel, sensor_size=(480, 640), output_dir=str(output_voxel_dir), 
                          name="final_output_voxel", duration_ms=20)
            
            folder_count = "8" if is_true_residual else "6"
            self.logger.info(f"🐛 Generated final output visualizations (7,9/{folder_count}) in {iteration_dir}")
            if is_true_residual:
                self.logger.info(f"🐛 Generated residual visualizations (8/8) in {iteration_dir}")
                self.logger.info(f"🐛 All debug visualizations completed! (8 folders total: 1,3,4,6,7,8,9)")
            else:
                self.logger.info(f"🐛 All debug visualizations completed! (6 folders total: 1,3,4,6,7,9)")
            

            self._generate_debug_summary(iteration_dir, batch_idx, epoch, is_true_residual)
            
        except Exception as e:
            self.logger.warning(f"🐛 Model output visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _generate_debug_summary(self, iteration_dir: Path, batch_idx: int, epoch: int, is_true_residual: bool = False):
        try:
            summary_file = iteration_dir / "debug_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write(f"Debug Visualization Summary\n")
                f.write(f"========================\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Batch: {batch_idx}\n")
                f.write(f"Model: {self.model.__class__.__name__}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"True Residual Learning: {'Yes' if is_true_residual else 'No'}\n")
                
                if is_true_residual:
                    f.write(f"\n8 Visualization Folders (True Residual Learning):\n")
                    f.write(f"1. 1_input_events/          - Input events (3D+2D+temporal) comprehensive\n")
                    f.write(f"2. 3_input_voxel/           - Input voxel temporal bins\n")
                    f.write(f"3. 4_target_events/         - Target events (3D+2D+temporal) comprehensive\n") 
                    f.write(f"4. 6_target_voxel/          - Target voxel temporal bins\n")
                    f.write(f"5. 7_output_events/         - Final output events (input + residual) comprehensive\n")
                    f.write(f"6. 8_residual_voxel/        - Learned residual voxel temporal bins\n")
                    f.write(f"7. 8_residual_events/       - Learned residual events (if non-zero)\n")
                    f.write(f"8. 9_output_voxel/          - Final output voxel temporal bins\n")
                    f.write(f"\nTrue Residual Architecture:\n")
                    f.write(f"  final_output = input_voxel + backbone_network(input_voxel)\n")
                    f.write(f"  backbone learns residual ≈ -flare_noise\n")
                    f.write(f"  Zero-initialized final layer for perfect initial identity mapping\n")
                else:
                    f.write(f"\n6 Visualization Folders (Standard Model):\n")
                    f.write(f"1. 1_input_events/          - Input events (3D+2D+temporal) comprehensive\n")
                    f.write(f"2. 3_input_voxel/           - Input voxel temporal bins\n")
                    f.write(f"3. 4_target_events/         - Target events (3D+2D+temporal) comprehensive\n") 
                    f.write(f"4. 6_target_voxel/          - Target voxel temporal bins\n")
                    f.write(f"5. 7_output_events/         - Model output events (3D+2D+temporal) comprehensive\n")
                    f.write(f"6. 9_output_voxel/          - Model output voxel temporal bins\n")
                
                f.write(f"\nData Format:\n")
                f.write(f"- Events: (N, 4) [t, x, y, p]\n")
                f.write(f"- Voxel: (8, 480, 640) [8 temporal bins]\n")
                f.write(f"- Duration: 20ms per segment\n")
                f.write(f"- All operations in voxel space (not event space)\n")
                
            self.logger.info(f"🐛 Debug summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.warning(f"🐛 Failed to generate debug summary: {e}")