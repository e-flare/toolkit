import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import random
import time

class FlareGeneratorUltraFastGPU:
    """
    GPU
    - 
    - 
    - ++
    """
    def __init__(self, output_size=(1920, 1080), device=None):
        self.output_size = output_size
        self.width, self.height = output_size
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("🚀 GPU (CUDA)")
            else:
                self.device = torch.device('cpu')
                print("⚠️  CUDA，CPU")
        else:
            self.device = device
            
        print(f"📊 : {self.device}")
        if self.device.type == 'cuda':
            print(f"🎯 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 : {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        
        self._precompute_uv_grids()
        
        self._preallocate_tensors()
        
        print("⚡ ，")
    
    def _precompute_uv_grids(self):
        print("🔧 UV...")
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        self.uv_x = (x_coords / self.width - 0.5) * 2.0 * (self.width / self.height)
        self.uv_y = (y_coords / self.height - 0.5) * 2.0
        self.uv_x_flat = self.uv_x.flatten()
        self.uv_y_flat = self.uv_y.flatten()
        self.uv_length = torch.sqrt(self.uv_x_flat**2 + self.uv_y_flat**2)
        
        self.fltr = torch.clamp(self.uv_length**2 * 0.5 + 0.5, max=1.0)
        
        print(f"✅ UV: {self.width}x{self.height}")
    
    def _preallocate_tensors(self):
        flat_size = self.width * self.height
        self.main_glow_color = torch.zeros(flat_size, 3, device=self.device, dtype=torch.float32)
        self.reflections_color = torch.zeros(flat_size, 3, device=self.device, dtype=torch.float32)
        
        self.i_values = torch.arange(20, device=self.device, dtype=torch.float32)
        self.zero_batch = torch.zeros(20, device=self.device, dtype=torch.float32)
        
        print("✅ ")
    
    def _ultra_fast_noise_sample(self, noise_texture, u, v):
        u_norm = torch.fmod(torch.abs(u), 1.0)
        v_norm = torch.fmod(torch.abs(v), 1.0)
        
        grid_u = u_norm * 2.0 - 1.0
        grid_v = v_norm * 2.0 - 1.0
        
        grid = torch.stack([grid_u, grid_v], dim=-1).view(1, -1, 1, 2)
        
        sampled = F.grid_sample(noise_texture, grid, 
                              mode='bilinear', padding_mode='reflection', 
                              align_corners=False)
        
        result = sampled.squeeze(-1).squeeze(0).transpose(0, 1)
        
        if result.shape[1] < 4:
            result = F.pad(result, (0, 4 - result.shape[1]))
        
        return result[:, :4]
    
    def _ultra_fast_flare_kernel(self, pos_x, pos_y, seed, flare_size,
                               noise_texture, generate_main_glow, generate_reflections):
        
        flat_size = self.uv_x_flat.size(0)
        
        self.main_glow_color.zero_()
        self.reflections_color.zero_()
        
        gn = self._ultra_fast_noise_sample(noise_texture, 
                                         torch.tensor([seed - 1.0], device=self.device), 
                                         torch.tensor([0.0], device=self.device))[0]
        gn[0] = flare_size
        
        d_x = self.uv_x_flat - pos_x
        d_y = self.uv_y_flat - pos_y
        
        if generate_main_glow:
            d_length = torch.sqrt(d_x**2 + d_y**2)
            core_intensity = (0.01 + gn[0] * 0.2) / (d_length + 0.001)
            self.main_glow_color[:, :] = core_intensity.unsqueeze(-1)
            
            angle = torch.atan2(d_y, d_x)
            halo_u = angle * 256.9 + pos_x * 2.0
            halo_noise = self._ultra_fast_noise_sample(noise_texture, halo_u, 
                                                     torch.zeros_like(halo_u))
            halo_factor = halo_noise[:, 1] * 0.25
            halo_contribution = halo_factor.unsqueeze(-1) * self.main_glow_color
            self.main_glow_color += halo_contribution
        
        if generate_reflections:
            n_u = seed + self.i_values
            n2_u = seed + self.i_values * 2.1
            nc_u = seed + self.i_values * 3.3
            
            n_batch = self._ultra_fast_noise_sample(noise_texture, n_u, self.zero_batch)
            n2_batch = self._ultra_fast_noise_sample(noise_texture, n2_u, self.zero_batch)
            nc_batch = self._ultra_fast_noise_sample(noise_texture, nc_u, self.zero_batch)
            
            nc_length = torch.sqrt(torch.sum(nc_batch**2, dim=1, keepdim=True))
            nc_batch = (nc_batch + nc_length) * 0.65
            
            i_indices = torch.arange(20, device=self.device).repeat_interleave(3)  # [0,0,0,1,1,1,...]
            j_indices = torch.arange(3, device=self.device).repeat(20)  # [0,1,2,0,1,2,...]
            
            n_selected = n_batch[i_indices]  # (60, 4)
            n2_selected = n2_batch[i_indices]
            nc_selected = nc_batch[i_indices]
            
            ip_batch = (n_selected[:, 0] * 3.0 + 
                       j_indices.float() * 0.1 * n2_selected[:, 1]**3)  # (60,)
            is_batch = n_selected[:, 1]**2 * 4.5 * gn[0] + 0.1  # (60,)
            ia_batch = (n_selected[:, 2] * 4.0 - 2.0) * n2_selected[:, 0] * n_selected[:, 1]  # (60,)
            
            mix_factors = 1.0 + (self.uv_length.unsqueeze(0) - 1.0) * n_selected[:, 3:4]**2  # (60, N)
            iuv_x_batch = self.uv_x_flat.unsqueeze(0) * mix_factors  # (60, N)
            iuv_y_batch = self.uv_y_flat.unsqueeze(0) * mix_factors
            
            cos_ia_batch = torch.cos(ia_batch).unsqueeze(-1)  # (60, 1)
            sin_ia_batch = torch.sin(ia_batch).unsqueeze(-1)
            rotated_x_batch = iuv_x_batch * cos_ia_batch + iuv_y_batch * sin_ia_batch  # (60, N)
            rotated_y_batch = -iuv_x_batch * sin_ia_batch + iuv_y_batch * cos_ia_batch
            
            ip_expanded = ip_batch.unsqueeze(-1)  # (60, 1)
            id_x_batch = ((rotated_x_batch - pos_x) * (1.0 - ip_expanded) + 
                         (rotated_x_batch + pos_x) * ip_expanded)  # (60, N)
            id_y_batch = ((rotated_y_batch - pos_y) * (1.0 - ip_expanded) + 
                         (rotated_y_batch + pos_y) * ip_expanded)
            id_length_batch = torch.sqrt(id_x_batch**2 + id_y_batch**2)  # (60, N)
            
            intensity_base_batch = torch.clamp(is_batch.unsqueeze(-1) - id_length_batch, min=0.0)  # (60, N)
            mask_batch = intensity_base_batch > 0
            
            intensity_batch = torch.zeros_like(intensity_base_batch)
            valid_mask = mask_batch.any(dim=1)
            
            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                for idx in valid_indices:
                    pixel_mask = mask_batch[idx]
                    if pixel_mask.any():
                        channel = j_indices[idx].item()
                        i_orig = i_indices[idx].item()
                        
                        intensity_val = (intensity_base_batch[idx, pixel_mask]**0.45 / 
                                       is_batch[idx] * 0.1 * gn[0] * 
                                       nc_batch[i_orig, channel] * self.fltr[pixel_mask])
                        
                        self.reflections_color[pixel_mask, channel] += intensity_val
        
        result = self.main_glow_color + self.reflections_color
        return result.view(self.height, self.width, 3)
    
    def generate(self, light_pos, noise_image_path, time=0.0, flare_size=0.15, light_color=(1.0, 1.0, 1.0),
                 generate_main_glow=False, generate_reflections=True):
        """
         - 
        """
        if not hasattr(self, '_cached_texture_path') or self._cached_texture_path != noise_image_path:
            noise_img = Image.open(noise_image_path).convert("RGBA")
            noise_array = np.array(noise_img).astype(np.float32) / 255.0
            self._noise_texture = torch.from_numpy(noise_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self._cached_texture_path = noise_image_path
        
        pos_x = (light_pos[0] / self.width - 0.5) * 2.0 * (self.width / self.height)
        pos_y = (light_pos[1] / self.height - 0.5) * 2.0
        
        img_array = self._ultra_fast_flare_kernel(
            pos_x, pos_y, time, flare_size,
            self._noise_texture, generate_main_glow, generate_reflections
        )
        
        light_color_tensor = torch.tensor(light_color, device=self.device, dtype=torch.float32)
        img_array *= light_color_tensor
        
        noise_addition = self._ultra_fast_noise_sample(
            self._noise_texture, 
            self.uv_x_flat / self.width,
            self.uv_y_flat / self.height
        )[:, :3].view(self.height, self.width, 3)
        img_array += noise_addition * 0.01
        
        img_array = torch.clamp(img_array, 0, 1)
        img_array = (img_array * 255).cpu().numpy().astype(np.uint8)
        
        return Image.fromarray(img_array)

if __name__ == '__main__':
    OUTPUT_RESOLUTION = (640, 480)
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare_ultra_fast_test'

    if not os.path.isdir(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_textures: 
        raise FileNotFoundError(f"： '{TEXTURE_SOURCE_DIR}' 。")

    generator = FlareGeneratorUltraFastGPU(output_size=OUTPUT_RESOLUTION)

    print(f"\n⚡ --- GPU ---")
    
    fixed_source_path = os.path.join(TEXTURE_SOURCE_DIR, random.choice(available_textures))
    fixed_light_pos = (generator.width * 0.4, generator.height * 0.4)
    print(f": '{os.path.basename(fixed_source_path)}'")
    
    print("\n🏃‍♂️ ：50...")
    
    total_start_time = time.time()
    
    for i in range(50):
        light_pos = (random.randint(100, generator.width-100), 
                    random.randint(100, generator.height-100))
        
        img = generator.generate(
            light_pos=light_pos, 
            noise_image_path=fixed_source_path,
            time=random.random() * 50,
            flare_size=random.uniform(0.1, 0.3),
            generate_main_glow=False, 
            generate_reflections=True
        )
        
        output_name = f"ultra_fast_{i+1:03d}.png"
        img.save(os.path.join(OUTPUT_DIR, output_name))
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start_time
            current_fps = (i + 1) / elapsed
            print(f" {i+1}/50， FPS: {current_fps:.2f}")
    
    total_time = time.time() - total_start_time
    final_fps = 50 / total_time
    
    print(f"\n🚀 :")
    print(f": {total_time:.2f}")
    print(f"FPS: {final_fps:.2f}")
    print(f": >20 FPS")
    print("==================================")