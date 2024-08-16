import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from architecture.ips_net import IPSNet
from data.traffic.traffic_dataset import TrafficSigns
from utils.utils import Struct

# Get the current directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration from YAML file
config_path = os.path.join(script_dir, 'config/traffic_config.yml')
with open(config_path, 'r') as ymlfile:
    c = yaml.load(ymlfile, Loader=yaml.FullLoader)
    conf = Struct(**c)

# Ensure image_size is added to the configuration
conf.data_dir = os.path.join(script_dir, 'data/traffic/dsets')
parameters_path = os.path.join(conf.data_dir, "parameters.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = IPSNet(device, conf).to(device)
model_weights_path = os.path.join(script_dir, 'model_weights_epoch_100.pth')
net.load_state_dict(torch.load(model_weights_path))
net.eval()

def get_attention_map(patches):
    with torch.no_grad():
        mem_patch_iter, mem_pos_enc_iter, mem_idx, attn = net.ips(patches)
    return mem_idx, attn

def to_dense(patches, grid_size_x, grid_size_y, patch_size):
    dense_image_height = grid_size_y * patch_size
    dense_image_width = grid_size_x * patch_size

    # Initialize the dense image with 3 channels for RGB
    dense_image = np.zeros((dense_image_height, dense_image_width, 3), dtype=np.float32)
    patch_idx = 0
    
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if patch_idx < len(patches):
                x_end = (i + 1) * patch_size
                y_end = (j + 1) * patch_size
                
                patch_height = x_end - i * patch_size
                patch_width = y_end - j * patch_size
                
                # Ensure there's space to place the patch
                if patch_height > 0 and patch_width > 0:
                    # Transpose the patch from (C, H, W) to (H, W, C)
                    dense_image[i * patch_size:x_end, j * patch_size:y_end, :] = np.transpose(patches[patch_idx][:, :patch_height, :patch_width], (1, 2, 0))
                
                patch_idx += 1
                
    return dense_image


def visualize_attention(image_sparse, mem_idx, attn, conf, save_path):
    patch_size = 100
    
    grid_size_x = 16  # Set to 15 patches along the width
    grid_size_y = 12  # Assuming the image has 12 patches along the height
    
    attention_grid = np.zeros((grid_size_y, grid_size_x))
    
    first_image_attention = attn[0].cpu().numpy()
    first_image_indices = mem_idx[0].cpu().numpy()

    for i, idx in enumerate(first_image_indices):
        row = idx // grid_size_x
        col = idx % grid_size_x
        if row < grid_size_y and col < grid_size_x:
            attention_grid[row, col] = first_image_attention[i]

    attention_grid = (attention_grid - np.min(attention_grid)) / (np.max(attention_grid) - np.min(attention_grid))

    dense_image = to_dense(image_sparse[0].cpu().numpy(), grid_size_x, grid_size_y, patch_size)
    attention_image = np.kron(attention_grid, np.ones((patch_size, patch_size)))

    cmap = plt.get_cmap('viridis')
    attention_colormap = cmap(attention_image)
    attention_colormap = np.delete(attention_colormap, 3, 2)

    original_image_rgb = (dense_image - dense_image.min()) / (dense_image.max() - dense_image.min())

    unselected_color = [1, 0, 0]
    selected_mask = np.zeros_like(original_image_rgb)
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            x_end = min((i + 1) * patch_size, original_image_rgb.shape[0])
            y_end = min((j + 1) * patch_size, original_image_rgb.shape[1])
            
            patch_height = x_end - i * patch_size
            patch_width = y_end - j * patch_size
            
            # Ensure there's space to place the patch
            if patch_height > 0 and patch_width > 0:
                if (i * grid_size_x + j) not in first_image_indices:
                    attention_colormap[i * patch_size:x_end, j * patch_size:y_end] = unselected_color
                else:
                    selected_mask[i * patch_size:x_end, j * patch_size:y_end] = attention_colormap[i * patch_size:x_end, j * patch_size:y_end]

    overlay_image = 0.6 * original_image_rgb + 0.4 * selected_mask
    overlay_image_uint8 = (overlay_image * 255).astype(np.uint8)
    overlay_image_pil = Image.fromarray(overlay_image_uint8)

    # Add patch numbers
    draw = ImageDraw.Draw(overlay_image_pil)
    font = ImageFont.load_default()

    for i in range(grid_size_y):
        for j in range(grid_size_x):
            patch_number = i * grid_size_x + j
            text_position = (j * patch_size + 10, i * patch_size + 10)
            draw.text(text_position, str(patch_number), fill=(255, 255, 255), font=font)

    overlay_image_pil.save(save_path)

    plt.figure(figsize=(15, 15))
    plt.imshow(overlay_image_pil)
    plt.axis('off')
    plt.title("Attention Map Overlay with Patch Numbers on Original Image")
    plt.show()



test_data = TrafficSigns(conf, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

for batch_idx, batch in enumerate(test_loader):
    image_sparse = batch['input']

    mem_idx, attn = get_attention_map(image_sparse.to(device))
    save_path = os.path.join(script_dir, f'attention_map_overlay_{batch_idx}.png')
    visualize_attention(image_sparse[0:1], mem_idx[0:1], attn[0:1], conf, save_path)
    
