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

# Attention maps script for the megapixel MNIST data set
# This script requires "Attention_maps" to be set to True in the config file for IPS to return the attention values as well as the attention indices 
# The script Downloads the Png files directly to the working directory

# Get the current directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load config file
config_path = os.path.join(script_dir, 'config/traffic_config.yml')
with open(config_path,  'r')  as ymlfile:
    c = yaml.load(ymlfile, Loader = yaml.FullLoader)
    conf = Struct(**c)

# Assign GPU to torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Call ips
net = IPSNet(device, conf).to(device)
model_weights_path = os.path.join(script_dir, 'model_weights_epoch_140_patch_size_25_25.pth')
net.load_state_dict(torch.load(model_weights_path))
net.eval()

# function to extract atteniton values from Ips.net
def get_attention_map(patches):
    with torch.no_grad():
        mem_patch_iter, mem_pos_enc_iter, mem_idx, attn = net.ips(patches)
    return mem_idx,  attn

# Create the grid
def to_dense(patches, grid_size_x, grid_size_y, patch_size):
    dense_image_height = grid_size_y * patch_size
    dense_image_width = grid_size_x * patch_size

    # Initialize the dense image with 3 channels for RGB
    dense_image = np.zeros((dense_image_height, dense_image_width, 3), dtype=np.float32)
    patch_idx = 0
    
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            # Check that the idx is not higher than the total number of patches
            if patch_idx < len(patches):
                # find the indexes of the bottom right corner of each patch
                x_end = (i + 1) * patch_size
                y_end = (j + 1) * patch_size
                
                patch_height = x_end - i * patch_size
                patch_width = y_end - j * patch_size
            
                if patch_height > 0 and patch_width > 0:
                    # Transpose the patch from (C, H, W) to (H, W, C)
                    dense_image[i * patch_size:x_end, j * patch_size:y_end, :] = np.transpose(patches[patch_idx][:, :patch_height, :patch_width], (1, 2, 0))
                
                patch_idx += 1
                
    return dense_image

def visualize_attention(image_sparse, mem_idx, attn, conf, save_path):
    # Ensure that the patch size is consitent with the patch size used for training the weights
    patch_size = 25

    # grid_size_x and grid_size_y are specific the the Swedish traffic sign data (960, 1280) and need to be consistent with the patch size
    grid_size_x = 64  
    grid_size_y = 48 

    # Create attention grid
    attention_grid = np.zeros((grid_size_y, grid_size_x))

    # Extract the attention from tensors to numpy
    first_image_attention = attn[0].cpu().numpy()
    first_image_indices = mem_idx[0].cpu().numpy()

    # Simple grid logic for placing values on a grid based on the indices and the grid height and width
    for i, idx in enumerate(first_image_indices):
        row = idx // grid_size_x
        col = idx % grid_size_x
        if row < grid_size_y and col < grid_size_x:
            attention_grid[row, col] = first_image_attention[i]

    # Normalize attention values to range [0, 1]
    attention_grid = (attention_grid - np.min(attention_grid)) / (np.max(attention_grid) - np.min(attention_grid))

    dense_image = to_dense(image_sparse[0].cpu().numpy(), grid_size_x, grid_size_y, patch_size)

    # custom colouring
    def attention_to_color(value):
        if value < 0.3:
            return (1, 0.4, 0.6)  #  Pink
        elif value < 0.5:
            return (0, 0, 1)  # Blue
        else:
            return (1, 1, 0)  # Yellow

    # Set the color transparency
    alpha_value = 0.4  

    # Create the colour map to be overlayed on top of the original image
    attention_colormap = np.zeros((grid_size_y * patch_size, grid_size_x * patch_size, 4), dtype=np.float32)

    # For loop to assign colours to each value if their attention values are superior to 0.
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if attention_grid[i, j] > 0:
                x_start = j * patch_size
                y_start = i * patch_size
                x_end = x_start + patch_size
                y_end = y_start + patch_size
                
                color = attention_to_color(attention_grid[i, j])
                
                # Fill color with adjustable transparency
                attention_colormap[y_start:y_end, x_start:x_end, :3] = color
                attention_colormap[y_start:y_end, x_start:x_end, 3] = alpha_value  # Apply transparency

    # Overlay the images
    attention_colormap_image = Image.fromarray((attention_colormap * 255).astype(np.uint8), 'RGBA')
    original_image_rgb = (dense_image - dense_image.min()) / (dense_image.max() - dense_image.min())
    original_image_pil = Image.fromarray((original_image_rgb * 255).astype(np.uint8)).convert('RGBA')

    # Combine original image with attention overlay
    combined_image = Image.alpha_composite(original_image_pil, attention_colormap_image)

    # Create red borders for each tile that has an attention value superior to 0
    draw = ImageDraw.Draw(combined_image)
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if attention_grid[i, j] > 0:
                x_start = j * patch_size
                y_start = i * patch_size
                x_end = x_start + patch_size
                y_end = y_start + patch_size
                draw.rectangle([x_start, y_start, x_end, y_end], outline="red", width=2)

    # Save and show the result
    combined_image.save(save_path)

    plt.figure(figsize=(15, 15))
    plt.imshow(combined_image)
    plt.axis('off')
    plt.title("Attention Map Overlay with Gradient Colors, Red Outlines, and Adjustable Transparency")
    plt.show()

# Load test data
test_data = TrafficSigns(conf, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
# Call the functions
for batch_idx, batch in enumerate(test_loader):
    image_sparse = batch['input']

    mem_idx, attn = get_attention_map(image_sparse.to(device))
    save_path = os.path.join(script_dir, f'attention_map_overlay_{batch_idx}.png')
    visualize_attention(image_sparse[0:1], mem_idx[0:1], attn[0:1], conf, save_path)
    
