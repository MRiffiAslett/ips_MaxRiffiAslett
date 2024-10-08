import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from architecture.ips_net import IPSNet
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from utils.utils import Struct

# Attention maps script for the megapixel MNIST data set
# This script requires "Attention_maps" to be set to True in the config file for IPS to return the attention values as well as the attention indices 
# The script Downloads the Png files directly to the working directory


# Get the current directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration 
config_path = os.path.join(script_dir, 'config/mnist_config.yml')
with open( config_path, 'r') as ymlfile:
    c = yaml.load(ymlfile, Loader=yaml.FullLoader )
    conf = Struct(**c)

# Set variables
conf.image_size = 1500

# Set data directory
conf.data_dir = os.path.join(script_dir, 'data/megapixel_mnist/dsets/megapixel_mnist_1500' )
parameters_path =  os.path.join(conf.data_dir, "parameters.json")

# Assign GPU to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# call IPS
net = IPSNet(device, conf).to(device)
model_weights_path =  os.path.join(script_dir, 'model_weights.pth')
net.load_state_dict(torch.load(model_weights_path))
net.eval()

# Define the function  to get the attention indexes and values
def get_attention_map(patches):
    with torch.no_grad():
        mem_patch_iter,  mem_pos_enc_iter, mem_idx,  attn = net.ips(patches)
    return mem_idx, attn

# Create a grid corresponding the each patch. 
def to_dense(patches, grid_size, patch_size):
    dense_image = np.zeros((grid_size  * patch_size, grid_size * patch_size) , dtype=np.float32)
    patch_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            dense_image[ i * patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[patch_idx, 0]
            patch_idx += 1
    return dense_image


# The following function performs a classic grid assignment logic
def visualize_attention(image_sparse, mem_idx, attn, conf, save_path):
    # The patch size and num_patches need to correspond to the parameters used while training
    patch_size =  50  
    num_patches  = 900  
    grid_size = int(np.sqrt(num_patches))

    attention_grid = np.zeros((grid_size, grid_size))
    first_image_attention = attn[0].cpu().numpy()
    first_image_indices = mem_idx[0].cpu().numpy()
    
    for i, idx in enumerate(first_image_indices):
        row = idx  // grid_size
        col =  idx % grid_size
        attention_grid[row, col] = first_image_attention[i]

    # Normalize the attention grid 
    attention_grid = (attention_grid   - np.min(attention_grid)) /(np.max(attention_grid) - np.min(attention_grid))

    # Convert sparse to dense image
    dense_image = to_dense(image_sparse[0].cpu().numpy(), grid_size, patch_size)

    # Invert the colors: black digits on a white  background
    inverted_image = 1 -  dense_image

    # Resize the  attention grid  to match the original image size
    attention_image = np.kron(attention_grid, np.ones((patch_size, patch_size)))

    # Create a colormap overlay of the attention
    cmap = plt.get_cmap('viridis')
    attention_colormap = cmap(attention_image)
    attention_colormap = np.delete(attention_colormap, 3, 2)  # Remove the alpha channel

    # Convert the original image to RGB 
    original_image_rgb =  np.stack([inverted_image ] *3, axis=-1)

    # Normalize the original image 
    original_image_rgb = (original_image_rgb  - original_image_rgb.min() ) / (original_image_rgb.max()  -  original_image_rgb.min())

    # Color the patches not in the  attention list
    unselected_color = [1, 0, 0] 
    selected_mask = np.zeros_like(original_image_rgb)
    for i in range(grid_size):
        for j  in range(grid_size):
            if (i *grid_size + j) not in first_image_indices:
                attention_colormap[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = unselected_color
            else:
                selected_mask[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = attention_colormap[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

    # Blend the original image with the attention colormap
    overlay_image = 0.6* original_image_rgb + 0.4 * selected_mask

    # Save the image
    overlay_image_uint8 = (overlay_image * 255).astype(np.uint8)
    overlay_image_pil = Image.fromarray(overlay_image_uint8)
    overlay_image_pil.save( save_path)

    # Display the image
    plt.figure(figsize=(15, 15) )
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.title("Attention Map  Overlay on Orginal Image")
    plt.show()

test_data = MegapixelMNIST(conf, train=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

for batch_idx, batch in enumerate(test_loader):
    image_sparse = batch['input']

    mem_idx, attn = get_attention_map(image_sparse.to(device))
    save_path = os.path.join(script_dir, f'attention_map_overlay_{batch_idx}.png')
    visualize_attention(image_sparse[0:1], mem_idx[0:1], attn[0:1], conf,  save_path)
    break
