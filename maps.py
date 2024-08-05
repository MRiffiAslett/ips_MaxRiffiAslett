import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from architecture.ips_net import IPSNet
from data.megapixel_mnist.mnist_dataset import MegapixelMNIST
from utils.utils import Struct

# Get the current directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration from YAML file
config_path = os.path.join(script_dir, 'config/mnist_config.yml')
with open(config_path, 'r') as ymlfile:
    c = yaml.load(ymlfile, Loader=yaml.FullLoader)
    conf = Struct(**c)

# Ensure image_size is added to the configuration
conf.image_size = 1500

print(f"Loaded configuration: {conf.__dict__}")

conf.data_dir = os.path.join(script_dir, 'data/megapixel_mnist/dsets/megapixel_mnist_1500')

parameters_path = os.path.join(conf.data_dir, "parameters.json")
print(f"Looking for parameters.json at: {parameters_path}")
if not os.path.exists(parameters_path):
    raise FileNotFoundError(f"{parameters_path} does not exist. Please ensure the file is in the correct location.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = IPSNet(device, conf).to(device)
model_weights_path = os.path.join(script_dir, 'model_weights.pth')
net.load_state_dict(torch.load(model_weights_path))
net.eval()

def get_attention_map(patches):
    print(f"Inside get_attention_map, patches shape: {patches.shape}")
    with torch.no_grad():
        _, _,= net(patches)
        print(f'attention_maps {attention_map.shape}')
        #attention_map = net.transf.get_scores(patches)
        print(f"Attention map shape inside get_attention_map: {attention_map.shape}")
    return attention_map

def save_attention_map(attention_map, conf, filename):
    attention_map = attention_map.cpu().numpy()
    mean_attention = attention_map.mean(axis=1).squeeze()

    attention_image = np.zeros((conf.image_size, conf.image_size, 4), dtype=np.uint8)
    attention_image[:, :, 3] = 0  # Fully transparent background

    patch_size = conf.patch_size[0]
    for idx, alpha in enumerate(mean_attention):
        row = idx // (conf.image_size // patch_size)
        col = idx % (conf.image_size // patch_size)
        color = (255, 0, 0, int(alpha * 255))  # Red with varying transparency
        attention_image[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = (255, 0, 0, int(alpha * 255))

    attention_image = Image.fromarray(attention_image, 'RGBA')
    attention_image.save(filename)

test_data = MegapixelMNIST(conf, train=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

for batch in test_loader:
    image_sparse = batch['input']
    print(f"Image sparse shape: {image_sparse.shape}")

    attention_map = get_attention_map(image_sparse.to(device))
    save_attention_map(attention_map, conf, "/mnt/data/attention_map.png")
    break
