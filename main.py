# Adapted from https://github.com/benbergner/ips.git
import os
import yaml
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import Logger, Struct

from architecture.ips_net import IPSNet
from training.iterative import train_one_epoch, evaluate

# Ensure the script is running from the correct directory
script_dir =  os.path.dirname( os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'mnist_config.yml')

# Assign the GPU to torch
os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Select the data
dataset = 'traffic'  # either one of {'mnist', 'camelyon', 'traffic'}

# get config
with  open(os.path.join('config', dataset + '_config.yml'),"r") as ymlfile:
    c  =  yaml.load(ymlfile, Loader=yaml.FullLoader)
    print("Used config:")
    pprint(c)
    conf = Struct(**c)

# fix the seed for reproducibility
torch.manual_seed(conf.seed)
np.random.seed(conf.seed)

# define datasets and dataloaders (note: no work was done camelyon)
if dataset=='mnist':
    from data.megapixel_mnist.mnist_dataset  import MegapixelMNIST
    train_data = MegapixelMNIST(conf,  train=True)
    test_data = MegapixelMNIST(conf, train=False)
elif dataset == 'traffic':
    from data.traffic.traffic_dataset import TrafficSigns
    train_data = TrafficSigns( conf, train=True)
    test_data = TrafficSigns(conf, train=False)
elif dataset == 'camelyon':
    train_data = CamelyonFeatures(conf, train=True)
    test_data = CamelyonFeatures(conf, train=False)

# Assign both the train and test loaders
train_loader =  DataLoader(train_data, batch_size=conf.B_seq, shuffle=True,
                          num_workers=conf.n_worker, pin_memory = conf.pin_memory, persistent_workers=True)
test_loader =  DataLoader(test_data, batch_size=conf.B_seq, shuffle=False ,
                        num_workers=conf.n_worker, pin_memory=conf.pin_memory, persistent_workers=True)

# Define IPS
net = IPSNet(device, conf).to(device)

# Assign the loss
loss_nll= nn.NLLLoss()
loss_bce = nn.BCELoss()

# define optimizer, lr not important at this point
optimizer = torch.optim.AdamW(net.parameters(), lr=0, weight_decay=conf.wd)

criterions = {}
for task in conf.tasks.values():
    criterions[task['name']] = loss_nll if task['act_fn'] == 'softmax' else loss_bce
# Assing the logger for printing the training statistics
log_writer_train = Logger( conf.tasks )
log_writer_test = Logger(conf.tasks)

# iterate through each epochs
for epoch in  range(conf.n_epoch):

    # Call train_one_epoch from iterative.py
    train_one_epoch(net, criterions, train_loader, optimizer, device, epoch, log_writer_train, conf)
    log_writer_train.compute_metric()

    more_to_print = {'lr': optimizer.param_groups[0]['lr']}
    log_writer_train.print_stats(epoch, train=True, **more_to_print)

    evaluate(net, criterions, test_loader, device, log_writer_test, conf, epoch)
    log_writer_test.compute_metric()
    log_writer_test.print_stats(epoch, train=False)

    # Save weights every 20 epochs for the attention maps (feature not useful otherwise)
    if (epoch + 1) % 20 ==0:  
        weights_path = os.path.join(script_dir, f'model_weights_epoch_{epoch + 1}.pth')
        torch.save(net.state_dict(), weights_path)
