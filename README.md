![My Logo](plots/header.jpg)

<h1 align="left">
    On the Generalizability <br>
    of Iterative Patch Selection <br>
    for High-Resolution Image Classification
</h1>


## 📚 Summary
This repository contains the code for my dissertation, which builds upon and experiments with IPS [benbergner/ips](https://github.com/benbergner/ips.git). IPS is a state-of-the-art memory-efficient and weakly supervised patch-based classifier. It saves memory consumption by iterating through each patch and keeping only the most M most salient in memory. The implementation of [benbergner/ips](https://github.com/benbergner/ips.git) employs a Multi-head cross-attention mechanism as a pooling operator to aggregate the M  most salient patches into a bag-level representation.

We performed repeated experiments with the Megapixel MNIST dataset and Swedish traffic sign datasets [idiap/attention-sampling](https://github.com/idiap/attention-sampling.git), in which we found that in low data scenarios, generalizability suffers as the object-to-image (O2I) ratio decreases. To robustify these scenarios, we perform repeated experiments  tuning the patch size and pretraining strategy. Our work identifies that as the object-to-image ratio decreases, performance suffers in low data settings for IPS (Bergner, Lippert, and Mahendran 2023). We find that this vulnerability can be mitigated by tuning the patch size.

**Note:** All features can be activated and deactivated via the config files.

## 📁 Repository Structure

### 1.  Architecture
- **`ips_net.py`**: Script where iterative patch selection is performed, including the initial encoding stage.
- **`transformer.py`**: Script defining the multi-head cross-attention pooling operator, attention scorer, and multi-layer perceptron.

### 2. Training
- **`iterative.py`**: Defines the loss functions, initializes batches, and handles training for one epoch.

### 3. Main Script
- **`main.py`**: The main script that orchestrates the entire process.

### 4.  Configuration Files
- **`mnist_config.yml`**: Configuration file for the MNIST dataset.
- **`camelyon_config.yml`**: Configuration file for the Camelyon dataset.
- **`traffic_config.yml`**: Configuration file for the Traffic dataset.

### 5.  Data Processing
**Megapixel MNIST**
- **`make_mnist.py`**: Original MegaMNIST generation script.
- **`PineneedleMegaMNIST.py`**: Custom data generation script with new Bezier noise and Object-to-Image (O2I) setup.
- **`mnist_dataset.py`**: Script to preprocess and patchify the data.

### 6. Utilities
- **`utils.py`**: Contains functions for logging memory, adjusting the learning rate, and printing statistics.


### 7. Results Library
The results are organized into the following categories:
- **`O2I_datasize`**: Results related to Object-to-Image (O2I) ratio and dataset size.
- **`Semantic_Diversity_Regularisation`**: Results related to semantic diversity regularization.
- **`Attention_Masking`**: Results related to attention masking.
- **`Dataset_Size`**: Results focusing on different dataset sizes.
- **`Noise_Size`**: Results focusing on the size and impact of noise.
- **`Digit_Thickness`**: Results related to the thickness of digits.
- **`Backbones`**: Results related to different backbone architectures.
- **`Patch_Size`**: Results focusing on the size of patches used in training.

**Note:** The naming convention for results is as follows:


**Example:** `results_84_84_3000_3000_400n_1000d_PS_50`
- This denotes 84x84 digit resolution on a 3000x3000 canvas with 400 noise points and 1000 training data points, with a patch size of 50.

## Swedish traffic Signs data set

<div align="center">
    <img src="plots/attention_maps_traffic_Github.jpg" alt="Attention Map 1" width="1000"/>
</div>
<p><strong>Figure 3:</strong> Attention maps for image 63 (validation set) of the Swedish traffic Signs data set. IPS was run for 140 epochs with a patch size and stride of 25</p>

## Megapixel MNSIT
<div align="center">
    <img src="plots/attention_map_1_black.jpg" alt="Attention Map 1" width="1000"/>
</div>
<p><strong>Figure 3:</strong> Attention maps for different object-to-image ratios: 0.008% (left) and 0.034% (right) on a 1500 × 1500 canvas, with 800 noise digits on the left and 600 on the right. The maps display the top M (100) most informative patches at the end of a full forward pass with IPS. The digit and noise size on the left is 28 × 28 and on the right 56 × 56.</p>

<div align="center">
    <img src="plots/attention_map_2_black.jpg" alt="Attention Map 2" width="1000"/>
</div>
<p><strong>Figure 4:</strong> Attention maps for different object-to-image ratios: 0.078% (left) and 0.13% (right) on a 1500 × 1500 canvas, with 400 noise digits on the left and 200 on the right. The maps display the top M (100) most informative patches at the end of a full forward pass with IPS. The digit and noise size on the left is 84 × 84 and on the right 112 × 112.</p>

