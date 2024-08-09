![My Logo](plots/header.jpg)

<h1 align="center">On the Generalizability of High-Resolution Image Classification with Memory-Efficient Transformers 🧠</h1>

## 📚 Summary
This repository contains the code for my dissertation, which adapts the IPS approach from [benbergner/ips](https://github.com/benbergner/ips.git). IPS is a simple patch-based method that decouples memory consumption from input size, enabling efficient processing of high-resolution images without running out of memory.

We performed repeated experiments with the Megapixel MNIST dataset, sourced from [idiap/attention-sampling](https://github.com/idiap/attention-sampling.git). The experiments varied object-to-image ratio, training size, noise generation strategy, pretraining strategy, and different previously introduced masking strategies to robustify the patch-based image classifier in scenarios with low data and small object-to-image ratios.

## 🏆 Contributions

Our contributions that build upon the implementation by [benbergner/ips](https://github.com/benbergner/ips.git) are as follows:

1. Adding a data generation script `PineneedleMegaMNIST.py` with updated noise and O2I varying strategy.
2. Including semantic and diversity loss features in the `iterative.py` script.
3. Adding stochastic attention masking features in `ips_net.py`.
4. Introducing a new backbone strategy with ResNet-50, freezing all weights until the last layer in `ips_net.py`.
5. Contributed a script for inference producing attention maps.

**Note:** All features can be activated and deactivated via the config files.

## 📁 Repository Structure

### 1.  Architecture
- **`ips_net.py`**: Script where iterative patch selection is performed, including the initial encoding stage.
- **`transformer.py`**: Script defining the multi-head cross-attention pooling operator, attention scorer, and multi-layer perceptron.

### 2. Training
- **`iterative.py`**: Defines the loss functions, initializes batches, and handles training for one epoch.

### 3. Main Script
**`main.py`**: The main script that orchestrates the entire process.

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


- **Pattern:** `results_`[`digit_size_x`]_`[`digit_size_y`]_`[`canvas_size_x`]_`[`canvas_size_y`]_`[`number_of_noise`]_`[`number_of_training_data_points`]_`[`regularization_and_special_feature`]`


**Example:** `results_84_84_3000_3000_400n_1000d_PS_50`
- This denotes 84x84 digit resolution on a 3000x3000 canvas with 400 noise points and 1000 training data points, with a patch size (PS) of 50.

# Noise Generation

<div align="center">
    <img src="plots/Tasksvstasks.jpg" alt="Task vs Tasks" width="500"/>
</div>
<p><strong>Figure 1:</strong> Visual representation of the Needle MNIST task (left) and the Megapixel MNIST dataset (225 × 225) with five noise digits using Bezier curves (right) (source: \parencite{pawlowski_needles_2020}).</p>

## Findings

### Object-to-Image Ratio

<div align="center">
    <img src="plots/task_plot.png" alt="Task Plot" width="1000"/>
</div>
<p><strong>Figure 2:</strong> Results of the experiments on MegaPixel_MNIST with a novel noise generation component. Four object-to-image ratios were tested: {0.008%, 0.034%, 0.078%, 0.13%} across four training dataset sizes {800, 1000, 2000, 4000}. Canvas size and patch size remain fixed at 3000 × 3000 and 50 × 50, respectively, and the O2I changes by varying the digit resolutions to 28 × 28, 56 × 56, 84 × 84, and 112 × 112 pixels. The noise digit thickness is set at 1.925. The model was trained for 100 epochs following the setup of IPS \parencite{bergner_iterative_2023}.</p>

#### Attention Maps

<div align="center">
    <img src="plots/attention_map_1.jpg" alt="Attention Map 1" width="1000"/>
</div>
<p><strong>Figure 3:</strong> Attention maps for different object-to-image ratios: 0.008% (left) and 0.034% (right) on a 1500 × 1500 canvas, with 800 noise digits on the left and 600 on the right. The maps display the top M (100) most informative patches at the end of a full forward pass with IPS. The digit and noise size on the left is 28 × 28 and on the right 56 × 56.</p>

<div align="center">
    <img src="plots/attention_map_2.jpg" alt="Attention Map 2" width="1000"/>
</div>
<p><strong>Figure 4:</strong> Attention maps for different object-to-image ratios: 0.078% (left) and 0.13% (right) on a 1500 × 1500 canvas, with 400 noise digits on the left and 200 on the right. The maps display the top M (100) most informative patches at the end of a full forward pass with IPS. The digit and noise size on the left is 84 × 84 and on the right 112 × 112.</p>

