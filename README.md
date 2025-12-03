# Efficient Noise Calculation in Deep Learning-based MRI Reconstructions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/ICML-2025-red.svg)](https://proceedings.mlr.press/v267/dalmaz25a.html)

Official implementation of the paper **"Efficient Noise Calculation in Deep Learning-based MRI Reconstructions"**, accepted at **ICML 2025**.

**Authors:** Onat Dalmaz, Arjun D. Desai, Reinhard Heckel, Tolga Cukur, Akshay S. Chaudhari, Brian Hargreaves.

---

## 📖 Overview

Accelerated MRI reconstruction is an ill-posed inverse problem where measurement noise propagates into the reconstructed image. While noise analysis is critical for quantifying reliability, it is often absent in Deep Learning (DL) reconstruction due to computational barriers.

This repository provides a **theoretically grounded** and **memory-efficient** framework to compute the **voxel-wise variance** (the diagonal of the noise covariance matrix) for DL-based MRI reconstructions.

### Why use this?

*   **🚀 Efficient:** Reduces compute and memory usage by an order of magnitude compared to Monte-Carlo simulations.
*   **🎯 Accurate:** Provides an unbiased estimator of voxel-wise variance via Jacobian sketching.
*   **🔌 Plug-and-Play:** Integrated with [meddlr](https://github.com/ad12/meddlr); works with supervised, self-supervised, and physics-driven models.

---

## ⚙️ Methodology

We model the propagation of noise covariance $\boldsymbol{\Sigma}$ through a reconstruction network $f_\theta$ via its Jacobian $\mathbf{J}$. The target voxel-wise variance is the diagonal of the posterior covariance:

$$ \text{Var}(\mathbf{x}) = \text{diag}(\mathbf{J} \boldsymbol{\Sigma} \mathbf{J}^H) $$

Since explicitly forming $\mathbf{J}$ is intractable for high-dimensional images, we leverage **Jacobian Vector Products (JVPs)**. We implement this efficiently using `torch.func.jvp` for forward-mode automatic differentiation and `torch.vmap` to vectorize the computation across a batch of random sketching vectors. This allows us to approximate the variance without ever instantiating the full Jacobian matrix.

![Results](methods_brain.png)

*Figure 1: Comparison of uncertainty maps on brain MRI. Our efficient estimator matches the Empirical Monte-Carlo references while requiring significantly fewer resources.*

---

## 🛠 Installation

### Prerequisites

*   Linux or macOS
*   Python 3.8+
*   CUDA-capable GPU (highly recommended)

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/onat-dalmaz/deep_recon_noise.git
    cd meddlr
    ```

2.  **Create a virtual environment:**

    ```bash
    conda create -n meddlr_env python=3.9
    conda activate meddlr_env
    ```

3.  **Install PyTorch:**

    Select the version matching your CUDA setup (e.g., CUDA 11.8):

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Install dependencies and the package:**

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

---

## 🚀 Quick Start

### 1. Training Models

Training scripts are located in `training_bash/`.

**Example: Train E2E-VarNet on Brain Data**

```bash
bash training_bash/run_training_e2e_varnet_brain.sh
```

### 2. Inference with Noise Estimation

To run reconstruction and compute the efficient noise map, use the scripts in `inference_bash/`.

**Example: Run MoDL Inference**

```bash
bash inference_bash/run_inference_modl.sh
```

### 3. Python API Usage

You can also integrate the noise estimator directly into your Python scripts:

```python
import torch
from meddlr.modeling import build_model
from meddlr.config import get_cfg

# 1. Load Configuration
cfg = get_cfg()
cfg.merge_from_file("configs/mri-recon/mridata-3dfse-knee/unrolled.yaml")

# 2. Build Model
model = build_model(cfg)
model.eval()

# 3. Run Reconstruction with Noise Estimation
# (Assuming 'kspace' is your raw data, 'A' is the forward operator, 
# and 'sigma_k' is the noise covariance matrix)
with torch.no_grad():
    # Run standard reconstruction
    output_image = model(inputs)
    
    # Compute voxel-wise noise variance using Jacobian Sketching
    # S: Number of sketching vectors (default=100)
    noise_map, time_taken = model.J_sketch_variance(kspace, A, sigma_k, S=100)
```

---

## 📊 Supported Models

This framework is agnostic to the underlying architecture. We provide configurations for the following:

| Model | Paradigm | Architecture | Description |
|-------|----------|--------------|-------------|
| **E2E-VarNet** | Supervised | Physics-Driven (Unrolled) | End-to-end Variational Network |
| **MoDL** | Supervised | Physics-Driven (Unrolled) | Model-based Deep Learning |
| **U-Net** | Supervised | Data-Driven | Fully Convolutional Network |
| **N2R** | Semi-Supervised | Physics-Driven (Unrolled) | Noise2Recon |
| **VORTEX** | Semi-Supervised | Physics-Driven (Unrolled) | Variable-density Optimized Reconstruction Through EXemplars |
| **SSDU** | Self-Supervised | Physics-Driven (Unrolled) | Self-supervised learning via Data Undersampling |

---

## 📂 Repository Layout

```text
meddlr/
├── configs/              # YAML configs for models and noise settings
├── datasets/             # Data loaders and preprocessing
├── inference_bash/       # Scripts for inference + variance estimation
├── training_bash/        # Scripts for model training
├── meddlr/               # Source code
│   ├── modeling/         # Network architectures & noise layers
│   ├── ops/              # Jacobian vector product utilities
│   └── ...
└── tools/                # Entry points (train_net.py, etc.)
```

---

## 📝 Citation

If you find this code or our paper useful, please cite:

```bibtex
@InProceedings{pmlr-v267-dalmaz25a,
  title     = {Efficient Noise Calculation in Deep Learning-based {MRI} Reconstructions},
  author    = {Dalmaz, Onat and Desai, Arjun D and Heckel, Reinhard and Cukur, Tolga and Chaudhari, Akshay S and Hargreaves, Brian},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = {12280--12313},
  year      = {2025},
  editor    = {Singh, A. and Fazel, M. and Hsu, D. and Lacoste-Julien, S. and Berkenkamp, F. and Maharaj, T. and Wagstaff, K. and Zhu, J.},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v267/dalmaz25a.html}
}
```

---

## ⚖️ License & Patent Notice

*   **Code License:** [MIT License](LICENSE).
*   **⚠️ Patent Notice:** This software is subject to one or more pending patent applications. Patent rights are not granted under the MIT License. Use of this software for commercial purposes may require a separate patent license. Please see the [PATENT_NOTICE](PATENT_NOTICE) file for details.

## 📧 Contact

For questions, issues, or collaboration, please open a GitHub Issue.
