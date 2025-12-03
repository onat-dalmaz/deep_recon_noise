# Efficient Noise Calculation in Deep Learning-based MRI Reconstructions

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Efficient Noise Calculation in Deep Learning-based MRI Reconstructions"**, accepted as a **Poster** at **ICML 2025**.

**Authors:** Onat Dalmaz · Arjun Desai · Reinhard Heckel · Tolga Cukur · Akshay Chaudhari · Brian Hargreaves

**ICML 2025 Poster:** West Exhibition Hall B2-B3 #W-204  
**Session:** Wed 16 Jul 11 a.m. PDT — 1:30 p.m. PDT

---

## 📋 Abstract

Accelerated MRI reconstruction involves solving an ill-posed inverse problem where noise in acquired data propagates to the reconstructed images. Noise analyses are central to MRI reconstruction for providing an explicit measure of solution fidelity and for guiding the design and deployment of novel reconstruction methods. However, deep learning (DL)-based reconstruction methods have often overlooked noise propagation due to inherent analytical and computational challenges, despite its critical importance. 

This work proposes a theoretically grounded, memory-efficient technique to calculate voxel-wise variance for quantifying uncertainty due to acquisition noise in accelerated MRI reconstructions. Our approach is based on approximating the noise covariance using the DL network's Jacobian, which is intractable to calculate. To circumvent this, we derive an unbiased estimator for the diagonal of this covariance matrix—voxel-wise variance—, and introduce a Jacobian sketching technique to efficiently implement it. 

We evaluate our method on knee and brain MRI datasets for both data-driven and physics-driven networks trained in supervised and unsupervised manners. Compared to empirical references obtained via Monte-Carlo simulations, our technique achieves near-equivalent performance while reducing computational and memory demands by an order of magnitude or more. Furthermore, our method is robust across varying input noise levels, acceleration factors, and diverse undersampling schemes, highlighting its broad applicability. Our work reintroduces accurate and efficient noise analysis as a central tenet of reconstruction algorithms, holding promise to reshape how we evaluate and deploy DL-based MRI.

---

## 🖼️ Method Overview

![Methods Overview](methods_brain.png)

*Overview of our efficient noise calculation method for deep learning-based MRI reconstructions.*

---

## ✨ Key Features

- **Memory-Efficient**: Reduces computational and memory demands by an order of magnitude compared to Monte-Carlo simulations
- **Theoretically Grounded**: Unbiased estimator for voxel-wise variance based on network Jacobian approximation
- **Broad Applicability**: Works with both data-driven and physics-driven networks, trained in supervised and unsupervised manners
- **Robust**: Validated across varying noise levels, acceleration factors, and undersampling schemes
- **Easy to Use**: Integrated into the meddlr framework with simple configuration files

---

## 🚀 Installation

### Prerequisites

- Python 3.6 or higher
- CUDA-capable GPU (recommended)
- Conda (recommended for environment management)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd meddlr
   ```

2. **Create and activate a conda environment:**
   ```bash
   conda create -n meddlr_env python=3.9
   conda activate meddlr_env
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   # For CUDA 10.1 (adjust version as needed)
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   
   # For other CUDA versions, visit: https://pytorch.org/get-started/locally/
   ```

4. **Install optional GPU-accelerated libraries:**
   ```bash
   pip install cupy-cuda101  # Adjust version (e.g., cupy-cuda111) based on your CUDA version
   ```

5. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

---

## 📁 Project Structure

```
meddlr/
├── configs/              # Configuration files for different models & datasets
│   └── mri-recon/       # MRI reconstruction configurations
│       ├── fastmri-brain/
│       └── mridata-3dfse-knee/
├── datasets/            # Dataset processing scripts and metadata
├── inference_bash/      # Bash scripts for running inference with noise calculation
├── training_bash/       # Bash scripts for training models
├── tools/              # Utility scripts (training, evaluation)
├── meddlr/             # Core implementation
│   ├── modeling/       # Model architectures
│   ├── transforms/     # Data transforms
│   ├── metrics/        # Evaluation metrics
│   └── ...
├── tests/              # Unit tests
├── docs/               # Documentation
└── README.md           # This file
```

---

## 🎯 Quick Start

### Training a Model

Training scripts are available in the `training_bash/` directory. To train a specific model:

```bash
bash training_bash/run_training_{model_name}.sh
```

**Examples:**
- Train **E2E-VarNet** on brain dataset:
  ```bash
  bash training_bash/run_training_e2e_varnet_brain.sh
  ```

- Train **MoDL** on knee dataset:
  ```bash
  bash training_bash/run_training_modl.sh
  ```

### Running Inference with Noise Calculation

To run inference on a trained model and calculate voxel-wise variance:

```bash
bash inference_bash/run_inference_{model_name}.sh
```

**Examples:**
- Run inference for **MoDL**:
  ```bash
  bash inference_bash/run_inference_modl.sh
  ```

- Run inference for **E2E-VarNet** on brain:
  ```bash
  bash inference_bash/run_inference_e2e_varnet_brain.sh
  ```

### Configuration

Model and training configurations are stored in the `configs/` directory as YAML files. These files define:
- Model architecture and hyperparameters
- Dataset paths and preprocessing
- Training settings (optimizer, learning rate, etc.)
- Noise calculation parameters

---

## 📊 Supported Models

This implementation supports various deep learning-based MRI reconstruction models:

- **E2E-VarNet**: End-to-end variational network
- **MoDL**: Model-based deep learning
- **N2R**: Noise2Recon
- **SSDU**: Self-supervised learning via data undersampling
- **VORTEX**: Variable-density Optimized Reconstruction Through EXemplars

All models can be trained and evaluated with our efficient noise calculation method.

---

## 📖 Citation

If you use this code in your research, please cite our ICML 2025 paper:

```bibtex
@inproceedings{dalmaz2025efficient,
  title={Efficient Noise Calculation in Deep Learning-based MRI Reconstructions},
  author={Dalmaz, Onat and Desai, Arjun and Heckel, Reinhard and Cukur, Tolga and Chaudhari, Akshay and Hargreaves, Brian},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

---

## 👥 Authors

- **Onat Dalmaz**
- **Arjun Desai**
- **Reinhard Heckel**
- **Tolga Cukur**
- **Akshay Chaudhari**
- **Brian Hargreaves**

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work is built on top of the [meddlr](https://github.com/ad12/meddlr) framework for medical image reconstruction. We thank the meddlr team for their excellent framework.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

## 🔗 Related Links

- [ICML 2025](https://icml.cc/) - International Conference on Machine Learning
- [meddlr Framework](https://github.com/ad12/meddlr) - Base framework for medical image reconstruction
