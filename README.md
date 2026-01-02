# 🤖 ACT: Action Chunking with Transformers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Implementation and experiments with Action Chunking Transformers for robot manipulation tasks.

<p align="center">
  <img src="assets/demo.gif" alt="ACT Demo" width="600"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Experiments](#-experiments)
- [Key Learnings](#-key-learnings)
- [References](#-references)

---

## 🎯 Overview

### Problem
Imitation learning for robot manipulation suffers from **compounding errors** - small mistakes accumulate over time, causing task failures.

### Solution
**Action Chunking with Transformers (ACT)** predicts a sequence of `k` future actions instead of single actions, reducing error accumulation and improving task success rates.

### Key Features
- 🔄 **Action Chunking**: Predict 100 future actions at once
- 🧠 **Transformer Architecture**: Encoder-decoder with self-attention
- 📊 **CVAE**: Handle multimodal human demonstrations
- ⚡ **Temporal Ensemble**: Smooth action execution

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="ACT Architecture" width="800"/>
</p>

### Components

| Component | Description |
|-----------|-------------|
| **Image Encoder** | ResNet18 backbone for visual feature extraction |
| **CVAE Encoder** | Encodes action sequences to latent space `z` (training only) |
| **Transformer Encoder** | Processes observations with self-attention |
| **Transformer Decoder** | Predicts action chunk from encoded observations |

### Model Configuration

```python
config = {
    "chunk_size": 100,        # Number of future actions to predict
    "hidden_dim": 512,        # Transformer hidden dimension
    "dim_feedforward": 3200,  # Feed-forward dimension
    "num_encoder_layers": 4,  # Transformer encoder layers
    "num_decoder_layers": 7,  # Transformer decoder layers
    "num_heads": 8,           # Attention heads
    "dropout": 0.1,
}
```

---

## 📊 Results

### Transfer Cube Task

| Experiment | Chunk Size | KL Weight | Success Rate | Notes |
|------------|------------|-----------|--------------|-------|
| Baseline | 100 | 10 | **XX%** | Default configuration |
| Small Chunk | 50 | 10 | XX% | More frequent replanning |
| Large Chunk | 150 | 10 | XX% | Longer action sequences |
| Low KL | 100 | 1 | XX% | More reconstruction focus |
| High KL | 100 | 50 | XX% | More regularization |

### Temporal Ensemble Ablation

| Setting | Success Rate | Smoothness |
|---------|--------------|------------|
| Without Temporal Ensemble | XX% | Jerky |
| With Temporal Ensemble | **XX%** | Smooth |

### Training Curves

<p align="center">
  <img src="assets/training_loss.png" alt="Training Loss" width="400"/>
  <img src="assets/success_rate.png" alt="Success Rate" width="400"/>
</p>

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x+ (for GPU training)
- ~20GB disk space

### Setup

```bash
# Clone repository
git clone https://github.com/[your-username]/act-project.git
cd act-project

# Create conda environment
conda create -n act python=3.8.10
conda activate act

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install DETR
cd detr && pip install -e . && cd ..
```

### Verify Installation

```bash
python -c "import mujoco; print('MuJoCo OK')"
python -c "from policy import ACTPolicy; print('ACT OK')"
```

---

## 🚀 Usage

### 1. Generate Demo Data

```bash
python record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir ./data/sim_transfer_cube \
    --num_episodes 50
```

### 2. Train Model

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./checkpoints/my_experiment \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0
```

### 3. Evaluate

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./checkpoints/my_experiment \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --eval \
    --temporal_agg
```

### 4. Visualize

```bash
# Add --onscreen_render for real-time visualization
python imitate_episodes.py [...] --eval --onscreen_render
```

---

## 🔬 Experiments

### Experiment 1: Chunk Size Ablation

**Hypothesis**: Larger chunk sizes provide more temporal consistency but may reduce reactivity.

```bash
# Small chunk
python imitate_episodes.py [...] --chunk_size 50

# Large chunk  
python imitate_episodes.py [...] --chunk_size 150
```

**Finding**: [Your observation here]

---

### Experiment 2: KL Weight Ablation

**Hypothesis**: KL weight balances reconstruction accuracy vs. latent regularization.

```bash
# Low KL (reconstruction focus)
python imitate_episodes.py [...] --kl_weight 1

# High KL (regularization focus)
python imitate_episodes.py [...] --kl_weight 50
```

**Finding**: [Your observation here]

---

### Experiment 3: Temporal Ensemble

**Hypothesis**: Averaging overlapping action predictions improves smoothness.

```bash
# Without temporal ensemble
python imitate_episodes.py [...] --eval

# With temporal ensemble
python imitate_episodes.py [...] --eval --temporal_agg
```

**Finding**: [Your observation here]

---

## 📚 Key Learnings

### 1. Why Transformer over LSTM?

| Aspect | LSTM | Transformer |
|--------|------|-------------|
| Processing | Sequential | Parallel |
| Long-range | Struggles | Self-attention handles well |
| Training | Slow | Fast (parallelizable) |
| Scaling | Limited | Scales with data & compute |

### 2. Action Chunking Benefits

```
Traditional (single action):
t=0: predict a₀ → small error ε
t=1: predict a₁ from wrong state → error grows
t=T: accumulated error = T × ε (compounding)

Action Chunking (k actions):
t=0: predict [a₀, a₁, ..., a₉₉] → consistent sequence
t=100: replan with fresh observation
Error accumulation reduced by factor of k
```

### 3. CVAE for Multimodal Demonstrations

- **Problem**: Human demonstrations have style variations
- **Solution**: CVAE encodes "style" into latent `z`
- **Training**: Learn distribution of `z` from action sequences
- **Inference**: Use `z = 0` (mean) for consistent behavior

### 4. Self-Attention Mechanism

```python
# Core attention formula
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

# Self-attention: Q = K = V from same sequence
# Cross-attention: Q from decoder, K & V from encoder
```

---

## 📁 Project Structure

```
act-project/
├── README.md
├── requirements.txt
├── policy.py              # ACT policy implementation
├── imitate_episodes.py    # Training script
├── record_sim_episodes.py # Data collection
├── detr/                  # DETR transformer modules
│   └── models/
│       └── transformer.py
├── data/                  # Demo datasets
│   └── sim_transfer_cube/
├── checkpoints/           # Trained models
│   ├── baseline/
│   └── best/
├── assets/                # Images, diagrams
│   ├── architecture.png
│   ├── demo.gif
│   └── training_loss.png
└── experiments/           # Experiment logs
    └── results.md
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` to 4 or 2 |
| MuJoCo not found | `pip install mujoco==2.3.7` |
| dm_control error | Check MuJoCo version compatibility |
| Training loss NaN | Reduce learning rate to `1e-6` |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 16GB+ |
| RAM | 16GB | 32GB |
| Storage | 20GB | 50GB |

---

## 📖 References

### Papers

- **ACT**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (RSS 2023)
- **Transformer**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
- **VAE**: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (ICLR 2014)

### Resources

- [Official ACT Repository](https://github.com/tonyzhaozh/act)
- [ALOHA Project Website](https://tonyzhaozh.github.io/aloha/)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 🙏 Acknowledgments

- Original ACT implementation by [Tony Z. Zhao et al.](https://github.com/tonyzhaozh/act)
- Stanford IRIS Lab for the ALOHA platform
- [Your additional acknowledgments]

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Cong Thai Luu**

- GitHub: [Cong Thai](https://github.com/zerokhong1)
- LinkedIn: [Luu Thai](https://www.linkedin.com/in/congthaineur/)
- Email: thailuucong.hust@gmail.com

---

<p align="center">
  <i>Built as part of AIO Conquer 2025 </i>
</p>
