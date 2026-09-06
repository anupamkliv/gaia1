# GAIA-1 PyTorch Reconstruction

A modular PyTorch reconstruction of **GAIA-1: A Generative World Model for Autonomous Driving**, based on the architecture and training methodology described in the original paper.

> **Note:** This is an independent research implementation based on publicly available information from the GAIA-1 paper. It is **not the official Wayve implementation**. Architectural details that are not disclosed in the paper are implemented using reasonable design choices and are clearly identified as such.

---

 Repository Structure

```text
gaia1/
│
├── configs/
│   ├── dev.yaml
│   └── gaia1_scale_approx.yaml
│
├── gaia1/
│   │
│   ├── tokenizer/
│   │   ├── model.py
│   │   ├── blocks.py
│   │   └── quantizer.py
│   │
│   ├── world_model/
│   │   ├── model.py
│   │   ├── blocks.py
│   │   ├── multimodal.py
│   │   ├── positional.py
│   │   ├── packing.py
│   │   └── generation.py
│   │
│   ├── diffusion/
│   │   ├── video_decoder.py
│   │   ├── blocks.py
│   │   ├── schedule.py
│   │   ├── tasks.py
│   │   ├── loss.py
│   │   └── sampling.py
│   │
│   ├── losses/
│   │   ├── tokenizer_losses.py
│   │   └── dino_teacher.py
│   │
│   ├── data/
│   │   └── dataset.py
│   │
│   └── utils/
│       ├── config.py
│       └── ema.py
│
├── scripts/
│   ├── train_tokenizer.py
│   ├── train_world_model.py
│   ├── train_video_decoder.py
│   └── rollout.py
│
├── tests/
│   └── test_shapes.py
│
├── ARCHITECTURE_NOTES.md
├── RECONSTRUCTION_STATUS.json
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <https://github.com/anupamkliv/gaia1>
cd gaia1
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

* PyTorch
* torchvision
* Hugging Face Transformers
* einops
* PyYAML
* Pillow
* NumPy

Large-scale distributed training may additionally require:

* FlashAttention
* DeepSpeed
* CUDA-compatible distributed training infrastructure

---

# Dataset Format

The world-model dataset is expected to contain driving sequences with synchronized video, text, speed, and curvature.

Example JSONL entry:

```json
{
  "frames": [
    "/data/sequence01/frame000.jpg",
    "/data/sequence01/frame001.jpg"
  ],
  "texts": [
    "Driving along a residential road",
    "Approaching a parked vehicle"
  ],
  "speed": [
    8.2,
    8.0
  ],
  "curvature": [
    0.001,
    0.002
  ]
}
```

A complete training sequence can contain up to 26 timesteps.

---

# Training

Training is divided into three stages.

## Stage 1: Train the Image Tokenizer

```bash
python scripts/train_tokenizer.py \
    --config configs/dev.yaml \
    --data /path/to/images \
    --out checkpoints/tokenizer.pt
```

After training, the tokenizer converts every frame into an \(18\times32\) grid of discrete tokens.

---

## Stage 2: Train the World Model

```bash
python scripts/train_world_model.py \
    --config configs/dev.yaml \
    --manifest /path/to/train.jsonl \
    --tokenizer_ckpt checkpoints/tokenizer.pt \
    --out checkpoints/world.pt
```

The trained tokenizer is frozen while training the world model.

Conceptually:

```text
Video
  │
  ▼
Frozen Tokenizer
  │
  ▼
Image Tokens ───────────┐
                        │
Text ───────────────────┤
                        ├──► World Model
Speed ──────────────────┤
                        │
Curvature ──────────────┘
                        │
                        ▼
                 Future Tokens
```

---

## Stage 3: Train the Video Diffusion Decoder

```bash
python scripts/train_video_decoder.py \
    --config configs/dev.yaml \
    --manifest /path/to/train.jsonl \
    --tokenizer_ckpt checkpoints/tokenizer.pt \
    --out checkpoints/decoder.pt
```

The decoder learns to transform discrete visual-token sequences into RGB video.

---

# Inference

A complete rollout can be generated using:

```bash
python scripts/rollout.py \
    --config configs/dev.yaml \
    --tokenizer_ckpt checkpoints/tokenizer.pt \
    --world_ckpt checkpoints/world.pt \
    --decoder_ckpt checkpoints/decoder.pt \
    --input_clip example.pt \
    --out generated.pt
```

The inference pipeline is:

```text
Observed Video
      │
      ▼
Image Tokenizer
      │
      ▼
Context Tokens
      │
      │      Text
      │      Speed
      │      Curvature
      │        │
      └────────┼──────► World Model
               │
               ▼
        Future Image Tokens
               │
               ▼
      Video Diffusion Decoder
               │
               ▼
        Future RGB Video
```

---



## Overview

GAIA-1 is a generative world model for autonomous driving that learns to predict future driving scenes from:

* Camera observations
* Natural-language descriptions
* Vehicle speed
* Vehicle curvature

Instead of explicitly learning a BEV representation, semantic segmentation, or supervised depth map, GAIA-1 learns a discrete visual representation and models how this representation evolves over time.

The overall architecture consists of three major components:

```text
                  ┌──────────────────┐
                  │   Driving Video  │
                  └────────┬─────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Image Tokenizer   │
                │                     │
                │ RGB → VQ Tokens     │
                └──────────┬──────────┘
                           │
                           │ 576 image tokens/frame
                           ▼

     Text ───────┐
                 │
     Speed ──────┼────► Autoregressive
                 │       World Model
     Curvature ──┘             │
                               │
                               ▼
                     Future Image Tokens
                               │
                               ▼
                  ┌──────────────────────┐
                  │   Video Diffusion    │
                  │       Decoder        │
                  └──────────┬───────────┘
                             │
                             ▼
                       Generated Video
```

The implementation is divided accordingly into:

1. **Image Tokenizer**
2. **Autoregressive World Model**
3. **Video Diffusion Decoder**

---

# 1. Image Tokenizer

The tokenizer converts RGB driving images into a compact sequence of discrete visual tokens.

```text
RGB Image
288 × 512
    │
    ▼
2D Encoder
    │
    ▼
Continuous Latent
18 × 32 × C
    │
    ▼
Vector Quantization
Codebook K = 8192
    │
    ▼
18 × 32 discrete IDs
    │
    ▼
576 image tokens
```

For an input image

$$
x \in \mathbb{R}^{3\times288\times512},
$$

the encoder produces a spatial latent representation

$$
z_e(x)\in\mathbb{R}^{C\times18\times32}.
$$

The spatial compression factor is therefore

$$
D=16.
$$

Each spatial location is mapped to its nearest vector in a learned codebook containing

$$
K=8192
$$

entries.

The final image representation contains

$$
18\times32=576
$$

discrete tokens per frame.

### Semantic feature learning

GAIA-1 additionally uses pretrained DINO visual representations to encourage semantic information to be retained by the tokenizer.

Conceptually:

```text
                 RGB Image
                 /       \
                /         \
               ▼           ▼
        VQ Encoder       DINO
               │           │
               ▼           ▼
        VQ Features   Semantic Features
               │           │
               └─────┬─────┘
                     │
                Cosine Loss
```

This encourages the discrete representation to preserve semantic structure rather than optimizing only for pixel reconstruction.

The tokenizer objective combines reconstruction, perceptual, adversarial, codebook, and semantic-distillation losses.


# 2. Multimodal Representation

At every timestep, GAIA-1 combines three modalities.

### Text

Natural-language conditioning is encoded using **T5-Large**.

A maximum of

$$
32
$$

text tokens is used per timestep.

### Image

Each frame contributes

$$
576
$$

discrete image tokens.

### Vehicle actions

Two vehicle-state quantities are represented:

* Speed
* Curvature

Each scalar is independently projected into the common model embedding space.

Therefore, one timestep contains

$$
32+576+2=610
$$

tokens.

The ordering is

```text
TEXT → IMAGE → ACTION
```

or

```text
[c1 ... c32]
[z1 ... z576]
[speed]
[curvature]
```

---

# 3. Spatio-Temporal Positional Encoding

GAIA-1 needs to understand both:

* **when** a token occurs
* **where** the token occurs within a timestep

Conceptually:

```text
Token Embedding
      +
Temporal Embedding
"Which frame?"
      +
Spatial Embedding
"Where inside the timestep?"
      │
      ▼
Transformer Input
```

For the reported large model, the common embedding dimension is

$$
d=4096.
$$

---

# 4. Autoregressive World Model

The core of GAIA-1 is a causal Transformer that predicts future visual tokens.

For multiple timesteps, the sequence becomes

```text
C1 Z1 A1 | C2 Z2 A2 | C3 Z3 A3 | ... | CT ZT AT
```

where

* \(C_t\): text tokens
* \(Z_t\): image tokens
* \(A_t\): vehicle action tokens

The paper uses

$$
T=26
$$

timesteps at approximately

$$
6.25\text{ Hz}.
$$

Therefore, the complete Transformer sequence contains

$$
26\times610=15,860
$$

positions.

The reported world model contains approximately **6.5 billion parameters**.

---

## Autoregressive objective

The model predicts every image token conditioned on everything it is causally allowed to observe.

The loss is applied only to **image-token predictions**.

Text and vehicle-action embeddings provide conditioning information but are not treated as image-token prediction targets.

---

# 5. Autoregressive Generation

During inference, future visual tokens are generated sequentially.

```text
Driving Context
      │
      ▼
World Model
      │
      ▼
P(z1 | context)
      │
    sample
      ▼
     z1
      │
      ▼
P(z2 | context, z1)
      │
    sample
      ▼
     z2
      │
      .
      .
      .
      ▼
    z576
      │
      ▼
Future Frame Representation
```

The implementation supports:

* Temperature sampling
* Top-k sampling
* Conditional generation
* Unconditional generation
* Classifier-free guidance

Top-k sampling restricts generation to the most probable visual-token candidates.

---

# 6. Conditioning

The world model can learn several generation modes.

The training distribution described in GAIA-1 uses:

| Conditioning  | Probability |
| ------------- | ----------: |
| Unconditional |         20% |
| Action        |         40% |
| Text          |         40% |

This allows a single model to perform:

```text
             GAIA-1
                │
       ┌────────┼────────┐
       │        │        │
       ▼        ▼        ▼
   Unconditional Action   Text
    Generation  Control  Control
```

Vehicle actions can therefore influence how the predicted world evolves.

---

# 7. Video Diffusion Decoder

The world model predicts **discrete visual tokens**, not final RGB pixels.

A video diffusion decoder converts these tokens back into temporally consistent video.

```text
Predicted Image Tokens
          │
          ▼
Token Conditioning
          │
          ▼
Noisy Video
          │
          ▼
      3D U-Net
          │
     ┌────┴────┐
     ▼         ▼
 Spatial    Temporal
Attention   Attention
     │         │
     └────┬────┘
          │
          ▼
     v Prediction
          │
          ▼
Diffusion Sampling
          │
          ▼
      RGB Video
```

The paper reports a video decoder with approximately **2.6 billion parameters**.

The decoder uses a 3D U-Net with factorized:

* Spatial attention
* Temporal attention

This allows the model to maintain both spatial image quality and temporal consistency.

---

# 8. Video Decoder Training Tasks

The decoder is trained using four tasks.

### Image generation

Generate an RGB image from image-token conditioning.

### Video generation

Generate temporally consistent video from a sequence of image tokens.

### Autoregressive decoding

Previously decoded RGB frames provide temporal context for generating subsequent frames.

### Video interpolation

Intermediate frames are generated from surrounding temporal context.

These tasks are sampled with equal probability during training.

---

# 9. Diffusion Objective

The decoder uses a cosine diffusion noise schedule and \(v\)-parameterization.

For

$$
x_t=
\alpha_t x_0+
\sigma_t\epsilon,
$$

the velocity target is

$$
v_t=
\alpha_t\epsilon-
\sigma_t x_0.
$$

The reconstructed implementation uses the weighted diffusion objective

---

# Repository Structure

```text
gaia1_reconstruction/
│
├── configs/
│   ├── dev.yaml
│   └── gaia1_scale_approx.yaml
│
├── gaia1/
│   │
│   ├── tokenizer/
│   │   ├── model.py
│   │   ├── blocks.py
│   │   └── quantizer.py
│   │
│   ├── world_model/
│   │   ├── model.py
│   │   ├── blocks.py
│   │   ├── multimodal.py
│   │   ├── positional.py
│   │   ├── packing.py
│   │   └── generation.py
│   │
│   ├── diffusion/
│   │   ├── video_decoder.py
│   │   ├── blocks.py
│   │   ├── schedule.py
│   │   ├── tasks.py
│   │   ├── loss.py
│   │   └── sampling.py
│   │
│   ├── losses/
│   │   ├── tokenizer_losses.py
│   │   └── dino_teacher.py
│   │
│   ├── data/
│   │   └── dataset.py
│   │
│   └── utils/
│       ├── config.py
│       └── ema.py
│
├── scripts/
│   ├── train_tokenizer.py
│   ├── train_world_model.py
│   ├── train_video_decoder.py
│   └── rollout.py
│
├── tests/
│   └── test_shapes.py
│
├── ARCHITECTURE_NOTES.md
├── RECONSTRUCTION_STATUS.json
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <https://github.com/anupamkliv/gaia1>
cd gaia1_reconstruction
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

* PyTorch
* torchvision
* Hugging Face Transformers
* einops
* PyYAML
* Pillow
* NumPy

Large-scale distributed training may additionally require:

* FlashAttention
* DeepSpeed
* CUDA-compatible distributed training infrastructure

---

# Dataset Format

The world-model dataset is expected to contain driving sequences with synchronized video, text, speed, and curvature.

Example JSONL entry:

```json
{
  "frames": [
    "/data/sequence01/frame000.jpg",
    "/data/sequence01/frame001.jpg"
  ],
  "texts": [
    "Driving along a residential road",
    "Approaching a parked vehicle"
  ],
  "speed": [
    8.2,
    8.0
  ],
  "curvature": [
    0.001,
    0.002
  ]
}
```

A complete training sequence can contain up to 26 timesteps.

---

# Training

Training is divided into three stages.

## Stage 1: Train the Image Tokenizer

```bash
python scripts/train_tokenizer.py \
    --config configs/dev.yaml \
    --data /path/to/images \
    --out checkpoints/tokenizer.pt
```

After training, the tokenizer converts every frame into an \(18\times32\) grid of discrete tokens.

---

## Stage 2: Train the World Model

```bash
python scripts/train_world_model.py \
    --config configs/dev.yaml \
    --manifest /path/to/train.jsonl \
    --tokenizer_ckpt checkpoints/tokenizer.pt \
    --out checkpoints/world.pt
```

The trained tokenizer is frozen while training the world model.

Conceptually:

```text
Video
  │
  ▼
Frozen Tokenizer
  │
  ▼
Image Tokens ───────────┐
                        │
Text ───────────────────┤
                        ├──► World Model
Speed ──────────────────┤
                        │
Curvature ──────────────┘
                        │
                        ▼
                 Future Tokens
```

---

## Stage 3: Train the Video Diffusion Decoder

```bash
python scripts/train_video_decoder.py \
    --config configs/dev.yaml \
    --manifest /path/to/train.jsonl \
    --tokenizer_ckpt checkpoints/tokenizer.pt \
    --out checkpoints/decoder.pt
```

The decoder learns to transform discrete visual-token sequences into RGB video.

---

# Inference

A complete rollout can be generated using:

```bash
python scripts/rollout.py \
    --config configs/dev.yaml \
    --tokenizer_ckpt checkpoints/tokenizer.pt \
    --world_ckpt checkpoints/world.pt \
    --decoder_ckpt checkpoints/decoder.pt \
    --input_clip example.pt \
    --out generated.pt
```

The inference pipeline is:

```text
Observed Video
      │
      ▼
Image Tokenizer
      │
      ▼
Context Tokens
      │
      │      Text
      │      Speed
      │      Curvature
      │        │
      └────────┼──────► World Model
               │
               ▼
        Future Image Tokens
               │
               ▼
      Video Diffusion Decoder
               │
               ▼
        Future RGB Video
```

---

# Development vs Full-Scale Configuration

Two configurations are provided.

### `configs/dev.yaml`

A reduced model intended for:

* Architecture debugging
* Unit testing
* Tensor-shape validation
* Small-scale experimentation
* Understanding the GAIA-1 pipeline

### `configs/gaia1_scale_approx.yaml`

Preserves the major dimensions reported in the paper, including:

```yaml
hidden_dim: 4096
codebook_size: 8192
image_tokens: 576
text_tokens: 32
action_tokens: 2
max_timesteps: 26
```

Some parameters such as Transformer depth and number of attention heads are reconstruction choices because they are not explicitly reported in the paper.

---

# Paper Specification vs Reconstruction

A central goal of this repository is to avoid presenting undocumented architectural assumptions as facts.

## Explicitly described in GAIA-1

The implementation preserves the reported:

* \(288\times512\) image resolution
* \(16\times\) tokenizer spatial compression
* \(18\times32\) visual-token grid
* 576 visual tokens per frame
* 8192-entry codebook
* T5-Large text encoder
* 32 text tokens per timestep
* Speed and curvature conditioning
* Text → Image → Action ordering
* 4096-dimensional common representation
* Factorized positional embeddings
* Causal autoregressive visual-token prediction
* 26 world-model timesteps
* 6.25 Hz world-model rate
* 15,860-token sequence
* Conditioning dropout strategy
* Top-k autoregressive sampling
* 3D U-Net video decoder
* Factorized spatial and temporal attention
* Multi-task decoder training
* \(v\)-parameterized diffusion
* Cosine diffusion schedule

## Reconstruction choices

The paper does not provide every implementation detail, including the exact:

* Transformer layer count
* Number of attention heads
* Feed-forward architecture
* Normalization implementation
* Tokenizer channel schedule
* Number of tokenizer residual blocks
* GAN discriminator architecture
* Perceptual-loss network
* DINO model variant and extraction layer
* Video U-Net channel schedule
* Number and location of video attention blocks
* Detailed interpolation masks
* DDIM discretization details

These components are implemented as reasonable research choices and are configurable.

See `ARCHITECTURE_NOTES.md` and `RECONSTRUCTION_STATUS.json` for details.

---

# Training Supervision

One interesting property of GAIA-1 is what the core architecture **does not require**.

The training pipeline does not rely on explicit:

```text
Depth ground truth            ✗
BEV semantic segmentation     ✗
3D bounding-box labels        ✗
Object detection labels       ✗
HD-map labels                 ✗
```

Instead, the world representation is learned primarily from:

```text
RGB video
   +
temporal evolution
   +
DINO visual representations
   +
text
   +
vehicle speed
   +
vehicle curvature
```

This makes GAIA-1 particularly interesting for research into self-supervised world models and reducing expensive geometric or BEV annotation requirements.

It should not, however, be interpreted as explicit supervised metric-depth estimation. Geometric structure is learned indirectly through the visual and temporal prediction objectives.

---

# Computational Requirements

The original GAIA-1 system is extremely large.

Reported model scales are approximately:

| Component               | Parameters |
| ----------------------- | ---------: |
| Image tokenizer         |       0.3B |
| World model             |       6.5B |
| Video diffusion decoder |       2.6B |

The complete system is therefore far beyond normal single-GPU training.

Use `configs/dev.yaml` when developing or studying the architecture.

Full-scale experimentation requires distributed GPU infrastructure, mixed precision, activation checkpointing, memory-efficient attention, and distributed optimizer/model-state management.

---

# Current Status

The repository currently provides:

* [x] Image tokenizer architecture
* [x] Vector quantization
* [x] 8192-token visual vocabulary support
* [x] DINO distillation interface
* [x] T5 text conditioning
* [x] Speed conditioning
* [x] Curvature conditioning
* [x] Multimodal sequence construction
* [x] Spatio-temporal positional embeddings
* [x] Causal Transformer world model
* [x] Image-token cross-entropy objective
* [x] Conditioning dropout
* [x] Top-k generation
* [x] Classifier-free guidance support
* [x] 3D video U-Net
* [x] Spatial attention
* [x] Temporal attention
* [x] Cosine diffusion schedule
* [x] \(v\)-prediction
* [x] Decoder multi-task framework
* [x] DDIM-style sampling
* [x] End-to-end rollout pipeline
* [x] Development configuration
* [x] Approximate paper-scale configuration

Future work may include:

* [ ] Distributed DeepSpeed training recipes
* [ ] FlashAttention integration and benchmarking
* [ ] Pre-tokenized dataset pipeline
* [ ] Multi-node training examples
* [ ] Improved DINO teacher configuration
* [ ] FVD evaluation
* [ ] Additional world-model evaluation metrics
* [ ] Longer autoregressive rollouts
* [ ] Driving-dataset adapters

---

# Reference

If you use this repository for research, please cite the original GAIA-1 paper:

```bibtex
@article{hu2023gaia1,
  title={GAIA-1: A Generative World Model for Autonomous Driving},
  author={Hu, Anthony and Russell, Lloyd and Yeo, Hudson and
          Murez, Zak and Fedoseev, George and Kendall, Alex and
          Shotton, Jamie and Corrado, Gianluca},
  journal={arXiv preprint arXiv:2309.17080},
  year={2023}
}
```

Original paper:

**A. Hu et al., "GAIA-1: A Generative World Model for Autonomous Driving," 2023.**

---

# Disclaimer

This repository is an **independent research reconstruction** created from the publicly available GAIA-1 paper and related published methods.

It is not affiliated with, endorsed by, or maintained by Wayve.

The repository does not claim to reproduce proprietary datasets, training infrastructure, unpublished hyperparameters, model weights, or internal implementation details used in the original GAIA-1 system.

Where the paper does not disclose sufficient implementation detail, the corresponding code uses documented reconstruction choices intended to preserve the published architecture and learning principles.
