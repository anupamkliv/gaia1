# GAIA-1: paper-faithful PyTorch reconstruction

This is a clean-room reconstruction of **GAIA-1: A Generative World Model for Autonomous Driving** (Hu et al., 2023). It is not Wayve's internal source code.

## What is directly specified by the paper

- RGB image size: **288 x 512**
- image-tokenizer downsampling: **16x**
- latent raster: **18 x 32 = 576 tokens/frame**
- codebook vocabulary: **8192**
- text encoder: **pretrained T5-large**, **32 tokens/timestep**
- action conditioning: **speed + curvature**, one scalar embedding each
- modality ordering: **TEXT -> IMAGE -> ACTION**
- common world-model dimension: **4096** for the reported large model
- learnable factorized temporal + within-timestep position embeddings
- 26 timesteps at **6.25 Hz**, ~4 s context
- 610 packed positions/timestep and **15,860** total positions
- autoregressive next-image-token objective with causal masking
- training conditioning mixture: 20% unconditional, 40% action-only, 40% text-only
- world model reported scale: **6.5B parameters**
- tokenizer reported scale: **0.3B parameters**
- decoder reported scale: **2.6B parameters**
- video decoder: 3D U-Net with factorized spatial/temporal attention
- decoder training uses image/video/autoregressive/interpolation tasks equally
- decoder token-conditioning dropout: **0.15**
- diffusion uses v-parameterization and a cosine noise schedule
- decoder training uses 7-frame clips sampled at 6.25/12.5/25 Hz
- top-k sampling for world-model inference; the paper illustrates k=50

## What the paper does not publish exactly

The exact transformer depth/head count, tokenizer U-Net widths, decoder U-Net widths, DINO variant/layer, perceptual network, GAN discriminator, exact task masks, and some sampling/stitching details are not specified. Those are marked in code/configs as reconstruction choices.

## Repo structure

```text
configs/
gaia1/
  tokenizer/
  world_model/
  diffusion/
  data/
  utils/
scripts/
tests/
```

## Training order

1. Train image tokenizer.
2. Freeze tokenizer encoder and train the autoregressive world model on discrete image tokens + T5 text features + speed/curvature.
3. Train the video diffusion decoder conditioned on tokenizer IDs.
4. Roll out image tokens autoregressively and decode them jointly into temporally consistent RGB video.

## Data sample

```python
{
    "frames": FloatTensor[T,3,288,512],
    "texts": list[str],
    "speed": FloatTensor[T],
    "curvature": FloatTensor[T],
}
```

The original GAIA-1 dataset is proprietary and is not included here.

## Important relation to MILE

This implementation requires no depth GT or BEV semantic GT for the GAIA-1 core objectives. Semantics enter partly through DINO feature distillation; temporal geometry is learned indirectly from video prediction, motion, and action conditioning. That does not automatically replace MILE's BEV supervision, but it gives the representation-learning components needed to test that hypothesis.
