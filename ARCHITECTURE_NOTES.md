# GAIA-1 architecture notes

## One timestep

```text
T5-large text                     image tokenizer                 vehicle signals
32 features                       18 x 32 IDs                     speed, curvature
     |                                 |                                |
linear -> d                         embedding -> d                  two linears -> d
     |                                 |                                |
     +--------------- TEXT (32) -> IMAGE (576) -> ACTION (2) ----------+
                                      610 positions
                                           |
                            temporal[t] + within_step[pos]
                                           |
                                 causal Transformer
                                           |
                              8192-way image-token logits
```

For 26 timesteps: 26 x 610 = 15,860 positions.

## Why causal ordering matters

For image token z[t,i], TEXT at time t has already appeared, while ACTION at time t occurs after the image block. Therefore a normal triangular causal mask exposes c[<=t], image history, and a[<t], matching the published conditional distribution.

## Tokenizer

The tokenizer reduces 288x512 by 16x to 18x32. Nearest-neighbor VQ creates one of 8192 IDs at each location. Reconstruction preserves image content; codebook/commitment losses train discrete usage; DINO feature cosine alignment pushes compressed features toward semantic representations. No depth or BEV label is required.

The exact U-Net channel schedule, discriminator, perceptual network, and DINO checkpoint/layer were not published, so those are reconstruction choices.

## Geometry

GAIA-1 does not train a supervised depth head. Geometry can emerge because future-token prediction must explain how scene appearance changes with time, ego motion, speed, curvature, motion parallax, occlusion, perspective, and object dynamics. This is an implicit geometric constraint, not metric-depth supervision.

## World model

Published: causal transformer, d=4096, ~6.5B parameters, FlashAttention v2. Exact layer count/head count/FFN/norm are not published. `gaia1_scale_approx.yaml` uses 32 layers and 32 heads as a scale-matched reconstruction, not as a claim about Wayve's private architecture.

Training modes are sampled 20% unconditional, 40% action-conditioned, 40% text-conditioned. Loss is CE only for predicted image-token IDs.

## Video decoder

Published: ~2.6B 3D U-Net, factorized spatial/temporal attention, token-conditioned, v-prediction, cosine schedule, 7 frames, image/video/autoregressive/interpolation tasks equally sampled, token dropout 0.15, EMA 0.999.

The exact U-Net widths and task masking probabilities inside each task are not fully published. The repo implements the mechanism with replaceable choices.

## Training labels actually needed

Tokenizer: RGB images. DINO supplies pretrained self-supervised teacher features.

World model: video/image IDs, text narration/metadata where available, speed and curvature.

Diffusion decoder: RGB video and frozen tokenizer IDs.

No BEV semantic segmentation GT and no depth GT are required by the core GAIA-1 objectives.
