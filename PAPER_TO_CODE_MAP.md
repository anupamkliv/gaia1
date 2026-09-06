# Paper-to-code map

| GAIA-1 component | Repository implementation |
|---|---|
| Eq. 1 autoregressive image-token likelihood | `gaia1/world_model/model.py::loss` |
| TEXT -> IMAGE -> ACTION ordering | `GaiaWorldModel.embeddings` |
| T5-large, 32 text tokens | `GaiaWorldModel.text_features` |
| speed + curvature embeddings | `GaiaWorldModel.speed`, `GaiaWorldModel.curv` |
| factorized temporal/spatial positions | `GaiaWorldModel.temporal`, `GaiaWorldModel.spatial` |
| 8192 image-token vocabulary | tokenizer VQ + world `image_emb/head` |
| 288x512 -> 18x32 | `ImageTokenizer` four 2x downsamples |
| nearest-neighbor projected normalized VQ | `ProjectedVQ` |
| reconstruction/GAN/codebook losses | `tokenizer/losses.py`, `train_tokenizer.py` |
| DINO semantic distillation hook | `tokenizer/dino_teacher.py`, `DinoAlignment` |
| 20/40/40 conditioning training | `GaiaWorldModel.sample_mode` |
| top-k autoregressive inference | `world_model/generation.py` |
| classifier-free logit guidance | `cfg_logits` |
| factorized spatial/temporal video attention | `diffusion/model.py` |
| cosine diffusion schedule | `diffusion/schedule.py` |
| v target | `CosineSchedule.v_target` |
| equal decoder task sampling | `diffusion/tasks.py` |
| 7-frame decoder training | `scripts/train_video_decoder.py` |
| DDIM-like decoding | `diffusion/sampling.py` |
| full rollout | `scripts/rollout.py` |

## Known non-identities

The following code is intentionally **not claimed to be exact Wayve architecture** because the paper does not disclose it:

1. tokenizer residual-block/channel schedule
2. VGG perceptual implementation
3. PatchGAN discriminator
4. exact DINO checkpoint/layer
5. transformer layer/head/FFN/norm details
6. diffusion U-Net channel hierarchy and attention placements
7. exact per-task mask distribution
8. exact long-video overlap/stitch strategy

The repo keeps these isolated so they can be replaced when better implementation evidence becomes available.
