# VGGT-DP

**Visual Geometry Grounded Diffusion Policy (VGGT-DP)** is a visual imitation
learning algorithm that uses the frozen [VGGT](https://github.com/facebookresearch/vggt)
aggregator as a geometry-aware visual prior for a diffusion policy. This
repository contains the model, training and evaluation code for the MetaWorld
experiments reported in the paper. It is built on top of
[3D Diffusion Policy (DP3)](https://github.com/YanjieZe/3D-Diffusion-Policy).

Pretrained policy checkpoints are **not** released. The VGGT-1B encoder weights
are pulled from the Hugging Face Hub at runtime.

## Method

```
RGB observations ---> frozen VGGT aggregator ---> patch tokens [B, S, N, 2048]
                                                        |
                                    transformer projector + mean pooling
                                                        |
proprioception ---> MLP -----------------------> global conditioning [B, 256]
                                                        |
                                          ConditionalUnet1D (DDIM) ---> action chunk
```

The VGGT trunk stays frozen; only the projector, the proprioception MLP and the
diffusion UNet are trained (about 255M trainable parameters out of 1.2B total).

## Installation

See [INSTALL.md](INSTALL.md).

## Data

Demonstrations are generated with the MetaWorld scripted experts:

```bash
bash scripts/gen_demonstration_metaworld.sh hand-insert True 10
```

This writes `data/metaworld_hand-insert_expert.zarr`. Pass `False` as the second
argument to render the arm invisible, which produces
`data/metaworld_hand-insert_expert_invis.zarr` and is what `robovis=false`
consumes. Because you regenerate the demonstrations yourself, results can differ
slightly from the table below: imitation learning is sensitive to demonstration
quality.

## Training

```bash
bash scripts/train_policy.sh metaworld_hand-insert vggtdp 0 0
```

Arguments are `<task> <exp_tag> <seed> <gpu_id> [eval_robovis] [robovis]
[prio_as_cond] [visual_prio_training]`, all four flags defaulting to `true`.
Checkpoints and rollout videos land in `data/outputs/<task>-vggt_dp-<tag>_seed<seed>/`.
Training runs for 3000 epochs and evaluates 20 episodes every 200 epochs;
success rates are logged to wandb.

To reproduce the paper memory footprint (about 20GB), quantize the frozen trunk:

```bash
bash scripts/train_policy.sh metaworld_hand-insert vggtdp 0 0 true true true true \
    policy.vggtq=true
```

## Evaluation

```bash
bash scripts/eval_policy.sh metaworld_hand-insert vggtdp 0 0
```

The arguments must match the training invocation so that the same run directory
is resolved. Evaluation loads `checkpoints/latest.ckpt` from that directory and
rolls out 20 episodes.

> A checkpoint trained with `policy.vggtq=true` can only be loaded with
> `policy.vggtq=true`, because post-training quantization adds buffers to the
> encoder state dict.

## Results

Success rate (%) on 10 MetaWorld tasks, mean and standard deviation. DP and DP3
numbers are quoted from the paper for reference; their implementations live in
the [DP3 repository](https://github.com/YanjieZe/3D-Diffusion-Policy).

| Task | DP | DP3 | VGGT-DP |
|---|---|---|---|
| Disassemble | 43 ± 7 | **69 ± 4** | 55 ± 2.5 |
| Peg Unplug Side | 74 ± 3 | **75 ± 5** | 63 ± 6 |
| Pick out of Hole | 0 ± 0 | 14 ± 9 | **55 ± 6** |
| Shelf Place | 11 ± 3 | **17 ± 10** | 10 ± 0 |
| Reach | 18 ± 2 | 24 ± 1 | **42 ± 8** |
| Soccer | 14 ± 4 | 18 ± 3 | **30 ± 7** |
| Sweep Into | 10 ± 4 | 15 ± 5 | **44 ± 4** |
| Hand Insert | 10 ± 4 | 15 ± 5 | **19 ± 4** |
| Pick Place | 0 ± 0 | **12 ± 4** | 0 ± 0 |
| Stick Pull | 11 ± 2 | 27 ± 8 | **48 ± 5** |
| **Average** | 19.1 | 28.6 | **36.6** |

## Repository layout

```
train.py                                   training entry point
eval.py                                    evaluation entry point
scripts/                                   demonstration / train / eval wrappers
third_party_patches/                       MetaWorld visible-robot assets
diffusion_policy_3d/
  config/vggt_dp.yaml                      main config
  config/task/metaworld_*.yaml             per-task configs
  policy/dp.py                             DP policy and DPEncoder
  model/vision/vggt/                       VGGT encoder (aggregator trunk only)
  model/diffusion/conditional_unet1d.py    diffusion UNet
  dataset/metaworld_dataset.py             zarr replay buffer
  env/metaworld/                           MuJoCo env wrapper, point clouds
  env_runner/metaworld_runner.py           rollout and success bookkeeping
```

## Acknowledgement

This code builds on [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy)
and [VGGT](https://github.com/facebookresearch/vggt).
