# Installation

Tested on Ubuntu with CUDA 12.1 and an NVIDIA L20 (46GB). Training VGGT-DP needs
about 20GB of VRAM with `policy.vggtq=true`, and noticeably more in full
precision.

## 1. Conda environment

```bash
conda create -n vggtdp python=3.8
conda activate vggtdp
```

## 2. PyTorch

```bash
# CUDA 12.1
pip3 install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3. This repository

```bash
git clone <this-repo> VGGT-DP && cd VGGT-DP
pip install -e .
```

## 4. MuJoCo 2.1.0

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz
```

Add to your shell rc file, then open a new shell:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```

## 5. Simulation stack

MetaWorld and gym are pinned forks; newer releases change the observation and
reward semantics and will not reproduce the numbers in the README.

```bash
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

mkdir -p third_party && cd third_party
git clone https://github.com/YanjieZe/Metaworld.git
git clone https://github.com/openai/gym.git -b v0.21.0 gym-0.21.0
git clone https://github.com/openai/mujoco-py.git -b v2.1.2.14 mujoco-py-2.1.2.14

cd gym-0.21.0 && pip install -e . && cd ..
cd mujoco-py-2.1.2.14 && pip install -e . && cd ..
cd Metaworld && pip install -e . && cd ..
cd ..
```

Apply the MetaWorld patch that adds the visible / invisible robot assets and the
`--robovis` flag to the demonstration generator, see
[third_party_patches/README.md](third_party_patches/README.md):

```bash
cp -r third_party_patches/Metaworld/. third_party/Metaworld/
```

## 6. pytorch3d (simplified)

Only the point cloud sampling ops are used:

```bash
cd third_party
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git dp3-src
cp -r dp3-src/third_party/pytorch3d_simplified . && rm -rf dp3-src
cd pytorch3d_simplified && pip install -e . && cd ../..
```

## 7. Remaining Python packages

```bash
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf \
    hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 \
    numba==0.56.4 moviepy imageio av matplotlib termcolor \
    safetensors huggingface_hub accelerate

# only needed when policy.vggtq=true
pip install coremltools
```

## 8. VGGT-1B weights

The frozen encoder loads the aggregator trunk of
[facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B). It is fetched from
the Hugging Face Hub on first use. To point at a local copy instead:

```bash
export VGGT_WEIGHT_PATH=/path/to/VGGT-1B/model.safetensors
```

## Verify

```bash
python -c "from diffusion_policy_3d.policy.dp import DP; print(\"ok\")"
python -c "from diffusion_policy_3d.env_runner.metaworld_runner import MetaworldRunner; print(\"ok\")"
```
