# TransFM

Code for **TransFM: Visible-to-Infrared Image Translation via Flow Matching**.

TransFM studies visible-to-infrared image translation. The current implementation trains a flow-matching model in the latent space of Stable Diffusion, with image caption prompts, CLIP text embeddings, and wavelet high-frequency guidance for detail preservation.

Paper: [IEEE/CAA Journal of Automatica Sinica, 2026, 13(5): 1239-1241](https://www.ieee-jas.com/en/article/doi/10.1109/JAS.2025.125930)

## Features

- Visible-to-infrared image translation with flow matching.
- Latent-space training through the Stable Diffusion 2.1 VAE.
- Text conditioning from BLIP captions and CLIP token embeddings.
- Wavelet high-frequency conditioning for preserving structural details.
- Training, validation, checkpointing, TensorBoard logging, and standalone inference scripts.

## Repository Structure

```text
TransFM/
+-- configs/
|   +-- fusion.yaml          # Main training configuration
+-- dataset/
|   +-- dataset_base.py      # Image pair datasets
|   +-- video_dataset_base.py # Video-style dataset utilities
|   +-- utils.py             # Image IO, transforms, samplers, helpers
+-- evaluation/
|   +-- evaluator.py         # Validation and metric calculation
+-- lutils/                  # Configuration, logging, loss, distributed helpers
+-- model/
|   +-- model.py             # Flow-matching model wrapper
|   +-- SD_Unet_Hp.py        # Vector-field regressor backbone
|   +-- layers/              # Attention and network blocks
+-- training/
|   +-- trainer.py           # Training loop, loss, checkpoints
|   +-- training_loop.py     # Dataset/model/trainer assembly
+-- train.py                 # Training entry point
+-- test.py                  # Standalone inference/evaluation script
```

## Environment

The code is intended for a CUDA-enabled PyTorch environment.

```bash
conda create -n transfm python=3.8 -y
conda activate transfm

# Install the PyTorch build that matches your CUDA version.
pip install torch torchvision

pip install diffusers transformers accelerate
pip install einops torchdiffeq PyYAML tqdm wandb tensorboard
pip install opencv-python scikit-image pillow numpy natsort PyWavelets
```

If you use a specific CUDA version, install PyTorch from the official PyTorch command selector instead of the generic command above.

## Data Preparation

The default configuration expects paired visible and infrared images in PNG format. The training dataset reads visible and infrared images from two folders and sorts them by filename, so corresponding pairs should share the same ordering and preferably the same filename.

Example layout:

```text
datasets/
+-- M3FD/
    +-- train/
    |   +-- Vis/
    |   |   +-- 000001.png
    |   |   +-- ...
    |   +-- Ir/
    |       +-- 000001.png
    |       +-- ...
    +-- test/
        +-- Vis/
        +-- Ir/
```

Update these paths in `configs/fusion.yaml` before training:

```yaml
datasets:
  train:
    dataroot_gt_vi: /path/to/M3FD/train/Vis
    dataroot_gt_ir: /path/to/M3FD/train/Ir
  test:
    dataroot_gt_vi: /path/to/M3FD/test/Vis
    dataroot_gt_ir: /path/to/M3FD/test/Ir
```

## Required Local Weights

Several paths in the research code are currently hard-coded and must be changed for your machine.

1. Stable Diffusion 2.1 directory

   Used by `training/trainer.py` and `test.py`:

   ```python
   vae_id = "/path/to/stable-diffusion-2-1"
   ```

   The directory should be loadable by Hugging Face `diffusers` and contain at least the `vae`, `text_encoder`, and `tokenizer` subfolders.

2. DepthFM checkpoint

   Used by `model/model.py`:

   ```python
   ckpt_path = "/path/to/depthfm-v1.ckpt"
   ```

3. Test checkpoint and data paths

   Used by `test.py`:

   ```python
   model_folder_name = "runs/FlowMatching_VideoFusion_run-your_run_name/"
   ckpt_path = model_folder_name + "checkpoints/step_xxx.pth"
   base_dir = Path("/path/to/M3FD/test/Vis")
   root_b = Path("/path/to/M3FD/test/Ir")
   ```

4. Output directory

   `test.py` writes translated images to `outputs_tmp/`. Create this folder before running inference:

   ```bash
   mkdir -p outputs_tmp
   ```

## Training

Edit `configs/fusion.yaml` first, then launch training:

```bash
python train.py \
  --run-name m3fd \
  --config configs/fusion.yaml \
  --num-gpus 1
```

Useful options:

```bash
python train.py --help
```

Resume from the latest checkpoint:

```bash
python train.py \
  --run-name m3fd \
  --config configs/fusion.yaml \
  --num-gpus 1 \
  --resume-step -1
```

Resume from a specific checkpoint named `step_20000.pth`:

```bash
python train.py \
  --run-name m3fd \
  --config configs/fusion.yaml \
  --num-gpus 1 \
  --resume-step 20000
```

Training outputs are saved under:

```text
runs/FlowMatching_VideoFusion_run-<run-name>/
+-- checkpoints/
+-- logs/
```

View TensorBoard logs:

```bash
tensorboard --logdir runs
```

## Inference and Evaluation

Before running inference, update `test.py` with your checkpoint, Stable Diffusion path, test visible folder, test infrared folder, and output directory.

```bash
python test.py
```

The script reports PSNR, SSIM, and average inference time, and saves translated images to `outputs_tmp/`.

## Notes

- `train.py` and `test.py` currently set `CUDA_VISIBLE_DEVICES="0"`. Change this line if you want to use a different GPU.
- `training/trainer.py`, `evaluation/evaluator.py`, `model/model.py`, and `test.py` contain local absolute paths from the original experiment environment. Replace them with paths on your machine before running.
- The default configuration uses M3FD-style visible/infrared pairs, but the image dataset loader can be reused for other paired datasets with the same folder structure.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{gong2026transfm,
  title={TransFM: Visible-to-Infrared Image Translation via Flow Matching},
  author={Gong, Meiqi and Zhang, Hao and Hui, Bingwei and Ma, Jiayi},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={13},
  number={5},
  pages={1239--1241},
  year={2026},
  doi={10.1109/JAS.2025.125930}
}
```

## Acknowledgements

This implementation builds on PyTorch, Hugging Face Diffusers, Transformers, and related open-source tools for diffusion models, flow matching, and image processing.
