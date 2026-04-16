# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Create and activate the conda environment before running the notebook:

```bash
conda create -n medsam python=3.10 -y
conda activate medsam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install requests numpy matplotlib Pillow transformers accelerate nibabel monai SimpleITK scikit-image connected-components-3d
conda install -n medsam ipykernel --update-deps --force-reinstall
```

Launch the notebook:
```bash
conda activate medsam
jupyter notebook assignment_04.ipynb
```

## Project Overview

HCI701 (Spring 2026) Assignment 04 — medical image segmentation using **MedSAM** (Segment Anything Model adapted for medical imaging).

The notebook demonstrates:
1. Loading 3D CT abdominal scans from `.nii.gz` files via SimpleITK
2. Visualizing 2D depth slices with multi-class organ mask overlays
3. Running MedSAM inference (`flaviagiammarino/medsam-vit-base` via HuggingFace Transformers) with a bounding box prompt
4. Computing Dice scores between predicted and ground-truth masks using MONAI's `DiceMetric`

## Data Layout

```
data/npy/CT_Abd/
  imgs/          # 2D slice .npy files: CT_Abd_FLARE22_Tr_XXXX-ZZZ.npy, shape (1024,1024,3), float [0,1]
  gts/           # Corresponding label .npy files: same naming, shape (1024,1024), uint8 multiclass
  CT_Abd_FLARE22_Tr_XXXX_img.nii.gz   # Full 3D image volumes [Z,H,W]
  CT_Abd_FLARE22_Tr_XXXX_gt.nii.gz    # Full 3D label volumes [Z,H,W]
```

The dataset contains 5 cases (`FLARE22_Tr_0001` through `FLARE22_Tr_0005`) with 13 abdominal organ classes (Liver, Kidneys, Spleen, Pancreas, Aorta, IVC, Adrenal Glands, Gallbladder, Esophagus, Stomach, Duodenum).

## Key Architectural Notes

**MedSAM inference pipeline** (`run_medsam`):
- Input: PIL RGB image + bounding box `[x_min, y_min, x_max, y_max]`
- `SamProcessor` handles preprocessing; output logits come from `model.pred_masks`
- Post-processing: `.sigmoid()` → `post_process_masks()` (handles resizing back to original dims) → threshold at 0.5
- Returns binary float tensor of shape `(1, 1, H, W)`

**GT mask shape convention**: `get_gt_mask()` returns shape `(1, 1, H, W)` — a binary mask for a single organ ID — to match the MedSAM prediction shape expected by `DiceMetric`.

**Dice computation**: MONAI's `DiceMetric(include_background=True)` expects inputs of shape `(B, C, spatial...)`. The starter code has a known bug where `pred_tensor` is not unsqueezed to 4D in `plot_medsam_results_with_gt` — verify shapes before computing Dice.

## Tasks

Assignment tasks (Task-1 through Task-4) are stub cells to be filled in. Task details are released April 16, 2026 at 09:00 AM; submission is due end of lab session.
