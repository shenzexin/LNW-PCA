# LNW-PCA

Point cloud registration components for *A Coarse-to-Fine 6-DoF Pose Measurement Algorithm using Graph Neural Networks and Reinforcement Learning*.

## Installation

```bash
pip install -e .
pip install -e ".[learning,coarse]"
```

## Registration

Store source and target XYZ coordinates as NumPy arrays with shape `(N, 3)`, in meters. Transformations map source coordinates into the target frame using `p_target = R @ p_source + t`.

```bash
python -m lnwpca source.npy target.npy --coarse --output transformation.json
python -m lnwpca source.npy target.npy --initial initial.npy --checkpoint reliability.pt
```

Without `--coarse` or `--initial`, refinement begins at the identity transformation. The optional coarse initializer extracts ISS keypoints and FPFH descriptors and estimates a rigid transformation with Open3D feature correspondence RANSAC. Refinement uses nearest-neighbor correspondences, distance rejection, weighted rigid SVD updates, and residual-based stopping.

## Reliability training

Each training NPZ file contains `source` coordinates and a binary `labels` array. Construct labels from nearest-neighbor distances after applying the known source-to-reference transformation, using a threshold of `0.01 * D`, where `D` is the reference bounding-box diagonal. Split models and acquisition sequences before generating pairs.

```bash
python -m lnwpca.train data/train --epochs 100 --output reliability.pt
```

The network uses two EdgeConv layers (64 and 128 channels), channel recalibration with reduction ratio 16, and a scalar reliability head. Training uses class-balanced binary cross-entropy. Training feature statistics are saved with the checkpoint and reused during inference.

## Adaptive weighting

`lnwpca.models` defines the six-state actor and state-action critic. `lnwpca.ddpg.DDPG` provides replay storage, critic and actor optimization, and soft target updates. Actions are normalized to `[-1, 1]`; map them to `alpha = 1 + 0.5 * action[0]` and `beta = 0.25 * action[1]`. Pass a callable returning `(alpha, beta)` as the `policy` argument to `lnwpca.geometry.refine`. State normalization must be fitted on training transitions and held fixed for inference.

## Tests

```bash
python -m unittest discover -s tests
```
