"""Register XYZ arrays with optional learned point weights."""
import argparse
import json
import numpy as np
from .geometry import features, refine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Source NPY array, in meters')
    parser.add_argument('target', help='Target NPY array, in meters')
    parser.add_argument('--initial', help='Initial source-to-target 4x4 NPY matrix')
    parser.add_argument('--coarse', action='store_true', help='Initialize with ISS/FPFH RANSAC')
    parser.add_argument('--checkpoint', help='Reliability model and feature statistics')
    parser.add_argument('--actor', help='Trained actor and state normalization checkpoint')
    parser.add_argument('--threshold', type=float, default=0.03)
    parser.add_argument('--output', default='transformation.json')
    args = parser.parse_args()
    source, target = np.load(args.source), np.load(args.target)
    weights = None
    if args.checkpoint:
        import torch
        from .models import ReliabilityNet
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = ReliabilityNet().eval()
        model.load_state_dict(checkpoint['model'])
        x, graph = features(source)
        x = ((x - np.asarray(checkpoint['mean'])) / np.maximum(checkpoint['std'], 1e-8)).astype(np.float32)
        with torch.no_grad():
            weights = model(torch.tensor(x)[None], torch.tensor(graph)[None]).sigmoid()[0].numpy()
    initial = np.load(args.initial) if args.initial else None
    if args.coarse:
        from .coarse import initialize
        initial = initialize(source, target)
    policy = None
    if args.actor:
        if weights is None:
            parser.error('--actor requires --checkpoint')
        from .models import Actor
        actor_checkpoint = torch.load(args.actor, map_location='cpu', weights_only=True)
        actor = Actor().eval()
        actor.load_state_dict(actor_checkpoint['actor'])
        def policy(state):
            state = (state - actor_checkpoint['mean']) / np.maximum(actor_checkpoint['std'], 1e-8)
            with torch.no_grad():
                action = actor(torch.tensor(state, dtype=torch.float32)).numpy()
            return 1 + .5 * action[0], .25 * action[1]
    matrix, history = refine(source, target, initial, weights, policy, threshold=args.threshold)
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump({'source_to_target': matrix.tolist(), 'mean_residual': history}, stream, indent=2)


if __name__ == '__main__':
    main()
