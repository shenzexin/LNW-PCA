"""Train adaptive weight modulation on source-to-reference registration pairs.

NPZ records contain source, target, ground_truth, and initial arrays.
The ground_truth and initial arrays are source-to-target 4x4 matrices.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from .geometry import features, nearest, refine, transform
from .models import ReliabilityNet
from .ddpg import DDPG, reward


def state_of(source, target, matrix, threshold, density):
    moved = transform(source, matrix)
    distances, _ = nearest(moved, target)
    reverse, _ = nearest(target, moved)
    ratio = (distances < threshold).mean()
    return np.array([distances.mean(), distances.max(), ratio,
                     .5 * (ratio + (reverse < threshold).mean()),
                     density.var(), 1 - ratio], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data')
    parser.add_argument('reliability')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--output', default='actor.pt')
    parser.add_argument('--threshold', type=float, default=.03)
    args = parser.parse_args()
    paths = sorted(Path(args.data).glob('*.npz'))
    if not paths:
        raise ValueError('No registration pairs found.')
    checkpoint = torch.load(args.reliability, map_location='cpu', weights_only=True)
    model = ReliabilityNet().eval()
    model.load_state_dict(checkpoint['model'])
    agent = DDPG()
    rng = np.random.default_rng(42)
    warmup, total = [], 0
    mean, std = np.zeros(6), np.ones(6)
    for episode in range(args.episodes):
        with np.load(paths[int(rng.integers(len(paths)))]) as record:
            source, target = record['source'], record['target']
            truth, matrix = record['ground_truth'], record['initial'].copy()
        diagonal = np.linalg.norm(np.ptp(target, axis=0))
        if diagonal <= 0:
            raise ValueError('Reference cloud has zero extent.')
        x, graph = features(source)
        normalized = ((x - checkpoint['mean']) / np.maximum(checkpoint['std'], 1e-8)).astype(np.float32)
        with torch.no_grad():
            weights = model(torch.from_numpy(normalized)[None], torch.from_numpy(graph)[None]).sigmoid()[0].numpy()
        state = state_of(source, target, matrix, args.threshold, x[:, -1])
        stable, returns = 0, 0.
        for step in range(10):
            if total < 1000:
                warmup.append(state.copy())
                action = rng.uniform(-1, 1, 2).astype(np.float32)
            else:
                with torch.no_grad():
                    action = agent.actor(torch.tensor((state - mean) / std, dtype=torch.float32)).numpy()
                sigma = .1 - .09 * min(episode / max(args.episodes - 1, 1), 1.)
                action = np.clip(action + rng.normal(0, sigma, 2), -1, 1).astype(np.float32)
            alpha, beta = 1 + .5 * action[0], .25 * action[1]
            matrix, _ = refine(source, target, matrix, weights,
                               policy=lambda _: (alpha, beta), steps=1, threshold=args.threshold)
            # Follow the weighted update with one unweighted ICP iteration.
            matrix, _ = refine(source, target, matrix, steps=1, threshold=args.threshold)
            next_state = state_of(source, target, matrix, args.threshold, x[:, -1])
            cosine = (np.trace(matrix[:3, :3] @ truth[:3, :3].T) - 1) / 2
            rotation = np.arccos(np.clip(cosine, -1, 1))
            translation = np.linalg.norm(matrix[:3, 3] - truth[:3, 3])
            value = reward(next_state[0], rotation, translation, next_state[2], next_state[5], diagonal)
            stable = stable + 1 if abs(state[0] - next_state[0]) < 1e-4 * diagonal else 0
            done = stable >= 2 or step == 9
            agent.remember(state.copy(), action.copy(), value, next_state.copy(), done)
            total += 1
            if total == 1000:
                mean = np.mean(warmup, axis=0)
                std = np.maximum(np.std(warmup, axis=0), 1e-8)
                agent.replay = type(agent.replay)(
                    [((s - mean) / std, a, r, (ns - mean) / std, d)
                     for s, a, r, ns, d in agent.replay], maxlen=100000)
            elif total > 1000:
                # The newly appended transition still uses physical state units.
                s, a, r, ns, d = agent.replay.pop()
                agent.remember((s - mean) / std, a, r, (ns - mean) / std, d)
                agent.update()
            state = next_state
            returns += value
            if done:
                break
        print(f'Episode {episode + 1:05d}: return={returns:.6f}')
    if total < 1000:
        raise ValueError('Training ended before the 1000-step normalization warmup completed.')
    torch.save({'actor': agent.actor.state_dict(), 'mean': mean.tolist(), 'std': std.tolist()}, args.output)


if __name__ == '__main__':
    main()
