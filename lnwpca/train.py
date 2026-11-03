"""Pretrain point reliability from labeled NPZ point clouds."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .geometry import features
from .models import ReliabilityNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='Directory of NPZ files with source and labels arrays')
    parser.add_argument('--output', default='reliability.pt')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    records = []
    for path in sorted(Path(args.data).glob('*.npz')):
        with np.load(path) as pair:
            x, graph = features(pair['source'])
            labels = np.asarray(pair['labels'], dtype=np.float32)
        if labels.shape != (len(x),) or not np.isin(labels, [0, 1]).all():
            raise ValueError(f'Invalid binary reliability labels: {path}')
        records.append((x, graph, labels))
    if not records:
        raise ValueError('No training records found.')
    all_features = np.concatenate([x for x, _, _ in records])
    mean, std = all_features.mean(0), np.maximum(all_features.std(0), 1e-8)
    model = ReliabilityNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.5)
    for epoch in range(args.epochs):
        losses = []
        optimizer.zero_grad()
        order = rng.permutation(len(records))
        for position, index in enumerate(order):
            x, graph, labels = records[index]
            logits = model(torch.from_numpy((x - mean) / std)[None],
                           torch.from_numpy(graph)[None])[0]
            labels = torch.from_numpy(labels)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            classes = [loss[labels == c].mean() for c in [0, 1] if (labels == c).any()]
            balanced = torch.stack(classes).mean()
            group_start = position // 8 * 8
            group_size = min(8, len(records) - group_start)
            (balanced / group_size).backward()
            losses.append(balanced.item())
            if (position + 1) % 8 == 0 or position + 1 == len(records):
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step()
        print(f'Epoch {epoch + 1:03d}: loss={np.mean(losses):.6f}')
    torch.save({'model': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist()}, args.output)


if __name__ == '__main__':
    main()
