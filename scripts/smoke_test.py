"""Exercise split, training, checkpoint loading, evaluation, and prediction on synthetic images."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/smoke'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new directory')
    root = args.output.resolve()
    for folder in ('Image', 'Label'):
        (root / 'data' / folder).mkdir(parents=True)
    rng = np.random.default_rng(42)
    for index in range(6):
        image = rng.integers(120, 220, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, 25 + index:28 + index] = 255
        image[mask > 0] = 20
        cv2.imwrite(str(root / 'data/Image' / f'{index}.png'), image)
        cv2.imwrite(str(root / 'data/Label' / f'{index}.png'), mask)
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], check=True, env=environment)
    run('scripts/prepare_split.py', '--data', root / 'data', '--output', root / 'split.json')
    config = yaml.safe_load(Path('methods/ogdnet/config.yaml').read_text(encoding='utf-8'))
    config['model'].update(backbone='resnet18', pretrained=False)
    config['data'].update(dataset_path=str(root / 'data'), split_file=str(root / 'split.json'), input_size=[64, 64])
    config['training'].update(epochs=1, batch_size=2)
    (root / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    run('train.py', '--config', root / 'config.yaml', '--device', 'cpu', '--output', root / 'train')
    checkpoint = root / 'train/best_model.pth'
    run('evaluate.py', '--checkpoint', checkpoint, '--device', 'cpu', '--output', root / 'evaluation')
    run('predict.py', '--checkpoint', checkpoint, '--device', 'cpu', '--image', root / 'data/Image/0.png', '--output', root / 'mask.png')
    metrics = json.loads((root / 'evaluation/metrics.json').read_text())
    assert metrics['n_images'] == 2
    mask = cv2.imread(str(root / 'mask.png'), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (64, 64) and set(np.unique(mask)) <= {0, 255}
    print('End-to-end synthetic smoke test passed; this is not a benchmark experiment.')


if __name__ == '__main__':
    main()
