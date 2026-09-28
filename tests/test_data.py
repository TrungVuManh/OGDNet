import json
import cv2
import numpy as np
import pytest
from scripts.prepare_split import create_split
from utils.datasets import CrackDataset, get_val_transforms


@pytest.fixture
def dataset(tmp_path):
    for folder in ['Image', 'Label']:
        (tmp_path / folder).mkdir()
    for i in range(6):
        image = np.full((64, 64, 3), 50 + i * 20, np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[:, 30:33] = 255
        cv2.imwrite(str(tmp_path / 'Image' / f'{i}.png'), image)
        cv2.imwrite(str(tmp_path / 'Label' / f'{i}.png'), mask)
    return tmp_path


def test_split_is_repeatable_disjoint_and_loadable(dataset):
    split = create_split(dataset)
    assert split == create_split(dataset)
    assert not set(split['train']) & set(split['val'])
    path = dataset / 'split.json'
    path.write_text(json.dumps(split))
    config = {'data': {'dataset_path': str(dataset), 'split_file': str(path), 'input_size': [32, 64]}}
    sample = CrackDataset(config, 'val', get_val_transforms(config))[0]
    assert sample['image'].shape == (3, 32, 64)
    assert sample['mask'].shape == (1, 32, 64)
    assert set(sample['mask'].unique().tolist()) <= {0., 1.}


def test_missing_or_overlapping_manifest_fails(dataset):
    path = dataset / 'missing.json'
    config = {'data': {'dataset_path': str(dataset), 'split_file': str(path)}}
    with pytest.raises(FileNotFoundError, match='Split missing'):
        CrackDataset(config)
    path.write_text(json.dumps({'train': ['0', '1'], 'val': ['1']}))
    with pytest.raises(ValueError, match='overlapping'):
        CrackDataset(config)


def test_group_split_keeps_acquisitions_together(dataset):
    groups = dataset / 'groups.csv'
    groups.write_text('image_id,group_id\n0,a\n1,a\n2,b\n3,b\n4,c\n5,c\n')
    split = create_split(dataset, groups=groups)
    for pair in ({'0', '1'}, {'2', '3'}, {'4', '5'}):
        assert pair <= set(split['train']) or pair <= set(split['val'])


def test_duplicate_images_across_subsets_rejected(dataset):
    (dataset / 'Image/5.png').write_bytes((dataset / 'Image/0.png').read_bytes())
    with pytest.raises(ValueError, match='Byte-identical'):
        create_split(dataset, order='sorted')
