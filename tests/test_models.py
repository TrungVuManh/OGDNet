import pytest
import torch
from ogdnet import METHODS, build_model, load_config, load_model
from train import ABLATIONS
from utils.losses import build_criterion
from utils.metrics.segmentation import SegmentationMetrics


@pytest.mark.parametrize('name', METHODS)
def test_model_output_contract(name):
    config = load_config(f'methods/{name}/config.yaml')
    # The historical UPerNet implementation fixes its Swin input size at 512.
    size = 512 if name == 'upernet' else 64
    config['model']['img_size'] = size
    config['data']['input_size'] = [size, size]
    model = build_model(config, pretrained=False).eval()
    with torch.inference_mode():
        prediction = model(torch.randn(1, 3, size, size))
    assert prediction.shape == (1, 1, size, size)
    assert torch.isfinite(prediction).all()
    assert prediction.min() >= 0 and prediction.max() <= 1


@pytest.mark.parametrize('variant', ABLATIONS)
def test_ogdnet_ablation_backward(variant):
    config = load_config('methods/ogdnet/config.yaml')
    config['model'].update(backbone='resnet18', pretrained=False, ablation=ABLATIONS[variant])
    model = build_model(config)
    prediction = model(torch.randn(2, 3, 64, 64))
    target = torch.zeros_like(prediction)
    target[:, :, :, 30:33] = 1
    loss = build_criterion(config)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_strict_checkpoint_roundtrip(tmp_path):
    config = load_config('methods/ogdnet/config.yaml')
    config['model'].update(backbone='resnet18', pretrained=False)
    model = build_model(config).eval()
    path = tmp_path / 'model.pth'
    torch.save({'config': config, 'model_state_dict': model.state_dict()}, path)
    restored, _ = load_model(path)
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), restored(x))


def test_global_confusion_matrix_metrics():
    metric = SegmentationMetrics()
    metric.update(torch.tensor([[[[1., 1.], [0., 0.]]]]), torch.tensor([[[[1., 0.], [1., 0.]]]]))
    result = metric.compute()
    assert result['iou'] == pytest.approx(1 / 3)
    assert result['f1'] == pytest.approx(0.5)
    assert result['pa'] == pytest.approx(0.5)
