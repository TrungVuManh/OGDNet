"""
Efficiency metrics for model evaluation.
"""

import torch
import torch.nn as nn
import time
from typing import Tuple, Optional


def measure_params(model: nn.Module) -> float:
    """
    Measure number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of parameters in millions
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6


def measure_fps(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: str = 'cuda',
    warmup_iters: int = 50,
    measure_iters: int = 100,
    num_runs: int = 5
) -> Tuple[float, float]:
    """
    Measure inference speed (FPS).

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
        num_runs: Number of runs for averaging

    Returns:
        Tuple of (mean_fps, std_fps)
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_size).to(device)
    fps_list = []

    for _ in range(num_runs):
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(dummy_input)

        # Synchronize
        if device == 'cuda':
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(measure_iters):
                _ = model(dummy_input)

        if device == 'cuda':
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        fps = measure_iters / elapsed_time
        fps_list.append(fps)

    mean_fps = sum(fps_list) / len(fps_list)
    std_fps = (sum((x - mean_fps) ** 2 for x in fps_list) / len(fps_list)) ** 0.5

    return mean_fps, std_fps


def _deform_conv_flops_hook(module: nn.Module, inputs, output):
    """thop hook for functional deformable-conv blocks (e.g. OGDNet's OGDConvBlock).

    These modules hold a raw 4-D ``weight`` Parameter and call
    ``torchvision.ops.deform_conv2d`` directly, so neither thop nor fvcore can see
    the main convolution. We count it like a standard conv: one MAC per
    output element per input-channel * kernel element. The bilinear sampling
    overhead is negligible relative to this term. Child modules (offset network,
    BN) are counted separately by thop, so this hook adds ONLY the deform conv.
    """
    out = output[0] if isinstance(output, (tuple, list)) else output
    out_ch, in_ch, kh, kw = module.weight.shape
    out_h, out_w = out.shape[-2], out.shape[-1]
    macs = out_h * out_w * out_ch * in_ch * kh * kw
    module.total_ops += torch.DoubleTensor([float(macs)])


def measure_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: str = 'cuda'
) -> Optional[float]:
    """
    Measure model complexity in GFLOPs (MAC-based, the fvcore/thop convention used
    in most segmentation papers; multiply by 2 for true add+mul FLOPs).

    Uses thop as the primary backend (it is lightweight and lets us add a custom
    hook for OGDNet's functional deformable convolution); falls back to fvcore.

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on

    Returns:
        GFLOPs (giga multiply-accumulate ops), or None if no backend is available.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)

    # ---- Primary backend: thop (with deformable-conv handling) ----
    try:
        from thop import profile as thop_profile

        # Standard layer types thop already counts; everything else holding a raw
        # 4-D weight Parameter is treated as a functional conv (e.g. deform conv).
        standard = (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                    nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.GroupNorm, nn.LayerNorm)
        custom_ops = {}
        for m in model.modules():
            w = getattr(m, 'weight', None)
            if (not isinstance(m, standard)
                    and isinstance(w, nn.Parameter) and w.dim() == 4):
                custom_ops[type(m)] = _deform_conv_flops_hook

        with torch.no_grad():
            macs, _ = thop_profile(
                model, inputs=(dummy_input,),
                custom_ops=custom_ops, verbose=False
            )
        return macs / 1e9

    except ImportError:
        pass
    except Exception as e:
        print(f"thop FLOPs failed ({e}); trying fvcore...")

    # ---- Fallback backend: fvcore ----
    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            return flops.total() / 1e9

    except ImportError:
        print("Warning: neither thop nor fvcore installed. Cannot compute FLOPs.")
        return None
    except Exception as e:
        print(f"FLOPs computation failed: {e}")
        return None


def measure_memory(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: str = 'cuda'
) -> float:
    """
    Measure peak GPU memory usage.

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on

    Returns:
        Peak memory in MB
    """
    if device != 'cuda':
        return 0.0

    model.eval()
    model.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        _ = model(dummy_input)

    peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

    return peak_memory


def get_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 640, 640),
    device: str = 'cuda'
) -> dict:
    """
    Get complete model efficiency summary.

    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on

    Returns:
        Dictionary with all efficiency metrics
    """
    params = measure_params(model)
    fps_mean, fps_std = measure_fps(model, input_size, device)
    gflops = measure_flops(model, input_size, device)
    memory = measure_memory(model, input_size, device) if device == 'cuda' else 0.0

    return {
        'params_m': params,
        'fps_mean': fps_mean,
        'fps_std': fps_std,
        'gflops': gflops,
        'memory_mb': memory
    }
