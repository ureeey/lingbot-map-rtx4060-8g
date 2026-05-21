"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Windowed inference (for very long sequences, >500 frames)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10 --window_size 64

    # From video with custom FPS sampling
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10
"""
import faulthandler
faulthandler.enable()  # 开启段错误捕获

import argparse
import glob
import os
import sys
import time

# Must be set before `import torch` / any CUDA init. Reduces the reserved-vs-allocated
# memory gap by letting the caching allocator grow segments on demand instead of
# pre-reserving fixed-size blocks.
#
# Caveat: `expandable_segments:True` is **incompatible** with torch.compile's
# `cudagraph_trees` (PyTorch ≤2.8) — checkpoint pool state restore assumes the
# classic fixed-segment topology, and trips
# `RuntimeError: Expected curr_block->next == nullptr` during compiled warmup
# / replay. So when `--compile` is requested we skip the env override and let
# the default allocator run.
if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.loadimage import load_images

# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    # Only windowed mode is supported now
    from lingbot_map.models.gct_stream_window import GCTStream

    # For windowed mode, pass window_size to optimize KV cache allocation
    window_size = None
    # Calculate actual window size based on parameters
    ws = args.num_scale_frames if hasattr(args, 'num_scale_frames') else 8
    window_size_param = getattr(args, 'window_size', 64)
    keyframe_interval = getattr(args, 'keyframe_interval', 1)
    # Handle None case for keyframe_interval
    if keyframe_interval is None:
        keyframe_interval = 1
    phase2_kf = max(window_size_param - ws, 0)
    phase2_frames = phase2_kf * max(keyframe_interval, 1)
    window_size = ws + phase2_frames
    print(f"[Window Size Calculation]")
    print(f"  window_size_param (--window_size): {window_size_param}")
    print(f"  num_scale_frames: {ws}")
    print(f"  keyframe_interval: {keyframe_interval}")
    print(f"  phase2_kf: {phase2_kf}")
    print(f"  phase2_frames: {phase2_frames}")
    print(f"  window_size (actual frames): {window_size}")

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        window_size=window_size,
    )
    
    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False, mmap=True)
        print("  After torch.load(...).")
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")
    else:
        print("Warning: No model_path provided, using random initialization!")

    model = model.to(device)
    return model.eval()

# =============================================================================
# torch.compile (opt-in via --compile)
# =============================================================================

def compile_model(model):
    """Compile hot, fixed-shape modules with mode="reduce-overhead".

    Mirrors the targets in gct_profile.py:compile_model. Unlike the profile script,
    `model.point_head` is **kept** — the demo needs world_points for visualization.
    """
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")


def _warm_streaming(model, images, scale_frames, warm_stream_n, dtype,
                    passes=1, keyframe_interval=1):
    """Drive `clean_kv_cache → Phase 1 → N streaming forwards` `passes` times.

    Warmup inputs are sliced from the already-preprocessed ``images`` tensor, so
    their **spatial shape (H×W) and number of scale frames adapt to the user's
    input** — this is what makes the captured CUDA graphs match what
    ``inference_streaming`` will replay (reduce-overhead mode keys on shape).

    The streaming loop alternates keyframe / non-keyframe forwards according to
    ``keyframe_interval``, mirroring ``inference_streaming``'s call pattern so
    the ``skip_append`` (defer+append+attend+rollback) path is also captured
    during warmup.  Without this, the first non-keyframe in the real run hits
    cold orchestration code and can confuse cudagraph_trees' allocator
    checkpoint state.
    """
    num_avail = int(images.shape[0])
    scale_frames = max(1, min(int(scale_frames), num_avail))
    # Keep at least one streaming frame for the per-frame compile path; if the
    # user supplied <= scale_frames images, shrink scale to free a stream slot.
    if scale_frames >= num_avail:
        scale_frames = max(1, num_avail - 1)
    warm_stream_n = max(1, min(int(warm_stream_n), num_avail - scale_frames))
    kf_int = max(int(keyframe_interval), 1)

    # images: [S, 3, H, W] on device already; slice + add batch dim, no copy of
    # spatial dims so warmup shape == real inference shape (H, W).
    warm_scale = images[:scale_frames].unsqueeze(0).to(dtype)
    warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(dtype)

    for _ in range(passes):
        model.clean_kv_cache()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(
                warm_scale,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        for i in range(warm_stream_n):
            is_keyframe = (kf_int <= 1) or (i % kf_int == 0)
            if not is_keyframe:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(
                    warm_stream[:, i:i + 1],
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            if not is_keyframe:
                model._set_skip_append(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Wipe warmup KV so real inference_streaming starts clean (it also calls
    # clean_kv_cache internally, but this is defensive + makes intent obvious).
    model.clean_kv_cache()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, image_size_hw=None, images=None):
    """Convert pose encoding to extrinsics (c2w) and move to CPU.
    
    Args:
        predictions: Model predictions dictionary
        image_size_hw: Tuple of (height, width) for intrinsic calculation. 
                      If None, will try to get from images tensor.
        images: Optional images tensor to move to CPU. If None, only processes predictions.
    
    Returns:
        predictions: Updated predictions with extrinsic/intrinsic on CPU
        images_cpu: Images on CPU (if provided), otherwise None
    """
    # Determine image dimensions
    if image_size_hw is not None:
        h, w = image_size_hw
    elif images is not None:
        h, w = images.shape[-2:]
    else:
        raise ValueError("Either image_size_hw or images must be provided")
    
    # Convert pose encoding to extrinsics and intrinsics
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (h, w))

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("[postprocess] Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    
    # Move images to CPU only if provided
    images_cpu = None
    if images is not None:
        images_cpu = images.to("cpu", non_blocking=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Prediction Script")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rotate_clockwise_90", action="store_true",
                        help="Rotate source images 90° clockwise before preprocessing")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Windowed options
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--overlap_size", type=int, default=16)
    parser.add_argument("--overlap_keyframes", type=int, default=None)

    # Inference options (formerly streaming-specific, now general or removed if not applicable)
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Every N-th frame after scale frames is kept as a keyframe.",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    # Removed --compile as it was tied to streaming warmup logic which is removed
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload per-frame predictions to CPU during inference.",
    )

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save prediction results (.pt file)")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export preprocessed images to this folder")

    # Advanced
    parser.add_argument("--max_height", type=int, default=None)

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    use_lazy = True
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
        max_height=args.max_height,
        use_lazy_loader=use_lazy,
    )

    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        print(f"Exporting {images.shape[0]} preprocessed images to {args.export_preprocessed}...")
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        print(f"Exported to {args.export_preprocessed}")

    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    num_frames = len(images) if hasattr(images, '__len__') else images.shape[0]
    print(f"Input: {num_frames} frames")
    if hasattr(images, 'shape'):
        print(f"Shape: {tuple(images.shape)}")
    print(f"Mode: windowed")

    if args.keyframe_interval is None:
        # Auto-select keyframe interval for windowed mode if needed, or default to 1
        # Original logic was specific to streaming > 320 frames. 
        # For windowed, we might keep it simple or adapt. Let's default to 1 if None.
        args.keyframe_interval = 1

    # Removed streaming-specific compile/warmup logic

    print(f"Running windowed inference (dtype={dtype})...")
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    
    class LazyImageWrapper:
        def __init__(self, loader, device, image_size, patch_size, max_height=None):
            self.loader = loader
            self.device = device
            self.image_size = image_size
            self.patch_size = patch_size
            self.max_height = max_height
            sample = loader[0]
            self._h, self._w = sample.shape[-2:]
            self.shape = (1, len(loader), 3, self._h, self._w)
        
        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                if len(idx) == 2:
                    batch_idx, seq_idx = idx
                    if isinstance(seq_idx, slice):
                        start = seq_idx.start if seq_idx.start is not None else 0
                        stop = seq_idx.stop if seq_idx.stop is not None else len(self.loader)
                        step = seq_idx.step if seq_idx.step is not None else 1
                        indices = list(range(start, stop, step))
                        batch = torch.stack([self.loader[i] for i in indices])
                        return batch.unsqueeze(0).to(self.device)
                    elif isinstance(seq_idx, int):
                        single = self.loader[seq_idx]
                        return single.unsqueeze(0).unsqueeze(0).to(self.device)
                    else:
                        raise IndexError(f"Unsupported sequence index type: {type(seq_idx)}")
                else:
                    raise IndexError(f"Expected 2D tuple index, got {len(idx)}D")
            elif isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else len(self.loader)
                step = idx.step if idx.step is not None else 1
                indices = list(range(start, stop, step))
                batch = torch.stack([self.loader[i] for i in indices])
                return batch.unsqueeze(0).to(self.device)
            elif isinstance(idx, int):
                single = self.loader[idx]
                return single.unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                raise IndexError(f"Unsupported index type: {type(idx)}")
        
        def unsqueeze(self, dim):
            return self
        
        def to(self, device):
            return self
        
        @property
        def ndim(self):
            return 5

    images_wrapper = LazyImageWrapper(
        images, device, images.image_size, 
        images.patch_size, images.max_height
    )

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_windowed(
            images_wrapper,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            overlap_keyframes=args.overlap_keyframes,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=output_device
        )
        
        if "images" in predictions:
            del predictions["images"]

    print(f"Inference done in {time.time() - t0:.1f}s")

    # Post-processing
    sample_img = images[0]
    h, w = sample_img.shape[-2], sample_img.shape[-1]
    predictions, images_cpu = postprocess(predictions, image_size_hw=(h, w), images=None)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "predictions.pt")
    
    save_data = {
        'predictions': predictions,
        'paths': paths,
        'resolved_image_folder': resolved_image_folder,
        'image_shape': (h, w),
        'args': vars(args),
    }
    
    if images_cpu is not None:
        save_data['images'] = images_cpu
    
    print(f"Saving predictions to {output_path}...")
    torch.save(save_data, output_path)
    print(f"Saved successfully!")
    
    print(f"Prediction keys: {list(predictions.keys())}")
    if 'depth' in predictions:
        print(f"Depth shape: {predictions['depth'].shape}")
    if 'extrinsic' in predictions:
        print(f"Extrinsic shape: {predictions['extrinsic'].shape}")


if __name__ == "__main__":
    main()