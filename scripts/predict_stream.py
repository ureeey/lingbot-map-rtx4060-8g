"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Streaming inference (frame-by-frame with KV cache)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/

    # Streaming inference with keyframe KV caching
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/ --mode streaming --keyframe_interval 6

    # Windowed inference (for very long sequences, >500 frames)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10 --mode windowed --window_size 64

    # From video with custom FPS sampling
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10
"""

import argparse
import glob
import os
import sys
import tempfile
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
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.loadimage import load_images,LazyImageWrapper

import torch.ao.quantization as quant
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig
from torchinfo import summary
from torchao.utils import get_model_size_in_bytes
def quantize_linear_to_int8(module):
    """纯PyTorch INT8逐通道权重量化，零依赖"""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            with torch.no_grad():
                w = child.weight.data
                # 逐通道计算 scale
                scale = w.abs().max(dim=1).values / 127.0
                # 量化到 INT8
                w_int8 = torch.clamp(torch.round(w / scale.unsqueeze(1)), -127, 127).to(torch.int8)
                child.weight.requires_grad_(False)
                # 替换权重
                child.weight.data = w_int8
                # 保存 scale
                child.register_buffer('quant_scale', scale.to(w.device))
                
                # 替换 forward：推理时反量化
                orig_forward = child.forward
                def forward_hook(x, child=child, orig_forward=orig_forward):
                    w_fp16 = child.weight.data.float() * child.quant_scale.unsqueeze(1).to(x.dtype)
                    return torch.nn.functional.linear(x, w_fp16, child.bias)
                child.forward = forward_hook
        else:
            quantize_linear_to_int8(child)

# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device, num_frames):
    """Load GCTStream model from checkpoint."""
    from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=num_frames,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        kv_cache_fp8=args.kv_cache_fp8,
        kv_cache_cut=args.kv_cache_cut,
        gqa_ratio=args.gqa_ratio,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        # 1. 先在CPU加载权重，避免GPU显存峰值，注意mmap选项，小心CPU内存都不够
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False, mmap=True)
        #ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        print("  After torch.load(...).")
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    # 2. 移动到设备
    model = model.to(device)
    return model.eval()

    #return model.to(device).eval()

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
    # Support both tensor and ImageLazyLoader
    if hasattr(images, 'shape'):
        num_avail = int(images.shape[0])
    else:
        # For ImageLazyLoader or similar objects
        num_avail = len(images)
    
    scale_frames = max(1, min(int(scale_frames), num_avail))
    # Keep at least one streaming frame for the per-frame compile path; if the
    # user supplied <= scale_frames images, shrink scale to free a stream slot.
    if scale_frames >= num_avail:
        scale_frames = max(1, num_avail - 1)
    warm_stream_n = max(1, min(int(warm_stream_n), num_avail - scale_frames))
    kf_int = max(int(keyframe_interval), 1)

    # Get device from model
    device = next(model.parameters()).device

    # images: [S, 3, H, W] on device already; slice + add batch dim, no copy of
    # spatial dims so warmup shape == real inference shape (H, W).
    # For ImageLazyLoader, we need to load and move to device
    if hasattr(images, 'shape'):
        warm_scale = images[:scale_frames].unsqueeze(0).to(device=device, dtype=dtype)
        warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(device=device, dtype=dtype)
    else:
        # For ImageLazyLoader, load slices and move to device
        warm_scale = images[:scale_frames].unsqueeze(0).to(device=device, dtype=dtype)
        warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(device=device, dtype=dtype)

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
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rotate_clockwise_90", action="store_true",
                        help="Rotate source images 90° clockwise before preprocessing "
                             "(crop/resize then operates on the rotated aspect ratio)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming"],
                        help="streaming: frame-by-frame with KV cache; windowed: overlapping windows for long sequences")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Every N-th frame after scale frames is kept as a keyframe. 1 = every frame. "
            "Streaming: if unset, auto-selected (1 when num_frames <= 320, else ceil(num_frames / 320)) "
            "to bound KV cache. Windowed: defaults to 1; --window_size counts keyframes, so values >1 "
            "expand each window's actual-frame coverage to "
            "scale_frames + (window_size - scale_frames) * keyframe_interval.",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4,
                        help="Camera head iterative-refinement steps. Default 4; set 1 for faster inference "
                            "(skips 3 refinement passes at a small accuracy cost).")
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile hot modules (reduce-overhead) with a CUDA-graph warmup. "
                            "Streaming mode only; ~5 FPS faster at 518x378. Adds ~30-60 s warmup time.")
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload per-frame predictions to CPU during inference to cut GPU peak memory "
            "(on by default).  Use --no-offload_to_cpu to keep outputs on GPU.",
    )

    # Advanced
    parser.add_argument("--max_height", type=int, default=294,
                        help="Maximum height for images. If specified, images taller than this will be center-cropped.")
    parser.add_argument("--lazyloader", action="store_true", default=True,
                        help="Use lazy loading for images to reduce memory usage. "
                             "Recommended for long sequences (>200 frames).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save prediction results (.pt file)")
    parser.add_argument("--quant_wa", type=str, default="none", choices=["none", "int", "fp8"],
                        help="quantization for weights and activations.\n" \
                             "none: no quantization\n"
                             "int:\n"
                             "  aggregator  -> INT4_WEIGHT_ONLY \n"
                             "  camera_head -> INT8_WEIGHT_ONLY \n"
                             "  depth_head  -> INT8_WEIGHT_ONLY \n"
                             "fp8: \n"
                             "  the whole model -> Float8DynamicActivationFloat8WeightConfig \n")
    parser.add_argument("--kv_cache_fp8", action="store_true", default=False,
                        help="Store KV cache in FP8 (FlashInfer only).")
    parser.add_argument("--kv_cache_cut", type=int, default=1,
                    help="Store KV cache by downsampled with Factor (FlashInfer only).")
    parser.add_argument("--gqa_ratio", type=int, default=1,
                        help="Deprecated !!! Fixed 1.\n" \
                        "Group Query Attention ratio. Default: 1, no GQA; set >1 to reduce KV cache size at a potential quality cost.")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"

    args.gqa_ratio = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
        max_height=args.max_height,
        use_lazy_loader=args.lazyloader,
    )
    num_frames = len(images) if hasattr(images, '__len__') else images.shape[0]
    if hasattr(images, 'shape'):
        print(f"Shape: {tuple(images.shape)}")

    model = load_model(args, device, num_frames)
    print(f"Total load time: {time.time() - t0:.1f}s")
        
    # 1 - 自定义权重压缩，没有加速计算
    # quantize_linear_to_int8(model.aggregator)
    # 2 - PyTorch动态量化，仅支持CPU
    # model.to('cpu')
    # model.aggregator = quant.quantize_dynamic(
    #     model.aggregator, {torch.nn.Linear}, dtype=torch.qint8
    # )
    # model.to(device) # 后续无法执行
    # 3 - torchao 的权重量化 IN8
    # quantize_(model, Int8WeightOnlyConfig())
    # 4 - torchao 的权重量化 IN4，依赖模型已经转 bf16
    # config = Int4WeightOnlyConfig(
    #     group_size=32,
    #     int4_packing_format="tile_packed_to_4d",
    #     int4_choose_qparams_algorithm="hqq"  # ✅ 选择 HQQ 算法，绕过 mslk
    # )
    # quantize_(model, config)

    # Pick inference dtype; autocast still runs for the ops that need fp32 (e.g. LayerNorm).
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Cast the aggregator (DINOv2-style trunk) to the inference dtype to remove the
    # redundant fp32 master weight copy + autocast bf16 weight cache (~2-3 GB saved,
    # no measurable quality change). gct_base._predict_* upcasts inputs to fp32 and
    # runs each head under `autocast(enabled=False)`, so camera/depth/point heads
    # keep fp32 weights automatically.
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)
        # 6 转换整个模型为 bf16 行不通，因为 camera_head.py line 282 need scalar type Float but found BFloat16
        # print(f"Casting the whole model to {dtype}")
        # model = model.to(dtype=dtype)        

#    model.camera_head = model.camera_head.to(dtype=torch.float16) # expected scalar type Float but found Half
#    model.depth_head = model.depth_head.to(dtype=torch.float16) # expected scalar type Float but found Half

    print(f"[ 量化前模型占用内存: {get_model_size_in_bytes(model) / (1024**3):.2f} GB ]") if args.quant_wa != "none" else None

    if args.quant_wa == "int":
        # 5 - torchao 的权重量化 INT4，必须放在在 model.aggregator 转 bf16 之后
        config = Int4WeightOnlyConfig(
            group_size=32,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq"  # ✅ 选择 HQQ 算法，绕过 mslk
        )

        quantize_(model.aggregator, config)
        quantize_(model.camera_head, Int8WeightOnlyConfig())
        quantize_(model.depth_head, Int8WeightOnlyConfig())
    elif args.quant_wa == "fp8":
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())

    print(f"[ 量化后模型占用内存: {get_model_size_in_bytes(model) / (1024**3):.2f} GB ]") if args.quant_wa != "none" else None

    if 0:
        dummy_input = torch.randn(1, 3, 518, 294, dtype=torch.bfloat16).cuda()
        summary(
            model,
            input_data=dummy_input,
            depth=1
        )

    if args.keyframe_interval is None:
        args.keyframe_interval = (num_frames + 319) // 320
        print(
            f"Auto-selected --keyframe_interval={args.keyframe_interval} "
        )

    if args.keyframe_interval > 1:
        print(
            f"Keyframe streaming enabled: interval={args.keyframe_interval} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )

    # ── Optional: torch.compile + CUDA-graph warmup (streaming only) ────────
    if args.compile:
        scale_for_warm = min(args.num_scale_frames, num_frames)
        if scale_for_warm >= num_frames:
            scale_for_warm = max(1, num_frames - 1)
        warm_stream_n = min(10, max(1, num_frames - scale_for_warm))
        # Get image dimensions - handle both tensor and lazy loader
        if hasattr(images, 'shape'):
            warm_h, warm_w = int(images.shape[-2]), int(images.shape[-1])
        elif hasattr(images, 'image_size') and hasattr(images, 'patch_size'):
            # For ImageLazyLoader, calculate actual dimensions from image_size and patch_size
            warm_h = images.image_size
            # Width is calculated based on aspect ratio preservation during preprocessing
            # The load_images function maintains aspect ratio, so we need to get actual dims
            # Load first image temporarily to get actual shape
            sample_img = images[0] if hasattr(images, '__getitem__') else None
            if sample_img is not None:
                warm_h, warm_w = int(sample_img.shape[-2]), int(sample_img.shape[-1])
                del sample_img
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                warm_w = warm_h  # Fallback to square if can't determine
        else:
            raise ValueError("Cannot determine image dimensions from input")
            
        print(
            f"Warmup eager (scale={scale_for_warm} + {warm_stream_n} streaming, "
            f"shape={warm_h}x{warm_w}, kf_int={args.keyframe_interval})..."
        )
        t_warm = time.time()
        _warm_streaming(
            model, images, scale_for_warm, warm_stream_n, dtype,
            passes=1, keyframe_interval=args.keyframe_interval,
        )
        print(f"  eager warmup: {time.time() - t_warm:.1f}s")

        print("Compiling hot modules...")
        compile_model(model)

        # 3 passes under compile: 1st captures CUDA graphs, 2nd/3rd replay so
        # the caching allocator / graph-address map converge on the state the
        # real inference will see. See gct_profile.py:302-306 for rationale.
        print("Warmup compiled (3x dress rehearsal)...")
        t_warm = time.time()
        _warm_streaming(
            model, images, scale_for_warm, warm_stream_n, dtype,
            passes=3, keyframe_interval=args.keyframe_interval,
        )
        print(f"  compiled warmup: {time.time() - t_warm:.1f}s")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\n    Running {args.mode} inference...\n")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.lazyloader:
            images_wrapper = LazyImageWrapper(
                    images, 
                    device, 
                    images.image_size, 
                    images.patch_size,
                    images.max_height
                )
            predictions = model.inference_streaming(
                images_wrapper,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )

    t_infer = time.time() - t0
    print(f"\n    Inference done in {t_infer:.1f}s, FPS = {num_frames / t_infer:.1f} \n")
    print(f"[ allocated 显存峰值：{torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB ]")

    # ── Aggressive memory cleanup before post-processing ─────────────────────
    print("Cleaning up inference memory...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
            
    # Post-processing
    sample_img = images[0]
    h, w = sample_img.shape[-2], sample_img.shape[-1]
    predictions, images_cpu = postprocess(predictions, image_size_hw=(h, w), images=None)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 
        f"pred-{os.path.basename(os.path.normpath(args.image_folder))}-"
        f"f{args.first_k}-s{args.num_scale_frames}-w{args.kv_cache_sliding_window}-"
        f"q{args.quant_wa}-KV-fp8{args.kv_cache_fp8}-cut{args.kv_cache_cut}.pt")
    
    save_data = {
        'predictions': predictions,
        'paths': paths,
        'resolved_image_folder': resolved_image_folder,
        'output_path': output_path,
        'image_shape': (h, w),
        'args': vars(args),
    }
    
    if images_cpu is not None:
        save_data['images'] = images_cpu
    else:
        print(f"images_cpu is None, saving only predictions.")
    
    torch.save(save_data, output_path)
    print(f"Saved prediction to {output_path} ")

if __name__ == "__main__":
    main()
