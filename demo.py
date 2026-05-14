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
from lingbot_map.utils.load_fn import load_and_preprocess_images

try:
    import rerun as rr
except ImportError:
    rr = None

# =============================================================================
# Lazy Image Loader
# =============================================================================

class ImageLazyLoader:
    """Lazy loader for images - loads and preprocesses on demand.
    
    Designed for windowed inference where only a subset of images
    are needed at any given time, reducing memory usage for long sequences.
    """
    def __init__(self, paths, image_size=518, patch_size=14, max_height=None, mode="crop"):
        self.paths = paths
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_height = max_height
        self.mode = mode
        self._cache = {}  # Cache for recently used frames
        self._cache_size = 16  # Cache last N frames
        self._access_order = []  # Track access order for LRU eviction
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        """Load and preprocess a single image or slice of images."""
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self.paths))))
            result = torch.stack([self._load_single(i) for i in indices])
            # Explicitly free intermediate tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return result
        elif isinstance(idx, (list, tuple)):
            result = torch.stack([self._load_single(i) for i in idx])
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return result
        else:
            return self._load_single(idx)
    
    def _load_single(self, idx):
        """Load and preprocess a single image with proper cache management."""
        # Check cache first
        if idx in self._cache:
            # Update access order for LRU
            if idx in self._access_order:
                self._access_order.remove(idx)
            self._access_order.append(idx)
            return self._cache[idx]
        
        path = self.paths[idx]
        img_tensor = load_and_preprocess_images(
            [path],
            mode=self.mode,
            image_size=self.image_size,
            patch_size=self.patch_size,
            param_max_height=self.max_height,
        )
        
        # Proper LRU cache eviction
        if len(self._cache) >= self._cache_size:
            # Remove least recently used entry
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    # Explicitly delete tensor to free memory
                    del self._cache[oldest_key]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Add to cache
        squeezed = img_tensor.squeeze(0)  # Remove batch dimension
        self._cache[idx] = squeezed
        self._access_order.append(idx)
        
        # Free the original tensor
        del img_tensor
        
        return squeezed
    
    def clear_cache(self):
        """Explicitly clear the cache to free memory."""
        self._cache.clear()
        self._access_order.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_batch(self, start, end):
        """Get a batch of images from start to end index."""
        return self[start:end]
    
# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png,.JPG",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8,
                rotate_clockwise_90=False, max_height=None, use_lazy_loader=False):
    """Load images from folder or video and preprocess into a tensor.

    Returns:
        (images, paths, resolved_image_folder): preprocessed tensor or lazy loader, 
        file paths, and the folder containing the source images (for sky mask caching etc.).
    """
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    if rotate_clockwise_90:
        rotated_dir = tempfile.mkdtemp(prefix="lingbot_rot_cw90_")
        rotated_paths = []
        # Image.ROTATE_270 = lossless 90° clockwise (270° counter-clockwise) reordering.
        for p in tqdm(paths, desc="Rotating images 90° CW"):
            out_path = os.path.join(rotated_dir, os.path.basename(p))
            Image.open(p).transpose(Image.ROTATE_270).save(out_path)
            rotated_paths.append(out_path)
        paths = rotated_paths
        resolved_folder = rotated_dir
        print(f"Rotated {len(paths)} images 90° clockwise → {rotated_dir}")

    if use_lazy_loader:
        print(f"Creating lazy loader for {len(paths)} images...")
        lazy_loader = ImageLazyLoader(
            paths,
            image_size=image_size,
            patch_size=patch_size,
            max_height=max_height,
        )
        h, w = lazy_loader[0].shape[-2:]
        print(f"Images will be preprocessed to {w}x{h} using canonical crop mode (lazy)")
        return lazy_loader, paths, resolved_folder
    else:
        print(f"Loading {len(paths)} images...")
        images = load_and_preprocess_images(
            paths,
            mode="crop",
            image_size=image_size,
            patch_size=patch_size,
            param_max_height=max_height,
        )
        h, w = images.shape[-2:]
        print(f"Preprocessed images to {w}x{h} using canonical crop mode")
        return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    # For windowed mode, pass window_size to optimize KV cache allocation
    window_size = None
    if getattr(args, "mode", "streaming") == "windowed":
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


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

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

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

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
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
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
        default=False,
        help="Offload per-frame predictions to CPU during inference to cut GPU peak memory "
            "(on by default).  Use --no-offload_to_cpu to keep outputs on GPU.",
    )
    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16,
                        help="Overlap between windows in *actual frames*")
    parser.add_argument("--overlap_keyframes", type=int, default=None,
                        help="Overlap expressed in *keyframes* (takes precedence over "
                             "--overlap_size). Converted internally to "
                             "max(num_scale_frames, overlap_keyframes * keyframe_interval) "
                             "actual frames.  Recommended when --keyframe_interval > 1.")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export stride-sampled, resized/cropped images to this folder")

    parser.add_argument("--max_height", type=int, default=None,
                        help="Maximum height for images. If specified, images taller than this will be center-cropped.")

    parser.add_argument("--lazyloader", action="store_true", default=False,
                        help="Use lazy loading for images to reduce memory usage. "
                             "Recommended for long sequences (>200 frames).")

    parser.add_argument("--offline_rerun", type=str, default=None,
                        help="Path to save .rrd file. If set, skips post-processing/visualization and saves results via Rerun SDK.")
    
    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    use_lazy = args.lazyloader
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
        max_height=args.max_height,
        use_lazy_loader=use_lazy,
    )

    # Export preprocessed images if requested
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

    if args.mode == "streaming":
        images = images.to(device)

    num_frames = len(images) if hasattr(images, '__len__') else images.shape[0]
    print(f"Input: {num_frames} frames")
    if hasattr(images, 'shape'):
        print(f"Shape: {tuple(images.shape)}")
    print(f"Mode: {args.mode}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU mem after load: "
            f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    # ── Debug: GPU Memory Breakdown ──────────────────────────────────────────
    if torch.cuda.is_available():
        def get_tensor_size_mb(t):
            if isinstance(t, torch.Tensor):
                return t.element_size() * t.nelement() / (1024 ** 2)
            return 0

        # 1. Calculate Model Size
        model_params_mb = sum(get_tensor_size_mb(p) for p in model.parameters())
        model_buffers_mb = sum(get_tensor_size_mb(b) for b in model.buffers())
        
        # 2. Calculate Input Images Size
        images_mb = get_tensor_size_mb(images)
        
        # 3. Get Total Allocated
        total_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # 4. Estimate "Other" (Activations, CUDA Context, Fragmentation, etc.)
        other_mb = total_allocated_mb - model_params_mb - model_buffers_mb - images_mb

        print("\n--- GPU Memory Breakdown (Before Inference) ---")
        print(f"  Model Parameters : {model_params_mb:.2f} MB")
        print(f"  Model Buffers    : {model_buffers_mb:.2f} MB")
        if isinstance(images, torch.Tensor) and images.device.type == 'cuda':
            print(f"  Input Images     : {images_mb:.2f} MB ({num_frames} frames, on GPU)")
        elif isinstance(images, torch.Tensor):
            print(f"  Input Images     : 0 MB on GPU, but {images_mb:.2f} MB ({num_frames} frames on CPU)")
        else:
            print(f"  Input Images     : Lazy loader ({num_frames} frames on disk - will load per-window)")         
        print(f"  -------------------------------------------")
        print(f"  Total Allocated  : {total_allocated_mb:.2f} MB ({total_allocated_mb/1024:.2f} GB)")
        print(f"  Total Reserved   : {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
        print("------------------------------------------------\n")

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
            print(
                f"Auto-selected --keyframe_interval={args.keyframe_interval} "
                f"(num_frames={num_frames} > 320)."
            )
        else:
            args.keyframe_interval = 1

    if args.keyframe_interval > 1:
        if args.mode == "streaming":
            print(
                f"Keyframe streaming enabled: interval={args.keyframe_interval} "
                f"(after the first {args.num_scale_frames} scale frames)."
            )
        else:  # windowed
            actual_per_window = (
                args.num_scale_frames
                + max(0, args.window_size - args.num_scale_frames) * args.keyframe_interval
            )
            print(
                f"Keyframe windowed enabled: interval={args.keyframe_interval}, "
                f"each window covers up to {actual_per_window} actual frames "
                f"(window_size={args.window_size} keyframes, scale={args.num_scale_frames})."
            )

    # ── Optional: torch.compile + CUDA-graph warmup (streaming only) ────────
    if args.compile:
        if args.mode != "streaming":
            print(
                f"--compile only applies to --mode streaming (got {args.mode!r}); "
                "skipping compile."
            )
        else:
            scale_for_warm = min(args.num_scale_frames, num_frames)
            if scale_for_warm >= num_frames:
                scale_for_warm = max(1, num_frames - 1)
            warm_stream_n = min(10, max(1, num_frames - scale_for_warm))
            warm_h, warm_w = int(images.shape[-2]), int(images.shape[-1])
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
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    
    # Track if we used lazy loader
    used_lazy_loader = hasattr(images, 'get_batch')
    images_wrapper = None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:  # windowed
            # For windowed mode with lazy loader, we need to wrap it appropriately
            if used_lazy_loader:
                # Lazy loader case - create a wrapper that mimics tensor behavior
                class LazyImageWrapper:
                    def __init__(self, loader, device, image_size, patch_size, max_height=None):
                        self.loader = loader
                        self.device = device
                        self.image_size = image_size
                        self.patch_size = patch_size
                        self.max_height = max_height
                        # Get actual dimensions from first image
                        sample = loader[0]
                        self._h, self._w = sample.shape[-2:]
                        # Shape should be [B, S, C, H, W] for compatibility
                        self.shape = (1, len(loader), 3, self._h, self._w)
                    
                    def __getitem__(self, idx):
                        """Support slicing like tensor[idx] -> returns [B, S, C, H, W]"""
                        # Handle tuple indexing like images[:, start:end]
                        if isinstance(idx, tuple):
                            # We expect (batch_slice, seq_slice) where batch_slice is usually ':'
                            if len(idx) == 2:
                                batch_idx, seq_idx = idx
                                # batch_idx should be slice(None) which means all batches (we have B=1)
                                if isinstance(seq_idx, slice):
                                    start = seq_idx.start if seq_idx.start is not None else 0
                                    stop = seq_idx.stop if seq_idx.stop is not None else len(self.loader)
                                    step = seq_idx.step if seq_idx.step is not None else 1
                                    indices = list(range(start, stop, step))
                                    batch = torch.stack([self.loader[i] for i in indices])
                                    # batch shape: [S, C, H, W], add batch dim -> [B, S, C, H, W]
                                    return batch.unsqueeze(0).to(self.device)
                                elif isinstance(seq_idx, int):
                                    single = self.loader[seq_idx]
                                    return single.unsqueeze(0).unsqueeze(0).to(self.device)
                                else:
                                    raise IndexError(f"Unsupported sequence index type: {type(seq_idx)}")
                            else:
                                raise IndexError(f"Expected 2D tuple index, got {len(idx)}D")
                        elif isinstance(idx, slice):
                            # Direct slice on sequence dimension
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
                        """Return self for compatibility - shape already includes batch dim"""
                        return self
                    
                    def to(self, device):
                        """Return self for compatibility - we handle device in __getitem__"""
                        return self
                    
                    @property
                    def ndim(self):
                        return 5
                
                images_wrapper = LazyImageWrapper(
                    images, 
                    device, 
                    images.image_size, 
                    images.patch_size,
                    images.max_height
                )
                predictions = model.inference_windowed(
                    images_wrapper,
                    window_size=args.window_size,
                    overlap_size=args.overlap_size,
                    overlap_keyframes=args.overlap_keyframes,
                    num_scale_frames=args.num_scale_frames,
                    keyframe_interval=args.keyframe_interval,
                    output_device=output_device
                )
                
                # Remove the LazyImageWrapper from predictions since it's not a real tensor
                if "images" in predictions:
                    del predictions["images"]
            else:
                predictions = model.inference_windowed(
                    images,
                    window_size=args.window_size,
                    overlap_size=args.overlap_size,
                    overlap_keyframes=args.overlap_keyframes,
                    num_scale_frames=args.num_scale_frames,
                    keyframe_interval=args.keyframe_interval,
                    output_device=output_device
                )

    print(f"Inference done in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(
            f"GPU peak during inference: "
            f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB "
            f"(reserved peak {torch.cuda.max_memory_reserved()/1e9:.2f} GB)"
        )

    # ── Aggressive memory cleanup before post-processing ─────────────────────
    print("Cleaning up inference memory...")
    
    # Delete model to free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if 0:
        # For lazy loader, delete the wrapper and loader to free cache
        if used_lazy_loader:
            del images_wrapper
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # For non-lazy mode, delete images tensor
        elif isinstance(images, torch.Tensor):
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "CPU-only mode")
            
    # ── Post-process ─────────────────────────────────────────────────────────
    # Reconstruct images only if needed for visualization
    if used_lazy_loader:
        print("Loading images from disk for visualization (lazy mode)...")
        # Load images one by one to avoid peak memory spike
        lazy_loader = ImageLazyLoader(
            paths,
            image_size=args.image_size,
            patch_size=args.patch_size,
            max_height=args.max_height,
        )
        images_for_post = torch.stack([lazy_loader[i] for i in range(len(lazy_loader))])
        del lazy_loader  # Free the loader
    else:
        # For non-lazy mode, we need to reload or use predictions
        images_for_post = predictions.get("images")
        if images_for_post is None:
            print("Warning: No images in predictions, visualization may be limited")
            images_for_post = torch.zeros((len(paths), 3, args.image_size, 
                                          int(args.image_size * 294 / 518)))

    predictions, images_cpu = postprocess(predictions, images_for_post)

    # ── Offline Rerun Save ───────────────────────────────────────────────────
    if args.offline_rerun:
        if rr is None:
            raise ImportError("rerun-sdk is required for --offline_rerun. Install with: pip install rerun-sdk")
        
        print(f"Saving results to Rerun file: {args.offline_rerun}")
        rr.init("lingbot_map_offline", spawn=False)
        rr.save(args.offline_rerun)

        h, w = int(images_cpu.shape[-2]), int(images_cpu.shape[-1])
        extrinsic = predictions["extrinsic"]
        intrinsic = predictions["intrinsic"]
        
        num_frames_pred = extrinsic.shape[0]
        
        print(f"[DEBUG] num_frames_pred: {num_frames_pred}")
        print(f"[DEBUG] extrinsic shape: {extrinsic.shape}")
        print(f"[DEBUG] intrinsic shape: {intrinsic.shape}")
        print(f"[DEBUG] predictions keys: {list(predictions.keys())}")
        
        if "depth" in predictions:
            print(f"[DEBUG] depth shape: {predictions['depth'].shape}")

        from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
        
        depth_for_unproject = predictions.get("depth")
    
        if depth_for_unproject is not None:
            print(f"[DEBUG] Computing world points from depth...")
            print(f"[DEBUG] depth_for_unproject shape before processing: {depth_for_unproject.shape}")
            
            # 点云反投影使用 c2w (camera-to-world)
            extrinsic_c2w = extrinsic.cpu().numpy()  # (S, 3, 4) c2w
            
            world_points = unproject_depth_map_to_point_map(
                depth_for_unproject.cpu().numpy(),
                extrinsic_c2w,  # 传入 c2w
                intrinsic.cpu().numpy()
            )
            print(f"[DEBUG] world_points shape: {world_points.shape}")
        else:
            world_points = None
            print("[WARNING] No depth available for point cloud computation")
        
        for i in tqdm(range(num_frames_pred), desc="Logging to Rerun"):
            # 最新版正确写法：用 rr.set_time + sequence 关键字参数
            rr.set_time("frame_idx", sequence=i)  # 帧序号用 sequence
            # 等价于旧版 set_time_sequence("frame_idx", i)，但接口统一
            
            # 相机使用 w2c (world-to-camera) 变换
            cam_transform_c2w = extrinsic[i].cpu().numpy()

            # 构建完整的 4x4 矩阵
            transform_4x4_c2w = np.eye(4)
            transform_4x4_c2w[:3, :] = cam_transform_c2w
            
            # 求逆得到 w2c
            transform_4x4_w2c = np.linalg.inv(transform_4x4_c2w)

            # 提取平移和旋转
            translation = transform_4x4_w2c[:3, 3]
            rotation_matrix = transform_4x4_w2c[:3, :3]

            # 转换为四元数 (xyzw 格式)
            from scipy.spatial.transform import Rotation as R
            rot = R.from_matrix(rotation_matrix)
            quat_xyzw = rot.as_quat()

            # 记录相机变换（使用 w2c）
            rr.log(f"world/camera/{i}", rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=quat_xyzw)
            ))
            
            intr = intrinsic[i].cpu().numpy()
            rr.log(f"world/camera/{i}", rr.Pinhole(
                image_from_camera=intr,
                width=w,
                height=h
            ))

            if 0:
            #if "depth" in predictions:
                depth = predictions["depth"][i].cpu().numpy()
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    depth = depth.squeeze(-1)
                rr.log(f"world/camera/{i}/depth", rr.DepthImage(depth))

            if world_points is not None:
                points = world_points[i]
                
                # Get the corresponding color image for this frame
                colors_for_points = None
                if images_cpu is not None:
                    img = images_cpu[i].numpy()
                    if img.ndim == 3:
                        # img shape: (C, H, W), convert to (H, W, C)
                        img_hwc = np.transpose(img, (1, 2, 0))
                        
                        # Check if image dimensions match point cloud dimensions
                        if img_hwc.shape[:2] == points.shape[:2]:
                            # Dimensions match, can use directly
                            colors_for_points = img_hwc
                        else:
                            # Dimensions don't match, need to resize image to match depth map
                            import cv2
                            img_resized = cv2.resize(
                                img_hwc, 
                                (points.shape[1], points.shape[0]),
                                interpolation=cv2.INTER_LINEAR
                            )
                            colors_for_points = img_resized
                
                conf = predictions.get("depth_conf", None)
                if conf is not None:
                    conf_i = conf[i].cpu().numpy()
                    mask = conf_i > args.conf_threshold
                    
                    # Filter points by confidence
                    points_filtered = points[mask]
                    
                    # Extract colors using the same mask
                    if colors_for_points is not None:
                        colors_filtered = colors_for_points[mask]
                        # Convert to uint8 RGB
                        colors_uint8 = (np.clip(colors_filtered, 0, 1) * 255).astype(np.uint8)
                    else:
                        colors_uint8 = None
                else:
                    valid_mask = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
                    points_filtered = points[valid_mask]
                    
                    if colors_for_points is not None:
                        colors_filtered = colors_for_points[valid_mask]
                        colors_uint8 = (np.clip(colors_filtered, 0, 1) * 255).astype(np.uint8)
                    else:
                        colors_uint8 = None
                
                if points_filtered.size > 0:
                    rr.log(f"world/points/{i}", rr.Points3D(
                        positions=points_filtered,
                        colors=colors_uint8 if colors_uint8 is not None else None,
                        radii=0.01
                    ))

        rr.disconnect()
        print(f"Saved to {args.offline_rerun}")
        return

    # Free the reconstructed images tensor
    del images_for_post
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Memory ready for visualization. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "CPU-only mode")

    # ── Memory analysis for predictions and images ───────────────────────────
    def get_tensor_memory_mb(tensor):
        """Calculate memory usage of a tensor in MB."""
        if isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.nelement() / (1024 ** 2)
        elif isinstance(tensor, np.ndarray):
            return tensor.nbytes / (1024 ** 2)
        return 0
    
    print("\n=== Memory Analysis for Visualization ===")
    total_predictions_mb = 0
    for key, value in predictions.items():
        mem_mb = get_tensor_memory_mb(value)
        total_predictions_mb += mem_mb
        if hasattr(value, 'shape'):
            print(f"  {key:25s}: {mem_mb:8.2f} MB | shape: {value.shape}")
        else:
            print(f"  {key:25s}: {mem_mb:8.2f} MB | type: {type(value).__name__}")
    
    images_mem_mb = get_tensor_memory_mb(images_cpu)
    print(f"  {'images':25s}: {images_mem_mb:8.2f} MB | shape: {images_cpu.shape if hasattr(images_cpu, 'shape') else 'N/A'}")
    total_predictions_mb += images_mem_mb
    
    print(f"\n  {'Total predictions + images':25s}: {total_predictions_mb:8.2f} MB ({total_predictions_mb/1024:.2f} GB)")
    
    # Estimate visualization memory (viser will create additional structures)
    # viser typically needs 2-3x the raw data size for rendering buffers
    estimated_vis_multiplier = 2.5
    estimated_total_mb = total_predictions_mb * estimated_vis_multiplier
    print(f"\n  Estimated visualization memory (with {estimated_vis_multiplier}x overhead):")
    print(f"    {estimated_total_mb:8.2f} MB ({estimated_total_mb/1024:.2f} GB)")
    
    # Check available system memory
    try:
        import psutil
        avail_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"\n  Available system memory: {avail_mem_gb:.2f} GB")
        if estimated_total_mb / 1024 > avail_mem_gb * 0.8:
            print(f"  ⚠️  WARNING: Estimated memory usage exceeds 80% of available memory!")
            print(f"     Consider reducing --downsample_factor or --conf_threshold")
    except ImportError:
        print("\n  (Install psutil for system memory monitoring: pip install psutil)")
    
    print("=" * 50 + "\n")

    # ── Visualize ────────────────────────────────────────────────────────────
    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        )
        print(f"3D viewer at http://localhost:{args.port}")
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
