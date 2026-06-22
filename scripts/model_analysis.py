"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

1. torchinfo summary 可用

2. torchvista trace_model 可用

3. torch.onnx 导出 ONNX 失败

4. captum IG 分析仅能跑一下功能，分析结果意义不大，且可以看出 8G 显存不足以支撑正常的分析：

    python scripts/model_analysis.py --model_path ../models/lingbot-map-long.pt \
        --output_dir ./output/model_analysis/ --first_k 1 --num_scale_frames 1 \
            --kv_cache_sliding_window 1 --camera_num_iterations 1 --ig

    为分析 IG， forward 需做如下改造：

            if 0:
                predictions.update(self._predict_depth(
                    aggregated_tokens_list, images, patch_start_idx,
                    gather_outputs=gather_outputs,
                ))

                predictions.update(self._predict_points(
                    aggregated_tokens_list, images, patch_start_idx,
                    gather_outputs=gather_outputs,
                ))

                predictions.update(self._predict_local_points(
                    aggregated_tokens_list, images, patch_start_idx,
                    gather_outputs=gather_outputs,
                ))

                if not self.training:
                    predictions["images"] = images

            # return predictions['depth'].flatten(start_dim=1).sum(dim=1)
            return predictions['pose_enc'].flatten(start_dim=1).sum(dim=1)
"""

import argparse
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

import torch

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.loadimage import load_images,LazyImageWrapper

from torchao.quantization import (
    quantize_, 
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig, 
    Float8WeightOnlyConfig,
    Float8DynamicActivationFloat8WeightConfig, 
    Int8DynamicActivationInt8WeightConfig,
    Float8DynamicActivationInt4WeightConfig # 不支持 hqq，无法绕过 mslk，所以用不了
    )
from torchao.utils import get_model_size_in_bytes

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
        gqa_ratio=1,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        # 1. 先在CPU加载权重，避免GPU显存峰值，注意mmap选项，小心CPU内存都不够
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False, mmap=True)
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
    parser.add_argument("--first_k", type=int, default=None, required=True,
                        help="If specified, only process the first k frames. "
                             "Specify the length for dummy_input.")
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
    parser.add_argument("--quant_wa", type=str, default="none", choices=["none", "mix", "fp8", "int8", "fp8w", "int8w", "int4w"],
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
    
    # Analysis
    parser.add_argument("--summary", action="store_true", default=False,
                        help="")
    parser.add_argument("--trace", action="store_true", default=False,
                        help="")
    parser.add_argument("--onnx", action="store_true", default=False,
                        help="")   
    parser.add_argument("--ig", action="store_true", default=False,
                        help="IntegratedGradients")   
    

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    num_frames = args.first_k
    model = load_model(args, device, num_frames)
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

    print(f"[ 量化前模型占用内存: {get_model_size_in_bytes(model) / (1024**3):.2f} GB ]") if args.quant_wa != "none" else None

    if args.quant_wa == "mix":
        # 5 - torchao 的权重量化 INT4，必须放在在 model.aggregator 转 bf16 之后
        config = Int4WeightOnlyConfig(
            group_size=32,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq"  # ✅ 选择 HQQ 算法，绕过 mslk
        )
        quantize_(model.aggregator, config)
        quantize_(model.camera_head, Float8DynamicActivationFloat8WeightConfig())
        quantize_(model.depth_head, Float8DynamicActivationFloat8WeightConfig())
    elif args.quant_wa == "fp8":
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
    elif args.quant_wa == "int8":
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
    elif args.quant_wa == "fp8w":
        quantize_(model, Float8WeightOnlyConfig())
    elif args.quant_wa == "int8w":
        quantize_(model, Int8WeightOnlyConfig())
    elif args.quant_wa == "int4w":
        config = Int4WeightOnlyConfig(
            group_size=32,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq"
        )
        quantize_(model.aggregator, config)
        # camera_head 和 depth_head 保留 float32，无法转换为 int4


    print(f"[ 量化后模型占用内存: {get_model_size_in_bytes(model) / (1024**3):.2f} GB ]") if args.quant_wa != "none" else None

    torch.cuda.reset_peak_memory_stats(device)

    dummy_input = torch.randn(num_frames, 3, 518, 294, dtype=torch.bfloat16).cuda()

    if args.summary:
        print("Summary...")
        from torchinfo import summary
        from contextlib import redirect_stdout
        summary_path = os.path.join(args.output_dir, "model_summary.txt")
        with open(summary_path, "w") as f:
            with redirect_stdout(f):
                summary(model, input_data=dummy_input, depth=10)
        print(f"Summary done. [{summary_path}]")        
    else:
        print("Skip summary.")

    if args.trace:
        print("Trace...")
        from torchvista import trace_model
        trace_path = os.path.join(args.output_dir, "model_trace.html")
        trace_model(
            model,
            inputs=dummy_input,
            export_path=trace_path
        )
        print(f"Trace done. [{trace_path}]")
    else:
        print("Skip trace.")

    if args.onnx:
        print("Export ONNX...")
        import torch.onnx as onnx
        onnx_path = os.path.join(args.output_dir, "model.onnx")
        onnx.export(
            model,
            dummy_input,
            onnx_path,
            dynamo=True
        )
        print(f"Export ONNX done. [{onnx_path}]")
    else:
        print("Skip export ONNX.")

    if args.ig:
        print("IntegratedGradients...")
        from captum.attr import IntegratedGradients
        import numpy as np
        from torch.utils.checkpoint import checkpoint

        class CheckpointWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return checkpoint(self.model, x, use_reentrant=False)
        
        model_ckpt = CheckpointWrapper(model).cuda()

        torch.manual_seed(123)
        np.random.seed(123)
        input = torch.rand(num_frames, 3, 518, 294, dtype=torch.bfloat16).cuda()
        baseline = torch.zeros(num_frames, 3, 518, 294, dtype=torch.bfloat16).cuda()
        ig = IntegratedGradients(model_ckpt)
        attributions, delta = ig.attribute(
            input, baseline, 
            return_convergence_delta=True,
            n_steps=2,
            internal_batch_size=1,
            )
        print('IG Attributions:', attributions)
        print('Convergence Delta:', delta)
    else:
        print("Skip IntegratedGradients.")

    print(f"[ allocated 显存峰值：{torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB ]")

if __name__ == "__main__":
    main()
