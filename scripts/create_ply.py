"""LingBot-MAP post-processing script: load predictions and visualize.

This script loads prediction results saved by predict.py and performs
post-processing, visualization, and optional TSDF fusion.

Usage:
    python create_ply.py --pred_path ./output_indoor/predictions.pt \
        --conf_threshold 5 --use_fusion --voxel_size 0.012

    python create_ply.py --pred_path ./output_indoor/predictions.pt \
        --conf_threshold 5 --use_fusion --voxel_size 0.012
"""
import faulthandler
faulthandler.enable()

import argparse
import os
import time
import gc

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

# Import load_images to reuse lazy loading logic
try:
    from lingbot_map.utils.loadimage import load_images
except ImportError:
    load_images = None


# =============================================================================
# Utility functions
# =============================================================================



def get_colors_for_points(img_tensor_or_array, points_shape):
    """
    Extract colors for points from image, handling dimension mismatches.
    
    Args:
        img_tensor_or_array: Image as tensor (C, H, W) or numpy array
        points_shape: Target shape (H, W) for the points
    
    Returns:
        colors: RGB colors in shape (H, W, 3) or None
    """
    if hasattr(img_tensor_or_array, 'numpy'):
        img_numpy = img_tensor_or_array.numpy()
    else:
        img_numpy = img_tensor_or_array
    
    if img_numpy.ndim != 3:
        return None
    
    img_hwc = np.transpose(img_numpy, (1, 2, 0))
    
    if img_hwc.shape[:2] == points_shape[:2]:
        return img_hwc
    else:
        img_resized = cv2.resize(
            img_hwc, 
            (points_shape[1], points_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        return img_resized


def batch_unproject_frames(depth_batch, extrinsic_batch, intrinsic_batch):
    """
    Unproject a batch of frames' depth maps to world coordinates efficiently.
    
    Args:
        depth_batch: Batch of depth maps (N, H, W, 1) or (N, H, W)
        extrinsic_batch: Batch of extrinsic matrices (N, 3, 4)
        intrinsic_batch: Batch of intrinsic matrices (N, 3, 3)
    
    Returns:
        world_points_batch: World coordinates (N, H, W, 3)
    """
    # Convert to numpy if needed
    if isinstance(depth_batch, torch.Tensor):
        depth_batch = depth_batch.cpu().numpy()
    if isinstance(extrinsic_batch, torch.Tensor):
        extrinsic_batch = extrinsic_batch.cpu().numpy()
    if isinstance(intrinsic_batch, torch.Tensor):
        intrinsic_batch = intrinsic_batch.cpu().numpy()
    
    from lingbot_map.utils.geometry import depth_to_world_coords_points
    
    num_frames = depth_batch.shape[0]
    h, w = depth_batch.shape[1], depth_batch.shape[2]
    
    # Pre-allocate output array
    world_points_batch = np.empty((num_frames, h, w, 3), dtype=np.float32)
    
    # Process each frame (still need loop due to per-frame camera params)
    for i in range(num_frames):
        depth_frame = depth_batch[i].squeeze(-1) if depth_batch.ndim == 4 else depth_batch[i]
        world_points, _, _ = depth_to_world_coords_points(
            depth_frame, 
            extrinsic_batch[i], 
            intrinsic_batch[i]
        )
        world_points_batch[i] = world_points
    
    return world_points_batch


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Post-processing and Visualization")

    # Input
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to predictions.pt file saved by predict.py")

    parser.add_argument("--conf_threshold", type=float, default=1.5)

    # Fusion options (Fusion is now always enabled)
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size for Fusion in meters (default: 0.05 = 5cm)")
    parser.add_argument("--sdf_trunc_multiplier", type=float, default=3.0,
                        help="SDF truncation distance multiplier (default: 3.0 * voxel_size)")
    parser.add_argument("--depth_max", type=float, default=10.0,
                        help="Maximum depth for fusion (default: 10.0)")
    
    # Add option to limit the number of frames processed
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to process (default: None, process all)")
    
    # Performance optimization options
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for processing frames (default: 50)")

    args = parser.parse_args()

    # ── Load predictions ─────────────────────────────────────────────────────
    print(f"Loading predictions from: {args.pred_path}")
    t0 = time.time()
    
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_path}")
    
    save_data = torch.load(args.pred_path, map_location="cpu", weights_only=False)
    

    predictions = save_data['predictions']
    paths = save_data.get('paths', None)
    resolved_image_folder = save_data.get('resolved_image_folder', None)
    image_shape = save_data.get('image_shape', None)
    loaded_args = save_data.get('args', {})
    

    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Prediction keys: {list(predictions.keys())}")
    
    # Determine output rrd path based on resolved_image_folder
    if resolved_image_folder is not None:
        folder_name = os.path.basename(resolved_image_folder)
        # Remove potential trailing slashes or spaces
        folder_name = folder_name.strip().rstrip('/')
        if not folder_name:
            folder_name = "output"
        offline_rerun = f"{folder_name}.rrd"
    else:
        offline_rerun = "output.rrd"
    

    # Initialize lazy image loader (Mandatory as per requirement)
    lazy_image_loader = None
    if load_images is not None and resolved_image_folder is not None:
        print("Initializing lazy image loader...")
        try:
            # Retrieve preprocessing parameters from saved args to ensure consistency
            img_size = loaded_args.get('image_size', 518)
            patch_size = loaded_args.get('patch_size', 14)
            max_height = loaded_args.get('max_height', None)
            
            print(f"  Image folder: {resolved_image_folder}")
            print(f"  Image size: {img_size}, Patch size: {patch_size}")
            
            # Create lazy loader
            lazy_image_loader, _, _ = load_images(
                image_folder=resolved_image_folder,
                image_size=img_size,
                patch_size=patch_size,
                max_height=max_height,
                use_lazy_loader=True,
                stride=loaded_args.get('stride', 1),
                first_k=loaded_args.get('first_k', None),
            )
            print(f"✓ Lazy loader initialized for {len(lazy_image_loader)} images.")
        except Exception as e:
            print(f"[Error] Failed to initialize lazy loader: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Lazy loader initialization failed. Cannot proceed without images.")
    else:
        raise RuntimeError("Image folder path not found in predictions or load_images module missing. Cannot initialize lazy loader.")
    
    if 'depth' in predictions:
        print(f"Depth shape: {predictions['depth'].shape}")
    if 'extrinsic' in predictions:
        print(f"Extrinsic shape: {predictions['extrinsic'].shape}")
    if 'intrinsic' in predictions:
        print(f"Intrinsic shape: {predictions['intrinsic'].shape}")
    
    # Determine image dimensions
    if image_shape is not None:
        h, w = image_shape
    elif lazy_image_loader is not None:
        # Infer shape from lazy loader if possible, otherwise fallback
        # Note: lazy_loader might not expose shape easily without loading one item
        # We assume image_shape is present in save_data as per typical workflow
        raise ValueError("Cannot determine image dimensions. 'image_shape' missing from predictions.")
    else:
        raise ValueError("Cannot determine image dimensions")
    
    print(f"Image dimensions: {w}x{h}")
    print(f"Number of frames: {len(paths) if paths else 'unknown'}")

    # ── Offline Rerun Save ───────────────────────────────────────────────────

    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]
    
    num_frames_pred = extrinsic.shape[0]
    
    # Limit the number of frames if specified
    if args.max_frames is not None:
        num_frames_to_process = min(args.max_frames, num_frames_pred)
        print(f"Limiting processing to first {num_frames_to_process} frames (out of {num_frames_pred})")
        
        # Optimization: Slice tensors to release memory for unused frames
        # Cloning ensures the new tensor owns its storage, allowing GC to free the rest
        for key in ['extrinsic', 'intrinsic', 'depth', 'depth_conf']:
            if key in predictions:
                predictions[key] = predictions[key][:num_frames_to_process].clone()
        
        if paths:
            paths = paths[:num_frames_to_process]
            
        # Update local references after slicing
        extrinsic = predictions["extrinsic"]
        intrinsic = predictions["intrinsic"]
    else:
        num_frames_to_process = num_frames_pred
    
    depth_for_unproject = predictions.get("depth")


    # Decide whether to cache all world points or process on-the-fly
    if depth_for_unproject is not None:
        print(f"Using batched world point computation (optimized mode)...")
    else:
        print("[WARNING] No depth available for point cloud computation")
    

    # Initialize Fusion (Always enabled)
    try:
        import open3d as o3d
        sdf_trunc = args.sdf_trunc_multiplier * args.voxel_size

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        print(f"\n✅ Fusion initialized successfully")
        print(f"  Voxel size: {args.voxel_size} m")
        print(f"  SDF truncation: {sdf_trunc} m")
        print(f"  Max depth: {args.depth_max} m")
    except Exception as e:
        print(f"\n❌ Fusion initialization failed: {e}")
        raise RuntimeError("Fusion initialization failed. Cannot proceed.")
    
    # Pre-compute Open3D intrinsic (same for all frames if camera is calibrated)
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,
        fx=intrinsic[0, 0, 0].item(), 
        fy=intrinsic[0, 1, 1].item(),
        cx=intrinsic[0, 0, 2].item(), 
        cy=intrinsic[0, 1, 2].item()
    )
    
    # Pre-convert all extrinsics to numpy once
    print("Pre-converting camera parameters...")
    extrinsic_np = extrinsic.cpu().numpy().astype(np.float64)
    intrinsic_np = intrinsic.cpu().numpy().astype(np.float32)
    
    # Prepare depth and confidence arrays
    depth_np_all = None
    conf_np_all = None
    if depth_for_unproject is not None:
        depth_np_all = depth_for_unproject.cpu().numpy().astype(np.float32)
        if depth_np_all.ndim == 4 and depth_np_all.shape[-1] == 1:
            depth_np_all = depth_np_all.squeeze(-1)
        
        if "depth_conf" in predictions:
            conf_np_all = predictions["depth_conf"].cpu().numpy()
    
    # Release original tensors to save memory
    del extrinsic, intrinsic, depth_for_unproject
    if "depth" in predictions:
        del predictions["depth"]
    if "depth_conf" in predictions:
        del predictions["depth_conf"]
    gc.collect()
    
    print(f"Starting batched processing with batch_size={args.batch_size}...")
    t_start = time.time()
    
    # Process in batches for better performance
    batch_size = args.batch_size
    num_batches = (num_frames_to_process + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_frames_to_process)
        actual_batch_size = end_idx - start_idx
        
        # Get batch data
        depth_batch = depth_np_all[start_idx:end_idx] if depth_np_all is not None else None
        extrinsic_batch = extrinsic_np[start_idx:end_idx]
        intrinsic_batch = intrinsic_np[start_idx:end_idx]
        conf_batch = conf_np_all[start_idx:end_idx] if conf_np_all is not None else None
        
        # Compute world points for entire batch at once
        if depth_batch is not None:
            world_points_batch = batch_unproject_frames(
                depth_batch, extrinsic_batch, intrinsic_batch
            )
        else:
            continue
        
        # Process each frame in the batch
        for i in range(actual_batch_size):
            frame_idx = start_idx + i
            
            # Get pose for this frame
            pose_c2w = np.eye(4, dtype=np.float64)
            pose_c2w[:3, :] = extrinsic_batch[i]
            
            # Apply confidence mask
            depth_frame = depth_batch[i].copy()
            if conf_batch is not None:
                low_conf_mask = conf_batch[i] <= args.conf_threshold
                depth_frame[low_conf_mask] = 2.0 * args.depth_max
            
            # Get colors
            colors_frame = None
            if lazy_image_loader is not None:
                try:
                    img_frame = lazy_image_loader[frame_idx]
                    if img_frame is not None:
                        colors_frame = get_colors_for_points(img_frame, (h, w))
                except Exception as e:
                    pass
            
            if colors_frame is None:
                colors_frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                colors_frame = (np.clip(colors_frame, 0, 1) * 255).astype(np.uint8)
            
            # Ensure contiguous arrays for Open3D
            depth_frame = np.ascontiguousarray(depth_frame)
            colors_frame = np.ascontiguousarray(colors_frame)
            
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(colors_frame),
                o3d.geometry.Image(depth_frame),
                depth_scale=1.0,
                depth_trunc=args.depth_max,
                convert_rgb_to_intensity=False
            )
            
            # Integrate into TSDF volume
            try:
                volume.integrate(rgbd, intrinsic_o3d, pose_c2w)
            except Exception as e:
                print(f"[Fusion warning] Frame {frame_idx} failed: {str(e)}")
        
        # Clean up batch data
        del depth_batch, extrinsic_batch, intrinsic_batch, conf_batch
        if depth_np_all is not None:
            del world_points_batch
        
        # Garbage collection every few batches (not every frame)
        if batch_idx % 5 == 0:
            gc.collect()
    
    elapsed = time.time() - t_start
    fps = num_frames_to_process / elapsed if elapsed > 0 else 0
    print(f"\n✅ Processing completed: {num_frames_to_process} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    # Extract and log fused point cloud
    print("\nExtracting reconstructed model...")
    pcd_from_volume = volume.extract_point_cloud()

    output_ply_path = offline_rerun.replace(".rrd", ".ply")
    o3d.io.write_point_cloud(output_ply_path, pcd_from_volume)
    print(f"Fused point cloud saved to: {output_ply_path}")

    print(f"Fused Point cloud Num: {len(pcd_from_volume.points):,} ")
    if hasattr(pcd_from_volume, 'colors') and len(pcd_from_volume.colors) > 0:
        print(f"Fused Point cloud colors Num: {len(pcd_from_volume.colors):,} ")
    if hasattr(pcd_from_volume, 'normals') and len(pcd_from_volume.normals) > 0:
        print(f"Fused Point cloud normals Num: {len(pcd_from_volume.normals):,} ")

    # # 1. 先记录父实体的变换

    # # 2. 在子路径记录点云数据

    # Print summary statistics for point clouds
    if conf_np_all is not None:
        conf_mask = conf_np_all > args.conf_threshold
        confident_points_count = conf_mask.sum().item()
        total_points_count = depth_np_all.size if depth_np_all is not None else 0
        print(f"\n{'='*60}")
        print(f"Point Cloud Statistics Summary:")
        print(f"  Total points (all): {total_points_count:,}")
        print(f"  Confident points (> {args.conf_threshold}): {confident_points_count:,}")
        print(f"  Confidence ratio: {confident_points_count/total_points_count*100:.2f}%")
        print(f"  Fused points: {len(pcd_from_volume.points):,}")
        if confident_points_count > 0:
            print(f"  Fused/Confident ratio: {len(pcd_from_volume.points)/confident_points_count*100:.2f}%")
        if total_points_count > 0:
            print(f"\n  Fused/Total ratio: {len(pcd_from_volume.points)/total_points_count*100:.2f}%")
        print(f"{'='*60}\n")

    return

if __name__ == "__main__":
    main()
