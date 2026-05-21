"""Utility functions for loading images in LingBot-MAP.

This module provides shared image loading functionality used by both
predict.py (inference) and postprocess.py (post-processing).
"""
import glob
import os
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from lingbot_map.utils.load_fn import load_and_preprocess_images


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
        self._cache = {}
        self._cache_size = 16
        self._access_order = []
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        """Load and preprocess a single image or slice of images."""
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self.paths))))
            result = torch.stack([self._load_single(i) for i in indices])
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
        if idx in self._cache:
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
        
        if len(self._cache) >= self._cache_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        squeezed = img_tensor.squeeze(0)
        self._cache[idx] = squeezed
        self._access_order.append(idx)
        
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
        file paths, and the folder containing the source images.
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