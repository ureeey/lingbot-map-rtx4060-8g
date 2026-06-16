"""
FlashInfer KV Cache Manager — Two-Stream Paged Design with FP8 and GQA support.

"""

import collections
import math
from typing import List

import torch
from torch import Tensor

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


class FlashInferKVCacheManager:
    """
    Two-stream paged KV cache: patch pages (recyclable) + special pages (append-only).
    KV cache is stored in float8_e4m3fn (FP8) or bfloat16 depending on kv_cache_fp8.

    Args:
        num_blocks:          Number of Transformer blocks (one cache per block).
        max_num_frames:      Maximum frames held in the KV window at once
                             (scale_frames + sliding_window + headroom).
        tokens_per_frame:    Total tokens per frame = patches + specials (e.g. 262).
        num_heads:           Number of query heads (Q heads). KV heads = num_heads // gqa_ratio (GQA).
        head_dim:            Head dimension (64 for ViT-L).
        dtype:               Storage dtype (ignored, FP8 used always).
        device:              CUDA device.
        num_special_tokens:  Special tokens per frame: camera + register×N + scale (6).
        scale_frames:        Number of always-resident scale frames (8).
        sliding_window:      Sliding window size (64).
        max_total_frames:    Upper bound on total frames ever processed; used to
                             pre-allocate the special page pool (default 2048).
        kv_cache_fp8:        Whether to use FP8 for KV cache (default False, set True for FP8).
        kv_cache_cut:        Whether to use downsampling in 2D (default 1).
        gqa_ratio:           GQA ratio (default 1).
    """

    def __init__(
        self,
        num_blocks: int,
        max_num_frames: int,
        tokens_per_frame: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        num_special_tokens: int = 6,
        scale_frames: int = 8,
        sliding_window: int = 64,
        max_total_frames: int = 2048,
        force_fp32: bool = False, # deprecated
        fa3: bool = False,
        window_size: int = None,
        kv_cache_fp8: bool = False,
        kv_cache_cut: int = 1,
        gqa_ratio: int = 1,
    ):
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError("FlashInfer is not available. Please install flashinfer.")

        print(f"[ FlashInferKVCache 之前 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB ，已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB ]")

        self.num_blocks = num_blocks
        self.num_special_tokens = num_special_tokens         # 6
        self.raw_patches_per_frame = tokens_per_frame - num_special_tokens  # 256 / 999 / ...
        self.cache_patches_per_frame = self.raw_patches_per_frame

        if kv_cache_cut > 1:
            if self.raw_patches_per_frame == 777: # 37x21
                self.H_cut = math.ceil(21 / kv_cache_cut)
                self.W_cut = math.ceil(37 / kv_cache_cut)
                assert self.H_cut > 0 and self.W_cut > 0, (
                    f"H_cut={self.H_cut} or W_cut={self.W_cut} <= 0 for kv_cache_cut={kv_cache_cut}"
                )
                self.cache_patches_per_frame = self.H_cut * self.W_cut
                print(f"--- H_cut = {self.H_cut}, W_cut = {self.W_cut} ---")
        else:
            print(f"--- KV cache does not use downsampling ---")

        # Use exact page_size = patches_per_frame to eliminate zero-padded slots.
        # FA2 (backend="fa2") supports non-power-of-2 page sizes.
        # FA3 (sm90) requires power-of-2 page sizes; use next_power_of_2 when fa3=True.
        p = self.cache_patches_per_frame
        if fa3:
            # Round up to next power-of-2 for FA3 SM90 kernel requirement.
            # e.g. 999 → 1024 (25 zero-padded slots per patch page)
            self.page_size = 1 << (p - 1).bit_length()
        else:
            self.page_size = p  # exact: no zero padding in patch pages
        self.scale_frames = scale_frames                     # 8
        self.sliding_window = sliding_window                 # 64
        self.num_heads = num_heads                           # query heads
        self.num_kv_heads = num_heads // gqa_ratio           # GQA
        assert num_heads % gqa_ratio == 0, f"num_heads ({num_heads}) must be divisible by gqa_ratio ({gqa_ratio})"
        self.head_dim = head_dim
        self.tokens_per_frame = tokens_per_frame
        self.kv_cache_fp8 = kv_cache_fp8
        self.kv_cache_cut = kv_cache_cut
        self.gqa_ratio = gqa_ratio

        assert self.cache_patches_per_frame > 0, (
            f"tokens_per_frame={tokens_per_frame} <= num_special_tokens={num_special_tokens}"
        )
        assert self.page_size > 0

        self.q_dtype = torch.bfloat16
        if kv_cache_fp8:
            # Use FP8 for KV cache
            if not hasattr(torch, 'float8_e4m3fn'):
                raise RuntimeError("PyTorch float8_e4m3fn not available (need torch>=2.2)")
            self.kv_dtype = torch.float8_e4m3fn
        else:
            self.kv_dtype = torch.bfloat16

        self.device = device

        # ── Page pool sizing ─────────────────────────────────────────────────
        # Patch: scale + window + 16 headroom  (pages recycled → fixed count)
        max_patch_pages = scale_frames + sliding_window + 1   # e.g. 88
        # Special: enough for max_total_frames × 6 tokens, plus 16 headroom
        effective_max_total_frames = max_total_frames
        # Use window_size to cap max_total_frames if provided
        if window_size is not None:
            effective_max_total_frames = min(max_total_frames, window_size)
        else:
            effective_max_total_frames = max_total_frames
        max_special_pages = (
            math.ceil(effective_max_total_frames * num_special_tokens / self.page_size + 1)
        )
        self.max_patch_pages = max_patch_pages
        self.max_num_pages = max_patch_pages + max_special_pages
        
        if 1:
            import sys
            print(f"--- 页面池配置 ---", file=sys.stderr)
            print(f"  scale_frames (尺度帧数): {scale_frames}", file=sys.stderr)
            print(f"  sliding_window (滑动窗口): {sliding_window}", file=sys.stderr)
            print(f"  effective_max_total_frames (有效最大总帧数): {effective_max_total_frames}", file=sys.stderr)
            print(f"  num_special_tokens (每帧特殊token数): {num_special_tokens}", file=sys.stderr)
            print(f"  page_size (页大小): {self.page_size}", file=sys.stderr)
            print(f"  max_patch_pages (最大patch页数): {max_patch_pages} = {scale_frames} + {sliding_window} + 1", file=sys.stderr)
            print(f"  max_special_pages (最大special页数): {max_special_pages} = ceil({effective_max_total_frames} * {num_special_tokens} / {self.page_size} + 1)", file=sys.stderr)
            print(f"  max_num_pages (总页数): {self.max_num_pages} = {max_patch_pages} + {max_special_pages}", file=sys.stderr)
            bytes_per_element = torch.finfo(self.kv_dtype).bits // 8 if self.kv_dtype != torch.float32 else 4
            elements_per_block = self.max_num_pages * 2 * self.page_size * self.num_kv_heads * head_dim
            bytes_per_block = elements_per_block * bytes_per_element
            total_bytes = bytes_per_block * num_blocks
            total_mb = total_bytes / (1024 ** 2)
            
            print(f"--- 内存分配详情 ---", file=sys.stderr)
            print(f"  num_blocks (层数): {num_blocks}", file=sys.stderr)
            print(f"  max_num_pages (总页数): {self.max_num_pages} (patch页={self.max_patch_pages}, special页={self.max_num_pages - self.max_patch_pages})", file=sys.stderr)
            print(f"  page_size (每页大小): {self.page_size} (patches_per_frame: CACHE={self.cache_patches_per_frame}, raw={self.raw_patches_per_frame})", file=sys.stderr)
            print(f"  num_kv_heads (KV头数): {self.num_kv_heads} (GQA {gqa_ratio}:1, query heads={num_heads})", file=sys.stderr)
            print(f"  head_dim (头维度): {head_dim}", file=sys.stderr)
            print(f"  kv dtype (数据类型): {self.kv_dtype} ({bytes_per_element} 字节/元素)", file=sys.stderr)
            print(f"  device (设备): {device}", file=sys.stderr)
            print(f"  每个块的元素数: {elements_per_block:,}", file=sys.stderr)
            print(f"  每个块占用内存: {bytes_per_block / (1024**2):.2f} MB", file=sys.stderr)
            print(f"  所有块总内存: {total_mb:.2f} MB", file=sys.stderr)
            print(f"  每个块的张量形状: [{self.max_num_pages}, 2, {self.page_size}, {self.num_kv_heads}, {head_dim}]", file=sys.stderr)

        # ── Physical paged KV caches ─────────────────────────────────────────
        # Shape per block: [max_num_pages, 2, page_size, H_kv, D]
        self.kv_caches: List[Tensor] = [
            torch.zeros(
                self.max_num_pages, 2, self.page_size, self.num_kv_heads, head_dim,
                dtype=self.kv_dtype, device=device,
            )
            for _ in range(num_blocks)
        ]

        # Per-block scale factors (only used when kv_cache_fp8=True)
        # Initialize with 1.0; will be updated on first frame.
        self.k_scales: List[float] = [1.0] * num_blocks
        self.v_scales: List[float] = [1.0] * num_blocks

        # ── Per-block state ──────────────────────────────────────────────────
        # Patch pages (IDs 0 .. max_patch_pages-1)
        self.scale_patch_pages: List[collections.deque] = [
            collections.deque() for _ in range(num_blocks)
        ]
        self.live_window_patch_pages: List[collections.deque] = [
            collections.deque() for _ in range(num_blocks)
        ]
        self.free_patch_pages: List[List[int]] = [
            list(range(max_patch_pages)) for _ in range(num_blocks)
        ]

        # Special pages (IDs max_patch_pages .. max_num_pages-1)
        self.all_special_pages: List[List[int]] = [[] for _ in range(num_blocks)]
        self.free_special_pages: List[List[int]] = [
            list(range(max_patch_pages, self.max_num_pages)) for _ in range(num_blocks)
        ]
        self.special_token_count: List[int] = [0] * num_blocks

        # Frame counter per block (determines scale vs window routing)
        self.frame_count: List[int] = [0] * num_blocks

        # Deferred eviction support for flow-based keyframe selection.
        # When True, evict_frames() becomes a no-op; caller must later call
        # execute_deferred_eviction() or rollback_last_frame().
        self._defer_eviction: bool = False

        # ── FlashInfer wrapper ───────────────────────────────────────────────
        # plan() is called once per frame step (block_idx == 0).
        # run() is called per layer, reusing the same aux structures.
        # backend: "fa2" (default) or "fa3" (SM90/H100, requires power-of-2 page_size).
        # FA2 supports non-power-of-2 page sizes and avoids a FA3 NaN bug seen in
        # FlashInfer 0.2.5 at 518×378 resolution.
        _fi_backend = "fa3" if fa3 else "fa2"
        print(f"--- backend: {_fi_backend} ---")
        self.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            kv_layout="NHD",
            backend=_fi_backend,
        )

        # plan() inputs (indices/indptr built fresh each step; qo_indptr is fixed)
        self._qo_indptr = torch.tensor(
            [0, tokens_per_frame], dtype=torch.int32, device=device
        )

        print(f"[ FlashInferKVCache 之后 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB ，已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB ]")


    # =========================================================================
    # Public API  (drop-in compatible with previous FlashInferKVCacheManager)
    # =========================================================================

    def append_frame(self, block_idx: int, k: Tensor, v: Tensor) -> None:
        """
        Append one frame's K/V tensors to the two-stream cache.

        Token layout must be: [camera, reg0, ..., regN, scale, patch0, ..., patchP-1]
        i.e. specials come first (matching stream.py's patch_start_idx convention).

        Args:
            block_idx: Block/layer index (0 … num_blocks-1).
            k: [raw_tokens_per_frame, H_q, D]  NHD layout.
            v: [raw_tokens_per_frame, H_q, D]  NHD layout.
        """
        n = self.num_special_tokens  # 6
        sp_k    = k[:n].to(self.q_dtype)
        patch_k = k[n:].to(self.q_dtype)
        sp_v    = v[:n].to(self.q_dtype)
        patch_v = v[n:].to(self.q_dtype)

        assert patch_k.shape[0] == self.raw_patches_per_frame, (
            f"block {block_idx}: expected {self.raw_patches_per_frame} patch tokens, "
            f"got {patch_k.shape[0]} (tokens_per_frame={k.shape[0]})"
        )

        # GQA: aggregate heads from H_q to H_kv by mean pooling over groups (if ratio > 1)
        if self.gqa_ratio > 1:
            def aggregate_heads(t: Tensor) -> Tensor:
                # t: [T, H_q, D]
                T = t.shape[0]
                # reshape to [T, H_kv, group_size, D] and mean over group_size
                group_size = self.num_heads // self.num_kv_heads  
                t_reshaped = t.view(T, self.num_kv_heads, group_size, self.head_dim)
                return t_reshaped.mean(dim=2)  # [T, H_kv, D]
            agg_sp_k = aggregate_heads(sp_k)
            agg_patch_k = aggregate_heads(patch_k)
            agg_sp_v = aggregate_heads(sp_v)
            agg_patch_v = aggregate_heads(patch_v)
        else:
            # No GQA: heads already match
            agg_sp_k = sp_k
            agg_patch_k = patch_k
            agg_sp_v = sp_v
            agg_patch_v = patch_v

        # FP8 quantization & scale management (only if kv_cache_fp8 enabled)
        if self.kv_cache_fp8:
            # Before writing, check if we need to update scales.
            # Compute amax of the new frame's aggregated K and V.
            new_k_amax = max(agg_patch_k.abs().max().item(), agg_sp_k.abs().max().item())
            new_v_amax = max(agg_patch_v.abs().max().item(), agg_sp_v.abs().max().item())

            # Current scale factors map FP8's max representable value (448 for e4m3fn)
            # to the amax. FP8 e4m3fn has max value 448.0.
            # scale = amax / 448.0   (since we will divide by scale before casting)
            old_k_scale = self.k_scales[block_idx]
            old_v_scale = self.v_scales[block_idx]

            # If this is the first frame (scale==1.0) or new amax exceeds current range,
            # we need to update scales and re-quantize everything.
            # We use a safety margin of 1.05 to avoid too-frequent re-quantization.
            need_update_k = (new_k_amax > old_k_scale * 448.0 * 0.95)  # >95% of range
            need_update_v = (new_v_amax > old_v_scale * 448.0 * 0.95)

            if self.frame_count[block_idx] == 0 or need_update_k or need_update_v:
                # Compute new scale = max(current_amax, new_amax) / 448.0
                # We need the max over all existing cache + new frame.
                # For performance, we recompute global amax by scanning all pages.
                # This is O(num_pages) and called infrequently (when range expands).
                new_k_scale, new_v_scale = self._compute_global_amax_and_scale(block_idx)
                # Incorporate the new frame's amax as well.
                new_k_scale = max(new_k_scale, new_k_amax / 448.0)
                new_v_scale = max(new_v_scale, new_v_amax / 448.0)
                # Add a tiny margin to avoid immediate re-quantization next step.
                new_k_scale *= 1.05
                new_v_scale *= 1.05

                # Re-quantize the entire cache with the new scales.
                if self.frame_count[block_idx] > 0:
                    self._requantize_all_pages(block_idx, new_k_scale, new_v_scale)
                self.k_scales[block_idx] = new_k_scale
                self.v_scales[block_idx] = new_v_scale

        if self.kv_cache_cut > 1:
            # downsampling agg_patch_k, agg_patch_v
            if 1:
                def downsample_2D(t: Tensor, step: int) -> Tensor:
                    """
                    将一维patch token序列转回二维按指定步长下采样
                    t: [T=H*W, H_q, D] -> [T'=H'*W', H_q, D]
                    step: 采样跨度，2=每隔1个取，3=每隔2个取...
                    """
                    T, H_q, D = t.shape
                    # 从777=37x21恢复到2D空间布局
                    H, W = 37, 21
                    assert T == H * W, f"Expected {H}x{W}={H*W} tokens, got {T}"
                    
                    # [H*W, H_q, D] -> [H, W, H_q, D]
                    t_2d = t.view(H, W, H_q, D)
                    
                    # 按步长下采样: ::step
                    t_downsampled = t_2d[::step, ::step, :, :]
                    
                    # [H_new, W_new, H_q, D] -> [T_new, H_q, D]
                    H_new, W_new = t_downsampled.shape[:2]
                    return t_downsampled.contiguous().view(H_new * W_new, H_q, D)            
                        # 对K和V分别执行下采样
                agg_patch_k = downsample_2D(agg_patch_k, self.kv_cache_cut)
                agg_patch_v = downsample_2D(agg_patch_v, self.kv_cache_cut)

            if 0:
                cut = self.kv_cache_cut
                
                # 只生成一次随机偏移，K和V共用
                H, W = 37, 21
                H_out = math.ceil(H / cut)
                W_out = math.ceil(W / cut)
                
                # 生成一次随机坐标
                offset_h = torch.randint(0, cut, (H_out, W_out), device=agg_patch_k.device)
                offset_w = torch.randint(0, cut, (H_out, W_out), device=agg_patch_k.device)
                base_h = torch.arange(0, H_out, device=agg_patch_k.device) * cut
                base_w = torch.arange(0, W_out, device=agg_patch_k.device) * cut
                idx_h = (base_h.unsqueeze(1) + offset_h).clamp(max=H-1)
                idx_w = (base_w.unsqueeze(0) + offset_w).clamp(max=W-1)
                
                def sample_with_mask(t: Tensor) -> Tensor:
                    t_2d = t.view(H, W, t.shape[1], t.shape[2])
                    return t_2d[idx_h, idx_w, :, :].contiguous().view(H_out * W_out, t.shape[1], t.shape[2])
                
                agg_patch_k = sample_with_mask(agg_patch_k)
                agg_patch_v = sample_with_mask(agg_patch_v)

        # Write patch and special tokens using current (maybe updated) scales.
        self._write_patch_page(block_idx, agg_patch_k, agg_patch_v)
        self._write_special_tokens(block_idx, agg_sp_k, agg_sp_v)
        self.frame_count[block_idx] += 1

    def evict_frames(
        self,
        block_idx: int,
        scale_frames: int,
        sliding_window: int,
        cross_frame_special: bool = True,
        include_scale_frames: bool = True,
        camera_only: bool = False,
        num_register_tokens: int = 4,
    ) -> None:
        """
        Evict old window patch pages (recycle to free list).

        Special pages are NEVER evicted.
        Scale pages are NEVER evicted.
        Only live_window_patch_pages beyond `sliding_window` are recycled.

        When ``_defer_eviction`` is True, this method is a no-op.  The caller
        is expected to later call ``execute_deferred_eviction()`` (keep frame)
        or ``rollback_last_frame()`` (discard frame).
        """
        if self._defer_eviction:
            return
        while len(self.live_window_patch_pages[block_idx]) > sliding_window:
            old_page = self.live_window_patch_pages[block_idx].popleft()
            self.free_patch_pages[block_idx].append(old_page)

    def execute_deferred_eviction(
        self,
        block_idx: int,
        scale_frames: int,
        sliding_window: int,
        **kwargs,
    ) -> None:
        """Run the eviction that was skipped while ``_defer_eviction`` was True."""
        while len(self.live_window_patch_pages[block_idx]) > sliding_window:
            old_page = self.live_window_patch_pages[block_idx].popleft()
            self.free_patch_pages[block_idx].append(old_page)

    def rollback_last_frame(self, block_idx: int) -> None:
        """Undo the most recent ``append_frame()`` for *block_idx*.

        This reverses all three sub-operations of ``append_frame``:
        patch page allocation, special-token write, and frame_count increment.
        It must be called **before** any eviction for that frame (i.e. while
        ``_defer_eviction`` is True or before ``evict_frames`` is called).
        """
        assert self.frame_count[block_idx] > 0, (
            f"block {block_idx}: cannot rollback, frame_count is 0"
        )

        # 1) Undo patch page ── pop from whichever deque it was routed to.
        if self.frame_count[block_idx] > self.scale_frames:
            page_id = self.live_window_patch_pages[block_idx].pop()
        else:
            page_id = self.scale_patch_pages[block_idx].pop()
        self.free_patch_pages[block_idx].append(page_id)

        # 2) Undo special tokens
        n = self.num_special_tokens
        new_count = self.special_token_count[block_idx] - n
        assert new_count >= 0, (
            f"block {block_idx}: special_token_count underflow "
            f"({self.special_token_count[block_idx]} - {n})"
        )
        new_num_pages = math.ceil(new_count / self.page_size) if new_count > 0 else 0
        while len(self.all_special_pages[block_idx]) > new_num_pages:
            freed = self.all_special_pages[block_idx].pop()
            self.free_special_pages[block_idx].append(freed)
        self.special_token_count[block_idx] = new_count

        # 3) Decrement frame count
        self.frame_count[block_idx] -= 1

    def get_cache_stats(self, block_idx: int = 0) -> dict:
        """Read-only snapshot of cache occupancy for one block.

        Useful for debugging keyframe / sliding-window behavior.

        Returns:
            dict with keys:
              - ``frame_count``   total frames ever appended (minus rollbacks)
              - ``scale_pages``   scale-region patch pages currently held
              - ``live_pages``    sliding-window patch pages currently held
              - ``free_pages``    patch pages on the free list
              - ``special_tokens`` running count of special tokens written
        """
        return {
            "frame_count":    int(self.frame_count[block_idx]),
            "scale_pages":    len(self.scale_patch_pages[block_idx]),
            "live_pages":     len(self.live_window_patch_pages[block_idx]),
            "free_pages":     len(self.free_patch_pages[block_idx]),
            "special_tokens": int(self.special_token_count[block_idx]),
            "k_scale":        self.k_scales[block_idx],
            "v_scale":        self.v_scales[block_idx],
        }
    def compute_attention(self, block_idx: int, q: Tensor) -> Tensor:
        """
        Compute cross-frame attention using FlashInfer BatchPrefillWithPagedKVCacheWrapper.

        plan() is called once per frame step (when block_idx == 0).
        All layers at the same step share the same visible page structure,
        so the plan is reused by calling run() with each layer's kv_cache.

        Args:
            block_idx: Block/layer index.
            q: [q_len, H_q, D]  NHD layout (q_len = tokens_per_frame = 262).

        Returns:
            out: [q_len, H_q, D]
        """
        if self.frame_count[block_idx] == 0:
            # No KV present yet (should not occur in normal usage after append_frame)
            return torch.zeros_like(q)


        if block_idx == 0:
            # ── Plan once per frame step ──────────────────────────────────────
            # Build visible page table from block 0's state.
            # All blocks have identical page structures, so this plan is valid
            # for all subsequent run() calls (block_idx = 1, 2, ...).
            visible  = self.build_visible_page_table(0)
            last_len = self.compute_last_page_len(0)

            assert visible, "visible page table is empty after append_frame"
            assert 1 <= last_len <= self.page_size, (
                f"block 0: last_page_len={last_len} out of [1, {self.page_size}]"
            )

            paged_kv_indices       = torch.tensor(visible, dtype=torch.int32, device=self.device)
            paged_kv_indptr        = torch.tensor([0, len(visible)], dtype=torch.int32, device=self.device)
            paged_kv_last_page_len = torch.tensor([last_len], dtype=torch.int32, device=self.device)

            self.prefill_wrapper.plan(
                self._qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads      = self.num_heads,
                num_kv_heads      = self.num_kv_heads,
                head_dim_qk       = self.head_dim,
                page_size         = self.page_size,
                causal            = False,          # custom page ordering; no causal mask
                pos_encoding_mode = "NONE",         # RoPE applied externally before append
                q_data_type       = self.q_dtype,
                kv_data_type      = self.kv_dtype,   # actual KV cache dtype (FP8 or bfloat16)
            )

        # ── Run attention for this layer ──────────────────────────────────────
        # Cast q to storage dtype (LayerNorm may upcast to float32 under autocast).
        return self.prefill_wrapper.run(
            q              = q.to(self.q_dtype).contiguous(),
            paged_kv_cache = self.kv_caches[block_idx],
            k_scale        = self.k_scales[block_idx],
            v_scale        = self.v_scales[block_idx],
        )

    def reset(self) -> None:
        for i in range(self.num_blocks):
            self.scale_patch_pages[i].clear()
            self.live_window_patch_pages[i].clear()
            self.all_special_pages[i].clear()
            self.free_patch_pages[i]   = list(range(self.max_patch_pages))
            self.free_special_pages[i] = list(range(self.max_patch_pages, self.max_num_pages))
            self.special_token_count[i] = 0
            self.frame_count[i] = 0
            self.k_scales[i] = 1.0
            self.v_scales[i] = 1.0
            # Reset KV cache to zeros (FP8 zero is well-defined)
            self.kv_caches[i].zero_()

    # =========================================================================
    # Helper methods
    # =========================================================================

    def build_visible_page_table(self, block_idx: int) -> List[int]:
        """
        Return page IDs in strict order: scale → window → special.

        Placing special pages last means only the final page may be partially
        full, so paged_kv_last_page_len = compute_last_page_len() is sufficient
        without a custom attention mask.
        """
        return (
            list(self.scale_patch_pages[block_idx])       +
            list(self.live_window_patch_pages[block_idx]) +
            list(self.all_special_pages[block_idx])
        )

    def compute_last_page_len(self, block_idx: int) -> int:
        """
        Valid token count in the last page of the visible sequence.

        - No special pages      → last page is a patch page.
                                  Returns patches_per_frame (real tokens written),
                                  which may be < page_size when page_size was rounded
                                  up to a power of 2.
        - Special tail partial  → special_token_count % page_size.
        - Special tail exactly full → page_size.
        """
        if not self.all_special_pages[block_idx]:
            # Last page is a patch page.  We wrote patches_per_frame tokens (0..P-1);
            # positions P..page_size-1 are zero padding.  Tell FlashInfer the true
            # valid count so it doesn't read beyond the real tokens.
            return self.cache_patches_per_frame

        tail = self.special_token_count[block_idx] % self.page_size
        return self.page_size if tail == 0 else tail

    def _compute_global_amax_and_scale(self, block_idx: int):
        """
        Scan all pages in the block to find the maximum absolute value in K and V.
        Returns (k_scale, v_scale) where scale = amax / 448.0 (max FP8 value).
        If no pages exist, returns (1.0, 1.0).
        Only meaningful when kv_cache_fp8=True.
        """
        if not self.kv_cache_fp8:
            return 1.0, 1.0
        max_k = 0.0
        max_v = 0.0
        kv = self.kv_caches[block_idx]
        # We need to dequantize current FP8 values using current scales to get real values.
        current_k_scale = self.k_scales[block_idx]
        current_v_scale = self.v_scales[block_idx]

        # Iterate over all allocated pages (scale+window+special)
        all_pages = (list(self.scale_patch_pages[block_idx]) +
                     list(self.live_window_patch_pages[block_idx]) +
                     self.all_special_pages[block_idx])
        for pid in all_pages:
            # For K: dequantize to float32 and find max
            k_page = kv[pid, 0].to(torch.float32) * current_k_scale
            v_page = kv[pid, 1].to(torch.float32) * current_v_scale
            max_k = max(max_k, k_page.abs().max().item())
            max_v = max(max_v, v_page.abs().max().item())

        if max_k == 0.0:
            k_scale = 1.0
        else:
            k_scale = max_k / 448.0
        if max_v == 0.0:
            v_scale = 1.0
        else:
            v_scale = max_v / 448.0
        return k_scale, v_scale

    def _requantize_all_pages(self, block_idx: int, new_k_scale: float, new_v_scale: float):
        """
        Re-quantize all existing pages in the block from the old scale to the new scale.
        This is done by dequantizing to float32 using the old scale, then quantizing
        using the new scale and storing back as FP8.
        Only effective when kv_cache_fp8=True.
        """
        if not self.kv_cache_fp8:
            return
        old_k_scale = self.k_scales[block_idx]
        old_v_scale = self.v_scales[block_idx]
        if old_k_scale == new_k_scale and old_v_scale == new_v_scale:
            return

        kv = self.kv_caches[block_idx]
        all_pages = (list(self.scale_patch_pages[block_idx]) +
                     list(self.live_window_patch_pages[block_idx]) +
                     self.all_special_pages[block_idx])

        for pid in all_pages:
            # Dequantize K and V to float32
            k_f32 = kv[pid, 0].to(torch.float32) * old_k_scale
            v_f32 = kv[pid, 1].to(torch.float32) * old_v_scale
            # Quantize with new scale
            k_q = (k_f32 / new_k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            v_q = (v_f32 / new_v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            kv[pid, 0] = k_q
            kv[pid, 1] = v_q

    # ── Internal write helpers ────────────────────────────────────────────────

    def _write_patch_page(self, block_idx: int, patch_k: Tensor, patch_v: Tensor) -> int:
        """
        Allocate one free patch page and write patches_per_frame patch tokens.

        Direct tensor assignment to kv_caches[block_idx][page_id, 0/1] avoids
        the Python→C++/CUDA dispatch overhead of flashinfer.page.append_paged_kv_cache.
        kv_caches layout: [max_num_pages, 2, page_size, H_kv, D]  (NHD, K=0, V=1).
        patch_k/v fill exactly one full page (patches_per_frame == page_size).

        Routes to scale_patch_pages if still filling scale quota,
        otherwise to live_window_patch_pages.

        Args:
            patch_k: [patches_per_frame, H_kv, D] (aggregated)
            patch_v: [patches_per_frame, H_kv, D]
        Returns:
            page_id: Physical page index used.
        """
        assert self.free_patch_pages[block_idx], (
            f"block {block_idx}: patch page pool exhausted — "
            f"scale={len(self.scale_patch_pages[block_idx])}, "
            f"window={len(self.live_window_patch_pages[block_idx])}, "
            f"free={len(self.free_patch_pages[block_idx])}"
        )

        page_id = self.free_patch_pages[block_idx].pop()

        # Direct slice write: positions 0..patches_per_frame-1.
        # When page_size == patches_per_frame (power-of-2 aligned, e.g. 256 for 224×224),
        # this is equivalent to a full-page write.  When page_size > patches_per_frame
        # (rounded up for FA3 alignment, e.g. page_size=1024 for patches_per_frame=999),
        # positions patches_per_frame..page_size-1 remain zero (kv_caches is zero-init).
        P = self.cache_patches_per_frame

        if self.kv_cache_fp8:
            # Quantize to FP8 using current scales
            k_quant = (patch_k / self.k_scales[block_idx]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            v_quant = (patch_v / self.v_scales[block_idx]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            self.kv_caches[block_idx][page_id, 0, :P] = k_quant
            self.kv_caches[block_idx][page_id, 1, :P] = v_quant
        else:
            # Directly store bfloat16 values (no scaling)
            self.kv_caches[block_idx][page_id, 0, :P] = patch_k.to(self.kv_dtype)
            self.kv_caches[block_idx][page_id, 1, :P] = patch_v.to(self.kv_dtype)

        if len(self.scale_patch_pages[block_idx]) < self.scale_frames:
            self.scale_patch_pages[block_idx].append(page_id)
        else:
            self.live_window_patch_pages[block_idx].append(page_id)

        return page_id

    def _write_special_tokens(self, block_idx: int, sp_k: Tensor, sp_v: Tensor) -> None:
        """
        Append num_special_tokens (6) special tokens to the special stream.

        Direct tensor slice assignment to kv_caches[block_idx][tail_page, 0/1,
        tail_offset : tail_offset+write_n] avoids the Python→C++/CUDA dispatch
        overhead of flashinfer.page.append_paged_kv_cache.

        Handles page-boundary crossing: if 6 tokens straddle two pages, performs
        two slice writes (rare — page_size=256 >> 6).

        Args:
            sp_k: [num_special_tokens, H_kv, D] (aggregated)
            sp_v: [num_special_tokens, H_kv, D]
        """
        remaining = self.num_special_tokens   # 6
        written   = 0

        while remaining > 0:
            tail_offset = self.special_token_count[block_idx] % self.page_size

            if tail_offset == 0:
                # Current tail page is full (or no page exists) — allocate a new one
                assert self.free_special_pages[block_idx], (
                    f"block {block_idx}: special page pool exhausted at "
                    f"special_token_count={self.special_token_count[block_idx]}. "
                    f"Increase max_total_frames."
                )
                new_page = self.free_special_pages[block_idx].pop()
                self.all_special_pages[block_idx].append(new_page)

            tail_page = self.all_special_pages[block_idx][-1]
            space     = self.page_size - tail_offset   # free slots in tail page
            write_n   = min(remaining, space)

            # Direct slice write: kv_caches[block_idx][tail_page, 0/1, offset:offset+n]
            # shape: [page_size, H_kv, D];  slice [tail_offset:tail_offset+write_n, :, :]
            end = tail_offset + write_n
            if self.kv_cache_fp8:
                k_quant = (sp_k[written:written+write_n] / self.k_scales[block_idx]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                v_quant = (sp_v[written:written+write_n] / self.v_scales[block_idx]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                self.kv_caches[block_idx][tail_page, 0, tail_offset:end] = k_quant
                self.kv_caches[block_idx][tail_page, 1, tail_offset:end] = v_quant
            else:
                self.kv_caches[block_idx][tail_page, 0, tail_offset:end] = sp_k[written:written+write_n].to(self.kv_dtype)
                self.kv_caches[block_idx][tail_page, 1, tail_offset:end] = sp_v[written:written+write_n].to(self.kv_dtype)

            self.special_token_count[block_idx] += write_n
            written   += write_n
            remaining -= write_n

    # ── Legacy property (used by stream.py) ──────────────────────────────────

    @property
    def num_frames(self) -> int:
        """Number of frames appended to block 0 (representative)."""
        return self.frame_count[0] if self.frame_count else 0


# =============================================================================
# Sanity check
# =============================================================================

def _sanity_check():
    """
    Minimal smoke test.
    Run with:  python -c "from lingbot_map.layers.flashinfer_cache import _sanity_check; _sanity_check()"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[sanity_check] CUDA not available — skipping.")
        return

    tokens_per_frame  = 262   # 256 patch + 6 special (224×224)
    num_special       = 6
    patches_per_frame = tokens_per_frame - num_special  # 256
    page_size         = patches_per_frame               # 256

    mgr = FlashInferKVCacheManager(
        num_blocks         = 2,
        max_num_frames     = 88,
        tokens_per_frame   = tokens_per_frame,
        num_heads          = 16,          # query heads, must be even
        head_dim           = 64,
        dtype              = torch.bfloat16,
        device             = device,
        num_special_tokens = num_special,
        scale_frames       = 8,
        sliding_window     = 64,
        max_total_frames   = 200,
        kv_cache_fp8       = True,        # test FP8 path
        gqa_ratio          = 2,           # test GQA path
    )

    def make_kv():
        k = torch.randn(tokens_per_frame, 16, 64, dtype=torch.bfloat16, device=device)
        v = torch.randn(tokens_per_frame, 16, 64, dtype=torch.bfloat16, device=device)
        return k, v

    def make_q():
        return torch.randn(tokens_per_frame, 16, 64, dtype=torch.bfloat16, device=device)

    for block in range(2):
        for t in range(100):
            k, v = make_kv()
            mgr.append_frame(block, k, v)
            mgr.evict_frames(block, scale_frames=8, sliding_window=64)

        # ── Page count checks ───────────────────────────────────────────────
        n_scale  = len(mgr.scale_patch_pages[block])
        n_window = len(mgr.live_window_patch_pages[block])
        n_spec   = len(mgr.all_special_pages[block])
        sp_count = mgr.special_token_count[block]

        assert n_scale  == 8,  f"block {block}: scale pages = {n_scale},  expected 8"
        assert n_window == 64, f"block {block}: window pages = {n_window}, expected 64"
        # 100 frames × 6 specials = 600 tokens; ceil(600/256) = 3 pages
        expected_spec_pages = math.ceil(100 * num_special / page_size)
        assert n_spec == expected_spec_pages, (
            f"block {block}: special pages = {n_spec}, expected {expected_spec_pages}"
        )
        assert sp_count == 100 * num_special, (
            f"block {block}: special_token_count = {sp_count}, expected {100*num_special}"
        )

        # ── last_page_len ────────────────────────────────────────────────────
        last_len = mgr.compute_last_page_len(block)
        tail = sp_count % page_size
        expected_len = page_size if tail == 0 else tail
        assert last_len == expected_len, f"block {block}: last_len={last_len}, expected={expected_len}"

        # ── visible page table order ─────────────────────────────────────────
        visible = mgr.build_visible_page_table(block)
        assert len(visible) == n_scale + n_window + n_spec, "visible page count mismatch"
        for pid in visible[:n_scale + n_window]:
            assert pid < mgr.max_patch_pages, f"patch page {pid} out of patch range"
        for pid in visible[n_scale + n_window:]:
            assert pid >= mgr.max_patch_pages, f"special page {pid} not in special range"

        # ── forward pass: plan() once for block 0, run() for both blocks ─────
        if block == 1:
            # Simulate the actual calling pattern: plan on block 0, run on both
            q0 = make_q()
            out0 = mgr.compute_attention(0, q0)   # triggers plan()
            q1 = make_q()
            out1 = mgr.compute_attention(1, q1)   # reuses plan, different kv_cache
            assert out0.shape == (tokens_per_frame, 16, 64)
            assert out1.shape == (tokens_per_frame, 16, 64)

        print(f"[block {block}] PASS: scale={n_scale}, window={n_window}, "
              f"special_pages={n_spec}, special_tokens={sp_count}, "
              f"last_page_len={last_len}, k_scale={mgr.k_scales[block]:.4f}")

    mgr.reset()
    assert mgr.frame_count[0] == 0
    print("\n[sanity_check] All assertions passed.")


if __name__ == "__main__":
    _sanity_check()