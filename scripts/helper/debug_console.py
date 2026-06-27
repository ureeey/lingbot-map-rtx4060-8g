def compute_attn_scores(q, kv_cache, visible_pages, last_page_len):
    """
    返回 softmax 前的注意力分数（推荐，能更清晰看到原始相关性）。
    如果需要 softmax 后的权重，自己在结果上加 .softmax(dim=-1) 即可。

    Args:
        q:               [q_len, num_heads, head_dim]
        kv_cache:        [num_pages, 2, page_size, num_kv_heads, head_dim]
        visible_pages:   list[int]  可见页索引
        last_page_len:   int        最后一页有效 token 数

    Returns:
        scores:  [num_heads, q_len, kv_len]  (即每头一个注意力分数矩阵)
    """
    # 取出所有可见页的 key
    k_pages = kv_cache[visible_pages, 0]          # [V, page_size, H_kv, D]
    V, page_size, H_kv, D = k_pages.shape

    # 计算总有效 token 数
    total_kv = (V - 1) * page_size + last_page_len

    # 拼接成 [total_kv, H_kv, D]
    key = torch.empty(total_kv, H_kv, D, dtype=q.dtype, device=q.device)
    idx = 0
    for i in range(V):
        length = page_size if i < V - 1 else last_page_len
        key[idx:idx+length] = k_pages[i, :length]   # [length, H_kv, D]
        idx += length

    # 点积 + 缩放
    scale = D ** 0.5
    # q: [qL, H, D]   key: [kvL, H, D]
    scores = torch.einsum('qhd,khd->hqk', q, key) / scale
    weights = scores.softmax(dim=-1)
    return scores, weights

# 示例：
# 获得页表 & 最后一页长度（复用模型内部方法）
visible = self.build_visible_page_table(0)
last_len = self.compute_last_page_len(0)

# 计算分数
scores, weights = compute_attn_scores(
    q,
    self.kv_caches[block_idx],
    visible,
    last_len
)
# scores.shape  ->  [16, 783, total_kv_len]

# 如果想看 softmax 后的权重：
# weights = scores.softmax(dim=-1)

# 查看第 0 头、第 0 个 query 对各个 key 的分数：
# scores[0, 0, :]

torch.save({'scores': scores.cpu(), 'weights': weights.cpu()}, \
           f'/home/jcy/workspace/lingbot-map/output/attn_analysis/{block_idx}th_frame_global_block_idx_{block_idx}.pt')


        {
            "name": "Python: predict_stream_kv_fp8",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/predict_stream.py",
            "args": [
                "--model_path", "../models/lingbot-map-long.pt",
                "--image_folder", "example/indoor_travel/indoor_travel_frames/",
                "--output_dir", "./output/",
                "--first_k", "107",
                "--num_scale_frames", "5",
                "--kv_cache_sliding_window", "48",
                "--kv_cache_fp8"
            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "${workspaceFolder}"
        }