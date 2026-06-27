# visualize_attention.py
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import math


def parse_heads(head_str):
    if head_str is None:
        return None
    heads = []
    parts = head_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            heads.extend(range(int(start), int(end) + 1))
        else:
            heads.append(int(part))
    return sorted(set(heads))


def visualize(pt_file, head_spec=None, query=0, mode='curve', html=False, dpi=150,
              kv_sp=0, q_sp=0, kv_patch=0, q_patch=0, cols=1):
    data = torch.load(pt_file, map_location='cpu')
    weights = data['weights']                     # [H, Q_orig, K_orig]
    H, Q_orig, K_orig = weights.shape
    print(f"Loaded: weights shape {list(weights.shape)}")

    # ---------- 裁切 query ----------
    if q_sp > 0:
        if q_sp >= Q_orig:
            raise ValueError(f"q_sp ({q_sp}) must be < Q ({Q_orig})")
        weights = weights[:, q_sp:, :]
        Q_rem = weights.shape[1]
        print(f"After removing first {q_sp} query tokens: Q = {Q_rem}")
    else:
        Q_rem = Q_orig

    if q_patch > 0:
        if q_patch >= Q_rem:
            raise ValueError(f"q_patch ({q_patch}) must be < Q ({Q_rem})")
        weights = weights[:, :Q_rem - q_patch, :]
        Q_new = weights.shape[1]
        print(f"After removing last {q_patch} query tokens: Q = {Q_new}")
    else:
        Q_new = Q_rem

    # ---------- 裁切 key ----------
    if kv_patch > 0:
        if kv_patch >= K_orig:
            raise ValueError(f"kv_patch ({kv_patch}) must be < K ({K_orig})")
        weights = weights[:, :, kv_patch:]
        K_rem = weights.shape[2]
        print(f"After removing first {kv_patch} key tokens: K = {K_rem}")
    else:
        K_rem = K_orig

    if kv_sp > 0:
        if kv_sp >= K_rem:
            raise ValueError(f"kv_sp ({kv_sp}) must be < K ({K_rem})")
        weights = weights[:, :, :-kv_sp]
        K_new = weights.shape[2]
        print(f"After removing last {kv_sp} key tokens: K = {K_new}")
    else:
        K_new = K_rem

    # 构造后缀提示
    suffix_parts = []
    if q_sp > 0:
        suffix_parts.append(f"first {q_sp} query tokens removed")
    if q_patch > 0:
        suffix_parts.append(f"last {q_patch} query tokens removed")
    if kv_patch > 0:
        suffix_parts.append(f"first {kv_patch} key tokens removed")
    if kv_sp > 0:
        suffix_parts.append(f"last {kv_sp} key tokens removed")
    suffix = " (" + "; ".join(suffix_parts) + ")" if suffix_parts else ""

    # --- headmap 模式 ---
    if mode == 'headmap':
        if query < 0 or query >= Q_new:
            raise ValueError(f"query {query} out of range [0, {Q_new-1}]")
        headmap_data = weights[:, query, :].cpu().numpy()
        plt.figure(figsize=(12, max(4, H // 4)))
        plt.imshow(headmap_data, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Attention weight')
        plt.xlabel("Key token index")
        plt.ylabel("Head index")
        plt.title(f"Attention across all heads for query token {query}{suffix}")
        plt.tight_layout()
        plt.show()
        return

    # --- 解析头列表 ---
    head_list = parse_heads(head_spec)

    # --- heatmap 模式 ---
    if mode == 'heatmap':
        data_aspect = Q_new / K_new

        # 没有指定头 → 平均热力图
        if head_list is None:
            attn = weights.mean(dim=0).cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 10 * data_aspect))
            im = ax.imshow(attn, aspect='equal', cmap='viridis', origin='upper')
            ax.set_box_aspect(data_aspect)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention weight')
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.set_xlabel("Key token index")
            ax.set_ylabel("Query token index")
            ax.set_title(f"Attention heatmap (average over {H} heads){suffix}", pad=12)
            plt.tight_layout()
            plt.show()
            return

        n_heads = len(head_list)

        # --- HTML 堆叠模式（保持正方形 token，精确高度，统一色条）---
        if html:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError:
                print("Plotly is required for HTML output. Falling back to matplotlib plot.")
                html = False
            else:
                # ========== 布局参数 ==========
                plot_width = 900          # 整张图总宽度
                colorbar_thickness = 16   # 色条厚度
                colorbar_pad = 10         # 色条与绘图区间距
                colorbar_total_width = colorbar_thickness + colorbar_pad
                
                margin_left = 70
                margin_right = 20
                margin_top = 80
                margin_bottom = 50
                
                # 实际可用绘图宽度（扣除边距 + 右侧色条宽度）
                draw_width = plot_width - margin_left - margin_right - colorbar_total_width
                # 单个子图的目标绘图高度（保持 token 正方形）
                row_plot_height = draw_width * data_aspect
                
                vertical_spacing = 0.025  # 子图间垂直间距（占总绘图区高度比例）
                # 推导总绘图区高度：总高 = 子图数 * 单子图高 / (1 - 间距比例*(子图数-1))
                if n_heads == 1:
                    total_plot_height = row_plot_height
                else:
                    total_plot_height = (n_heads * row_plot_height) / (1 - vertical_spacing * (n_heads - 1))
                
                total_height = math.ceil(total_plot_height) + margin_top + margin_bottom

                fig = make_subplots(
                    rows=n_heads, cols=1,
                    shared_xaxes=True,
                    shared_yaxes=False,
                    vertical_spacing=vertical_spacing
                )

                # ========== 逐个添加子图 ==========
                for i, head_idx in enumerate(head_list):
                    attn = weights[head_idx].cpu().numpy()
                    is_first = (i == 0)
                    is_last = (i == n_heads - 1)
                    
                    # 修正：Plotly 第一个轴叫 x/y，第二个起才是 x2/y2
                    subplot_idx = i + 1
                    x_axis = "x" if subplot_idx == 1 else f"x{subplot_idx}"
                    y_axis = "y" if subplot_idx == 1 else f"y{subplot_idx}"
                    
                    # 仅最后一个子图显示色条，其余隐藏
                    show_scale = is_last
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=attn,
                            x=list(range(K_new)),
                            y=list(range(Q_new)),
                            colorscale='Viridis',
                            name=f'Head {head_idx}',
                            showscale=show_scale,
                            colorbar=dict(
                                thickness=colorbar_thickness,
                                len=0.6,          # 色条占总高度比例
                                yanchor='middle',
                                y=0.5,
                                xpad=colorbar_pad,
                                title=dict(text='Weight', side='right')
                            )
                        ),
                        row=subplot_idx, col=1
                    )
                    
                    # y轴：0在顶部，绑定x轴保持等比例（正方形token）
                    fig.update_yaxes(
                        autorange='reversed',
                        scaleanchor=x_axis,
                        scaleratio=1,
                        title_text="Query index" if is_first else "",
                        row=subplot_idx, col=1
                    )
                    
                    # x轴：仅最底部子图显示标签
                    fig.update_xaxes(
                        title_text="Key token index" if is_last else "",
                        showticklabels=is_last,
                        row=subplot_idx, col=1
                    )
                    
                    # 子图标题（使用子图自身 domain 坐标系）
                    fig.add_annotation(
                        x=0.5, y=1.0,
                        xref=f"{x_axis} domain",
                        yref=f"{y_axis} domain",
                        text=f"Head {head_idx}",
                        showarrow=False,
                        font=dict(size=11),
                        yshift=8,
                        row=subplot_idx, col=1
                    )

                # ========== 全局布局 ==========
                fig.update_layout(
                    title=dict(
                        text=f"Attention heatmaps{suffix}",
                        x=0.5,
                        y=0.98
                    ),
                    width=plot_width,
                    height=total_height,
                    showlegend=False,
                    margin=dict(
                        l=margin_left,
                        r=margin_right,
                        t=margin_top,
                        b=margin_bottom
                    ),
                    plot_bgcolor='white'
                )

                # 自动生成 HTML 文件名
                if pt_file.endswith('.pt'):
                    html_file = pt_file[:-3] + '_heatmap.html'
                else:
                    html_file = pt_file + '_heatmap.html'
                fig.write_html(html_file, include_plotlyjs='cdn')
                print(f"Scrollable HTML heatmap saved to {html_file}")
                return


        # --- 静态 matplotlib 多子图（仅显示，保持正方形 token）---
        if cols <= 0:
            if data_aspect < 0.3:
                max_cols = 2
            elif data_aspect < 0.7:
                max_cols = 3
            else:
                max_cols = 4
            cols_used = min(max_cols, n_heads)
        else:
            cols_used = min(cols, n_heads)
        total_rows = math.ceil(n_heads / cols_used)

        subplot_height_inch = 4.0 * data_aspect
        fig_height = total_rows * subplot_height_inch + 1.2 * (total_rows - 1) + 0.8
        fig_width = cols_used * 4.5 + 1.0
        fig, axes = plt.subplots(total_rows, cols_used, figsize=(fig_width, fig_height))
        if total_rows * cols_used == 1:
            axes = np.array([axes])
        elif total_rows == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, head_idx in enumerate(head_list):
            attn = weights[head_idx].cpu().numpy()
            ax = axes[i]
            im = ax.imshow(attn, aspect='equal', cmap='viridis', origin='upper')
            ax.set_box_aspect(data_aspect)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.set_xlabel("Key token index")
            ax.set_ylabel("Query token index")
            ax.set_title(f"Head {head_idx}", fontsize=10, pad=3)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Attention heatmaps{suffix}", fontsize=14, y=0.99)
        plt.subplots_adjust(hspace=0.6, wspace=0.35, top=0.92)
        plt.show()
        return

    # --- curve 模式 ---
    if mode == 'curve':
        if head_list is not None:
            head_idx = head_list[0]
            if head_idx < 0 or head_idx >= H:
                raise ValueError(f"head {head_idx} out of range [0, {H-1}]")
            data_title = f"Head {head_idx}"
            attn = weights[head_idx]
        else:
            data_title = f"Average over {H} heads"
            attn = weights.mean(dim=0)
        attn_np = attn.cpu().numpy()
        if query < 0 or query >= Q_new:
            raise ValueError(f"query {query} out of range [0, {Q_new-1}]")
        vec = attn_np[query]
        plt.figure(figsize=(14, 4))
        plt.plot(vec, linewidth=0.6, color='navy')
        plt.xlabel("Key token index")
        plt.ylabel("Attention weight")
        plt.title(f"{data_title}, query token {query}{suffix}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention weights.")
    parser.add_argument("pt_file", help="Path to the .pt file")
    parser.add_argument("--mode", choices=['curve', 'heatmap', 'headmap'], default='curve')
    parser.add_argument("--head", type=str, default=None)
    parser.add_argument("--query", type=int, default=0)
    parser.add_argument("--html", action='store_true',
                        help="Generate scrollable HTML heatmap (heatmap mode only)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for any matplotlib figures (not used in HTML)")
    parser.add_argument("--kv-sp", type=int, default=0, help="Remove last N key tokens")
    parser.add_argument("--q-sp", type=int, default=0, help="Remove first N query tokens")
    parser.add_argument("--kv-patch", type=int, default=0, help="Remove first N key tokens")
    parser.add_argument("--q-patch", type=int, default=0, help="Remove last N query tokens")
    parser.add_argument("--cols", type=int, default=1,
                        help="Number of subplot columns in heatmap mode (matplotlib only). Default=1.")
    args = parser.parse_args()

    visualize(args.pt_file, args.head, args.query, args.mode, args.html, args.dpi,
              args.kv_sp, args.q_sp, args.kv_patch, args.q_patch, args.cols)