"""LingBot-MAP Predictions Data Analysis Script:
Analyze statistical features and data distribution of predictions.pt saved by predict.py
Save analysis results to JSON and generate distribution visualization plots.

Core features:
- Depth & depth confidence global statistics + distribution plots
- Extrinsic: **Adjacent frame relative translation distance** (sum/mean/median/std/min/max)
- Extrinsic: **Adjacent frame relative rotation angle** (sum/mean/median/std/min/max, rad + deg)
- Intrinsic: Per-element independent statistics for all 9 coefficients

Usage:
    python scripts/data_analysis.py --pred_path ./output/predictions.pt \
        --max_frames 1000 --output_dir ./output/analysis_results
"""
import argparse
import os
import json
import time
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# 设置matplotlib字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# 自定义JSON编码器（解决numpy类型序列化问题）
# =============================================================================
class NumpyJSONEncoder(json.JSONEncoder):
    """支持numpy数值类型和数组的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)

# =============================================================================
# 相机参数专用统计函数（相邻帧相对运动）
# =============================================================================
def analyze_extrinsic_relative(extrinsic_arr):
    """
    分析相机外参的**相邻帧相对运动**统计（真实轨迹长度和旋转）
    Args:
        extrinsic_arr: 外参数组 (N, 3, 4)，格式 [R | t]，表示相机到世界的变换(c2w)
    Returns:
        stats: 包含相邻帧平移和旋转统计的字典
    """
    if isinstance(extrinsic_arr, torch.Tensor):
        extrinsic_arr = extrinsic_arr.cpu().numpy()
    
    num_frames = extrinsic_arr.shape[0]
    if num_frames < 2:
        print("[WARNING] 帧数少于2，无法计算相邻帧相对运动")
        return {
            "relative_translation": None,
            "relative_rotation": None,
            "note": "Insufficient frames for relative motion analysis"
        }
    
    # 预分配数组存储相邻帧差值
    relative_translations = np.zeros(num_frames - 1, dtype=np.float32)
    relative_rotations_rad = np.zeros(num_frames - 1, dtype=np.float32)
    
    # 遍历所有相邻帧对
    for i in tqdm(range(num_frames - 1), desc="Calculating relative motion"):
        # 转换为4x4齐次矩阵
        c2w_i = np.eye(4, dtype=np.float64)
        c2w_i[:3, :] = extrinsic_arr[i]
        
        c2w_j = np.eye(4, dtype=np.float64)
        c2w_j[:3, :] = extrinsic_arr[i+1]
        
        # 计算相对变换: T_ij = c2w_j * inv(c2w_i)
        # 表示从第i帧相机坐标系到第j帧相机坐标系的变换
        T_ij = c2w_j @ np.linalg.inv(c2w_i)
        
        # 提取相对平移向量并计算距离
        t_ij = T_ij[:3, 3]
        relative_translations[i] = np.linalg.norm(t_ij)
        
        # 提取相对旋转矩阵并计算旋转角度
        R_ij = T_ij[:3, :3]
        trace = np.trace(R_ij)
        # 处理数值精度问题，确保cos_theta在[-1, 1]范围内
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        relative_rotations_rad[i] = np.arccos(cos_theta)
    
    # 转换为角度
    relative_rotations_deg = np.rad2deg(relative_rotations_rad)
    
    # 平移统计（单位：米）
    translation_stats = {
        "total_trajectory_length": float(np.sum(relative_translations)),
        "mean_per_frame": float(np.mean(relative_translations)),
        "median_per_frame": float(np.median(relative_translations)),
        "std_per_frame": float(np.std(relative_translations)),
        "min_per_frame": float(np.min(relative_translations)),
        "max_per_frame": float(np.max(relative_translations)),
        "unit": "meters"
    }
    
    # 旋转统计（同时保留弧度和角度）
    rotation_stats = {
        "total_rotation_rad": float(np.sum(relative_rotations_rad)),
        "mean_per_frame_rad": float(np.mean(relative_rotations_rad)),
        "median_per_frame_rad": float(np.median(relative_rotations_rad)),
        "std_per_frame_rad": float(np.std(relative_rotations_rad)),
        "min_per_frame_rad": float(np.min(relative_rotations_rad)),
        "max_per_frame_rad": float(np.max(relative_rotations_rad)),
        "total_rotation_deg": float(np.sum(relative_rotations_deg)),
        "mean_per_frame_deg": float(np.mean(relative_rotations_deg)),
        "median_per_frame_deg": float(np.median(relative_rotations_deg)),
        "std_per_frame_deg": float(np.std(relative_rotations_deg)),
        "min_per_frame_deg": float(np.min(relative_rotations_deg)),
        "max_per_frame_deg": float(np.max(relative_rotations_deg))
    }
    
    print(f"✅ 完成外参相邻帧相对运动统计: 轨迹长度 & 旋转角度")
    return {
        "relative_translation": translation_stats,
        "relative_rotation": rotation_stats,
        "num_frame_pairs": int(num_frames - 1)
    }

def analyze_intrinsic_elementwise(intrinsic_arr):
    """
    分析相机内参的逐元素独立统计（9个系数）
    Args:
        intrinsic_arr: 内参数组 (N, 3, 3)
    Returns:
        stats: 包含逐元素统计和关键参数汇总的字典
    """
    if isinstance(intrinsic_arr, torch.Tensor):
        intrinsic_arr = intrinsic_arr.cpu().numpy()
    
    element_wise_stats = {}
    key_params = {}
    
    # 遍历3x3矩阵的每个元素
    for i in range(3):
        for j in range(3):
            # 提取所有帧的该元素值
            element_values = intrinsic_arr[:, i, j]
            # 计算统计特征
            stats = {
                "mean": float(np.mean(element_values)),
                "median": float(np.median(element_values)),
                "std": float(np.std(element_values)),
                "min": float(np.min(element_values)),
                "max": float(np.max(element_values))
            }
            element_wise_stats[f"element_{i}_{j}"] = stats
            
            # 单独记录关键内参参数
            if i == 0 and j == 0:
                key_params["fx"] = stats
            elif i == 1 and j == 1:
                key_params["fy"] = stats
            elif i == 0 and j == 2:
                key_params["cx"] = stats
            elif i == 1 and j == 2:
                key_params["cy"] = stats
    
    print(f"✅ 完成内参逐元素独立统计: 9个系数")
    return {
        "element_wise": element_wise_stats,
        "key_parameters": key_params
    }

# =============================================================================
# 通用统计分析工具函数
# =============================================================================
def compute_statistics(arr, name, axis=None):
    """
    计算数组的核心统计特征
    Args:
        arr: 输入numpy数组或torch张量
        name: 数据字段名称（用于日志）
        axis: 计算维度，None表示全局统计，tuple表示指定维度
    Returns:
        stats: 统计特征字典（所有值均为Python原生类型）
    """
    if arr is None:
        return None
    
    # 统一转换为numpy数组
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    
    # 处理inf/nan值
    arr_clean = arr[np.isfinite(arr)]
    if len(arr_clean) == 0:
        print(f"[WARNING] {name}: 无有效数据（全为inf/nan）")
        return None
    
    # 计算核心统计量（全部转换为Python原生类型）
    stats = {
        "field_name": name,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "valid_samples": int(len(arr_clean)),
        "total_samples": int(np.prod(arr.shape)),
        "missing_ratio": float(1 - (len(arr_clean) / np.prod(arr.shape))),
        "min": float(np.min(arr_clean)),
        "max": float(np.max(arr_clean)),
        "mean": float(np.mean(arr_clean)),
        "median": float(np.median(arr_clean)),
        "std": float(np.std(arr_clean)),
        "var": float(np.var(arr_clean)),
        "q25": float(np.percentile(arr_clean, 25)),
        "q75": float(np.percentile(arr_clean, 75)),
        "iqr": float(np.percentile(arr_clean, 75) - np.percentile(arr_clean, 25)),
    }
    
    # 若指定维度，计算该维度下的统计
    if axis is not None:
        stats["axis_stats"] = {
            "mean_by_axis": np.mean(arr_clean, axis=axis).tolist(),
            "std_by_axis": np.std(arr_clean, axis=axis).tolist()
        }
    
    return stats

def plot_distribution(arr, name, save_path, bins=50, log_scale=False):
    """
    生成数据分布直方图并保存
    Args:
        arr: 输入numpy数组或torch张量
        name: 字段名称
        save_path: 保存路径
        bins: 直方图分箱数
        log_scale: 是否使用对数刻度
    """
    if arr is None:
        return
    
    # 预处理
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    arr_clean = arr[np.isfinite(arr)]
    if len(arr_clean) == 0:
        return
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 绘制直方图
    ax.hist(
        arr_clean.flatten(), 
        bins=bins,
        alpha=0.7,
        color='#2E86AB',
        edgecolor='#19547b'
    )
    
    # 设置刻度和标签
    ax.set_title(f'Distribution of {name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # 对数刻度（可选）
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (log scale)', fontsize=12)
    
    # 添加统计信息标注
    stats_text = f"""
    Mean: {np.mean(arr_clean):.4f}
    Std: {np.std(arr_clean):.4f}
    Min: {np.min(arr_clean):.4f}
    Max: {np.max(arr_clean):.4f}
    Median: {np.median(arr_clean):.4f}
    """.strip()
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存并关闭
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f"📊 已保存 {name} 分布图表: {save_path}")

# =============================================================================
# 主分析流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Predictions Data Analysis")
    
    # 核心参数
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to predictions.pt file saved by predict.py")
    parser.add_argument("--output_dir", type=str, default="./pred_analysis_results",
                        help="Directory to save analysis results (default: ./pred_analysis_results)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to analyze (default: None, analyze all)")
    
    # 可视化参数
    parser.add_argument("--plot_bins", type=int, default=50,
                        help="Number of bins for histograms (default: 50)")
    parser.add_argument("--log_scale", action='store_true',
                        help="Use log scale for frequency in histograms")
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(args.output_dir)}")

    # ── 加载预测数据 ─────────────────────────────────────────────────────
    print(f"\n🔍 加载预测文件: {args.pred_path}")
    t0 = time.time()
    
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"预测文件不存在: {args.pred_path}")
    
    # 加载数据（CPU模式）
    save_data = torch.load(args.pred_path, map_location="cpu", weights_only=False)
    predictions = save_data['predictions']
    loaded_args = save_data.get('args', {})
    image_shape = save_data.get('image_shape', None)
    paths = save_data.get('paths', None)
    resolved_image_folder = save_data.get('resolved_image_folder', None)
    
    print(f"✅ 加载完成 (耗时: {time.time() - t0:.1f}s)")
    print(f"📋 预测数据包含字段: {list(predictions.keys())}")
    print(f"📏 图像尺寸: {image_shape if image_shape else '未知'}")
    print(f"📁 原始图像文件夹: {resolved_image_folder}")
    print(f"📷 总帧数: {len(paths) if paths else '未知'}")

    # ── 数据截断（可选） ─────────────────────────────────────────────────
    num_frames = None
    if 'extrinsic' in predictions:
        num_frames = predictions['extrinsic'].shape[0]
        if args.max_frames is not None:
            num_frames_to_analyze = min(args.max_frames, num_frames)
            print(f"\n✂️ 截断数据至前 {num_frames_to_analyze} 帧 (原始: {num_frames} 帧)")
            
            # 截断所有帧维度的字段
            for key in predictions.keys():
                if predictions[key].ndim >= 1 and predictions[key].shape[0] == num_frames:
                    predictions[key] = predictions[key][:num_frames_to_analyze].clone()
    else:
        print("\n⚠️ 未找到extrinsic字段，无法确定帧数，跳过数据截断")

    # ── 核心统计分析 ─────────────────────────────────────────────────────
    analysis_results = {
        "metadata": {
            "pred_path": os.path.abspath(args.pred_path),
            "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_frames_used": args.max_frames,
            "total_frames_available": num_frames,
            "image_shape": image_shape,
            "resolved_image_folder": resolved_image_folder,
            "original_args": loaded_args
        },
        "statistics": {},
        "field_info": {}
    }

    # 定义需要分析的核心字段及配置
    fields_to_analyze = {
        "depth": {
            "axis": None,
            "plot": True,
            "log_scale": args.log_scale
        },
        "depth_conf": {
            "axis": None,
            "plot": True,
            "log_scale": args.log_scale
        },
        "extrinsic": {
            "axis": (0,),
            "plot": False,
            "special_analysis": analyze_extrinsic_relative  # 相邻帧相对运动分析
        },
        "intrinsic": {
            "axis": (0,),
            "plot": False,
            "special_analysis": analyze_intrinsic_elementwise  # 逐元素内参分析
        }
    }

    # 逐个分析字段
    print("\n📈 开始统计分析...")
    for field_name, config in tqdm(fields_to_analyze.items(), desc="Analyzing fields"):
        if field_name not in predictions:
            print(f"\n⚠️ 字段 {field_name} 不存在，跳过分析")
            analysis_results["statistics"][field_name] = None
            continue
        
        # 获取数据
        data = predictions[field_name]
        
        # 记录字段基本信息
        analysis_results["field_info"][field_name] = {
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "is_tensor": isinstance(data, torch.Tensor)
        }
        
        # 计算通用统计特征
        stats = compute_statistics(data, field_name, axis=config["axis"])
        
        # 执行专用分析（如果有）
        if stats is not None and "special_analysis" in config:
            special_stats = config["special_analysis"](data)
            stats.update(special_stats)
        
        analysis_results["statistics"][field_name] = stats
        
        # 生成分布可视化（若配置）
        if config["plot"] and stats is not None:
            plot_path = os.path.join(args.output_dir, f"{field_name}_distribution.png")
            plot_distribution(
                data, 
                field_name, 
                plot_path, 
                bins=args.plot_bins, 
                log_scale=config["log_scale"]
            )

    # ── 保存分析结果 ─────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4, ensure_ascii=False, cls=NumpyJSONEncoder)
    print(f"\n💾 完整分析结果已保存至: {json_path}")

    # ── 输出增强版摘要报告 ─────────────────────────────────────────────
    print("\n" + "="*100)
    print("📊 数据分析摘要报告")
    print("="*100)
    
    # 深度和置信度摘要
    for field_name in ["depth", "depth_conf"]:
        stats = analysis_results["statistics"].get(field_name)
        if stats is None:
            continue
        
        unit = "meters" if field_name == "depth" else "confidence score"
        print(f"\n[{field_name}] ({unit})")
        print(f"  形状: {stats['shape']} | 有效值占比: {100*(1-stats['missing_ratio']):.2f}%")
        print(f"  均值±标准差: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  数值范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  中位数/四分位距: {stats['median']:.4f} / {stats['iqr']:.4f}")
    
    # 外参相邻帧相对运动摘要
    extrinsic_stats = analysis_results["statistics"].get("extrinsic")
    if extrinsic_stats is not None and extrinsic_stats.get("relative_translation") is not None:
        print(f"\n[extrinsic] 相机外参 (总帧数={extrinsic_stats['shape'][0]}, 相邻帧对={extrinsic_stats['num_frame_pairs']})")
        print(f"  形状: {extrinsic_stats['shape']} | 有效值占比: {100*(1-extrinsic_stats['missing_ratio']):.2f}%")
        print(f"\n  📍 相邻帧平移距离统计 (单位: ?):")
        trans = extrinsic_stats["relative_translation"]
        print(f"    累计总轨迹长度: {trans['total_trajectory_length']:.4f}")
        print(f"    单帧均值: {trans['mean_per_frame']:.4f}")
        print(f"    单帧中位数: {trans['median_per_frame']:.4f}")
        print(f"    范围: [{trans['min_per_frame']:.4f}, {trans['max_per_frame']:.4f}]")
        print(f"\n  🔄 相邻帧旋转角度统计:")
        rot = extrinsic_stats["relative_rotation"]
        print(f"    累计总旋转角度: {rot['total_rotation_deg']:.2f}°")
        print(f"    单帧均值: {rot['mean_per_frame_deg']:.2f}°")
        print(f"    单帧中位数: {rot['median_per_frame_deg']:.2f}°")
        print(f"    范围: [{rot['min_per_frame_deg']:.2f}°, {rot['max_per_frame_deg']:.2f}°]")
    
    # 内参逐元素摘要
    intrinsic_stats = analysis_results["statistics"].get("intrinsic")
    if intrinsic_stats is not None:
        print(f"\n[intrinsic] 相机内参 (总帧数={intrinsic_stats['shape'][0]})")
        print(f"  形状: {intrinsic_stats['shape']} | 有效值占比: {100*(1-intrinsic_stats['missing_ratio']):.2f}%")
        print(f"\n  🎯 关键参数统计 (均值±标准差):")
        key_params = intrinsic_stats["key_parameters"]
        print(f"    fx (x轴焦距): {key_params['fx']['mean']:.2f} ± {key_params['fx']['std']:.2f}")
        print(f"    fy (y轴焦距): {key_params['fy']['mean']:.2f} ± {key_params['fy']['std']:.2f}")
        print(f"    cx (x轴主点): {key_params['cx']['mean']:.2f} ± {key_params['cx']['std']:.2f}")
        print(f"    cy (y轴主点): {key_params['cy']['mean']:.2f} ± {key_params['cy']['std']:.2f}")
        print(f"\n  完整9元素逐帧统计已保存至JSON文件")
    
    print("\n" + "="*100)
    print(f"✅ 分析完成！所有结果已保存至: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
