#!/usr/bin/env python3
"""Check dtype support for current hardware."""

import torch
import sys


def check_dtype_support():
    """Comprehensive dtype support check."""
    
    print("=" * 80)
    print("PyTorch dtype Support Checker")
    print("=" * 80)
    
    # 1. All available dtypes
    print("\n[1] All PyTorch dtypes:")
    all_dtypes = [attr for attr in dir(torch) if isinstance(getattr(torch, attr), torch.dtype)]
    for dtype_name in sorted(all_dtypes):
        dtype_obj = getattr(torch, dtype_name)
        print(f"    {dtype_name:20s} = {dtype_obj}")
    
    # 2. Floating point dtypes (most relevant for deep learning)
    print("\n[2] Floating-point dtypes:")
    float_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    for dtype in float_dtypes:
        bits = torch.finfo(dtype).bits
        print(f"    {str(dtype):20s} ({bits}-bit)")
    
    # 3. CPU support
    print("\n[3] CPU dtype support:")
    cpu_device = torch.device('cpu')
    for dtype in float_dtypes:
        try:
            x = torch.randn(100, 100, dtype=dtype, device=cpu_device)
            y = x @ x.T
            print(f"    ✓ {str(dtype):20s} supported on CPU")
        except Exception as e:
            print(f"    ✗ {str(dtype):20s} NOT supported on CPU: {e}")
    
    # 4. CUDA support
    if torch.cuda.is_available():
        print("\n[4] CUDA device information:")
        num_gpus = torch.cuda.device_count()
        print(f"    Number of GPUs: {num_gpus}")
        
        for gpu_id in range(num_gpus):
            print(f"\n    GPU {gpu_id}:")
            print(f"      Name:              {torch.cuda.get_device_name(gpu_id)}")
            print(f"      Compute Capability: {torch.cuda.get_device_capability(gpu_id)}")
            
            cuda_device = torch.device(f'cuda:{gpu_id}')
            
            print(f"      Dtype support:")
            for dtype in float_dtypes:
                try:
                    x = torch.randn(100, 100, dtype=dtype, device=cuda_device)
                    y = x @ x.T
                    torch.cuda.synchronize()
                    
                    # Check if operation produced valid results
                    if torch.isnan(y).any() or torch.isinf(y).any():
                        print(f"        ⚠ {str(dtype):20s} supported but produces NaN/Inf")
                    else:
                        print(f"        ✓ {str(dtype):20s} fully supported")
                        
                except RuntimeError as e:
                    print(f"        ✗ {str(dtype):20s} NOT supported: {str(e)[:60]}")
                except Exception as e:
                    print(f"        ✗ {str(dtype):20s} Error: {str(e)[:60]}")
            
            # Memory info
            total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            print(f"      Total Memory:     {total_mem:.2f} GB")
    else:
        print("\n[4] CUDA: Not available")
    
    # 5. Recommendation for your code
    print("\n[5] Recommendation for lingbot-map:")
    if torch.cuda.is_available():
        cc_major = torch.cuda.get_device_capability()[0]
        if cc_major >= 8:
            print(f"    ✓ Your GPU (CC {cc_major}.x) supports bfloat16 natively")
            print(f"    → Recommended dtype: torch.bfloat16")
            print(f"    → Fallback: torch.float16")
        elif cc_major >= 7:
            print(f"    ⚠ Your GPU (CC {cc_major}.x) has limited bfloat16 support")
            print(f"    → Recommended dtype: torch.float16")
            print(f"    → Note: bfloat16 may be emulated (slower)")
        else:
            print(f"    ✗ Your GPU (CC {cc_major}.x) does not support bfloat16 well")
            print(f"    → Recommended dtype: torch.float16")
            print(f"    → For better precision: torch.float32 (uses more memory)")
    else:
        print(f"    → Using CPU: torch.float32 recommended")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    check_dtype_support()