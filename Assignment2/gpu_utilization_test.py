import torch
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备：{device}")

if device.type == "cuda":
    print(f"GPU型号：{torch.cuda.get_device_name(0)}")
    print(f"GPU内存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建大张量并测量GPU利用率
    print("\n正在进行GPU密集型计算测试...")
    
    # 记录初始内存使用情况
    initial_memory = torch.cuda.memory_allocated(device)
    print(f"初始GPU内存使用: {initial_memory / 1024**2:.1f} MB")
    
    # 创建大矩阵进行计算
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 记录中间内存使用
    mid_memory = torch.cuda.memory_allocated(device)
    print(f"创建矩阵后GPU内存使用: {mid_memory / 1024**2:.1f} MB")
    
    # 执行密集计算
    start_time = time.time()
    for i in range(10):
        c = torch.mm(a, b)  # 矩阵乘法
        if i % 5 == 0:
            print(f"完成第 {i+1}/10 次矩阵乘法")
    end_time = time.time()
    
    computation_time = end_time - start_time
    print(f"GPU计算耗时: {computation_time:.2f} 秒")
    
    # 检查最终内存使用
    final_memory = torch.cuda.memory_allocated(device)
    print(f"计算完成后GPU内存使用: {final_memory / 1024**2:.1f} MB")
    
    # 清理
    del a, b, c
    torch.cuda.empty_cache()
    
    print("\n现在对比CPU和GPU计算性能...")
    
    # CPU上的等效计算
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    for i in range(5):  # 较少次数，因为CPU较慢
        c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    
    print(f"CPU计算耗时 (5次): {cpu_time:.2f} 秒")
    
    # GPU上的等效计算
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    
    # 预热GPU
    for i in range(5):
        _ = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()  # 确保GPU操作完成
    
    start_time = time.time()
    for i in range(5):
        c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()  # 确保GPU操作完成
    gpu_time = time.time() - start_time
    
    print(f"GPU计算耗时 (5次, 同步): {gpu_time:.2f} 秒")
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"GPU相对于CPU的加速比: {speedup:.2f}x")
    
    if speedup > 1:
        print("✅ GPU正在有效加速计算!")
    else:
        print("❌ GPU加速效果不佳，可能存在兼容性问题")
        
else:
    print("CUDA不可用，无法进行GPU测试")