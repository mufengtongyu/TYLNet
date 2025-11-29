import torch
import time
import os

# 导入您项目中的模块
from modules import create_model
from utils import startup

def run_benchmark():
    # --- 1. 加载配置 ---
    print("Loading config...")
    config_path = 'configs/TCLNET_STAGE1.yaml' 
    configs = startup.SetupConfigs(config_path=config_path).setup()
    
    # **重要**: 将 test_epoch 设置为您要评估的最佳模型
    # 例如，如果您在第 20 轮得到了最佳结果
    configs['test_epoch'] = 20 # <--- !! 修改这里 !!
    
    configs['status'] = 'test'
    configs['gpu_ids'] = [0] # 确保在 GPU 上
    
    # --- 2. 加载模型 ---
    print(f"Loading model: {configs['name']} (Epoch: {configs['test_epoch']})")
    model = create_model(configs)
    model.setup(configs) # (File 10, line 78) 加载网络权重
    
    # !!! 关键：必须设置为评估模式 !!!
    # 这会关闭 Dropout 等，对速度至关重要
    model.eval() 
    
    device = model.device
    netG = model.netG # 我们只测试 netG 的速度

    # --- 3. 准备输入数据 ---
    img_size = int(configs['img_size'])
    input_channels = 6 # (File 4)
    
    # 创建一个虚拟的输入张量 (batch size = 1)
    # 并将其预先加载到 GPU 内存中
    dummy_input = torch.randn(1, input_channels, img_size, img_size).to(device)
    
    print(f"\n--- Benchmarking on {torch.cuda.get_device_name(0)} ---")
    print(f"Input tensor: {dummy_input.shape}")

    # --- 4. GPU 预热 (Warm-up) ---
    print("Warming up GPU...")
    warmup_runs = 20
    with torch.no_grad(): # 禁用梯度计算
        for _ in range(warmup_runs):
            _ = netG(dummy_input)
    
    # 确保预热完成
    torch.cuda.synchronize() 
    print("Warm-up complete.")

    # --- 5. 运行基准测试 ---
    num_runs = 100 # 循环 100 次以获得平均值
    
    print(f"Running benchmark ({num_runs} runs)...")
    
    # (关键) 确保在计时前所有 GPU 操作已完成
    torch.cuda.synchronize()
    
    # 使用 time.perf_counter() 进行高精度计时
    start_time = time.perf_counter()

    with torch.no_grad(): # 确保在无梯度模式下运行
        for _ in range(num_runs):
            _ = netG(dummy_input)

    # (关键) 确保在计时结束前所有 GPU 操作已完成
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # --- 6. 计算并报告结果 ---
    total_time_seconds = end_time - start_time
    avg_latency_ms = (total_time_seconds / num_runs) * 1000 # 平均延迟（毫秒）
    fps = 1.0 / (total_time_seconds / num_runs) # 吞吐量 (FPS)

    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_runs} runs: {total_time_seconds:.3f} seconds")
    print(f"Average Latency (ms):   {avg_latency_ms:.3f} ms / frame")
    print(f"Throughput (FPS):       {fps:.2f} FPS")

if __name__ == '__main__':
    run_benchmark()


