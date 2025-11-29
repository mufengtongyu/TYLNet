import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm # 用于显示进度条

# 导入您项目中的模块
from modules import create_model
from utils import startup

# --- 关键：从 heatmap_dataset.py (File 4) 复制归一化函数 ---
# (此版本现在与 File 4 的逻辑 100% 匹配)
def normalization(data, norm_type="standard"):
    if norm_type == "standard":
        mean = [0.5]
        std = [0.5]
    elif norm_type == 'imagenet':
        mean = [0.485]
        std = [0.229]
    else:
        raise NotImplementedError("norm_type mismatched!")
    
    # (File 4 的逻辑是对 [1, 224, 224] 的张量应用此归一化)
    transform = transforms.Normalize(mean, std, inplace=False)
    return transform(data)
# --------------------------------------------------------

def process_folder(model, device, data_root_list, norm_type, output_file):
    """
    遍历文件夹，处理所有图像并保存置信度
    data_root_list: 包含6个通道文件夹路径的列表
    """
    
    print(f"Processing data folders: {data_root_list[0]} ... etc.")
    
    try:
        image_names = [f for f in os.listdir(data_root_list[0]) if f.endswith('.png')]
        if not image_names:
            print(f"Error: No .png files found in {data_root_list[0]}")
            return
    except FileNotFoundError:
        print(f"Error: Folder not found {data_root_list[0]}")
        return

    confidence_scores = []
    model.eval() # 确保模型在评估模式

    with torch.no_grad():
        for img_name in tqdm(image_names, desc=f"Processing {output_file}"):
            try:
                image_inputs_normalized = []
                for path in data_root_list:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) 
                    
                    # --- 修复 Bug 2：归一化逻辑 ---
                    # 严格遵循 File 4, line 48 的逻辑
                    # 1. 转为 [1, 224, 224] 张量
                    img_tensor = torch.from_numpy(img).unsqueeze(0).float() / 255
                    # 2. 对单通道进行归一化
                    img_tensor_normalized = normalization(img_tensor, norm_type)
                    image_inputs_normalized.append(img_tensor_normalized)
                    # --------------------------------
                
                # (File 4, line 50) 在归一化 *之后* 堆叠
                image_tensor = torch.cat(image_inputs_normalized, dim=0) # [6, 224, 224]
                
                # 增加-batch 维度并发送到设备
                image_tensor = image_tensor.unsqueeze(0).to(device) # [1, 6, 224, 224]
                
                # --- 核心推理与置信度提取 ---
                # (File 11, line 50) 模型前向传播
                output_heatmap_56 = model.netG(image_tensor) 
                
                # (File 11, line 59) 上采样到 224x224
                PREDICTION_HEATMAP = F.upsample_nearest(output_heatmap_56, (224, 224))
                
                # (File 11, line 61) 提取热力图的峰值
                confidence = torch.max(PREDICTION_HEATMAP).item()
                confidence_scores.append(confidence)
                # ------------------------------

            except Exception as e:
                print(f"Warning: Failed to process {img_name}. Error: {e}")
                continue
    
    # 保存结果
    np.savetxt(output_file, np.array(confidence_scores), fmt='%.8f')
    print(f"Saved {len(confidence_scores)} scores to {output_file}")


def main():
    # --- 1. 加载您训练好的SOTA模型 ---
    print("Loading config...")
    # (File 1) 配置文件
    config_path = 'configs/TCLNET_STAGE1.yaml' 
    configs = startup.SetupConfigs(config_path=config_path).setup()
    
    # **重要**: 将 test_epoch 设置为您要评估的最佳模型
    configs['test_epoch'] = 21 # <--- !! 修改这里 !!
    configs['status'] = 'test'
    
    # --- 修复 Bug 1：TypeError ---
    # `gpu_ids` 必须是一个列表(list)，而不是整数(int)
    # 原始错误代码: configs['gpu_ids'] = 0
    configs['gpu_ids'] = [0] # 确保在GPU上运行
    # -----------------------------
    
    print("Loading model...")
    model = create_model(configs)
    model.setup(configs) # (File 10, line 78) 加载网络权重
    
    norm_type = configs['norm_type']
    device = model.device
    
    # --- 2. 定义您的正样本集路径 (原始测试集) ---
    # (File 4, line 30) 路径结构
    # (来自您的 traceback)
    dataroot_base = "/home/littlered/Documents/Improved-TCLNet/datasets/data/TCLDwithTSsame" 
    positive_paths = [
        "/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/10/TEST_INPUT",
        "/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/11/TEST_INPUT",
        "/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/12/TEST_INPUT",
        "/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/13/TEST_INPUT",
        "/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/14/TEST_INPUT",
        "/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/15/TEST_INPUT"
    ] # <--- !! 修改这里为你自己的路径 !!
    
    # --- 3. 定义您的负样本集路径 ---
    negative_paths = [
        "/home/littlered/Documents/TYLNet2/datasets/A202511121231116824/C09/tiles_224x224_s122",
        "/home/littlered/Documents/TYLNet2/datasets/A202511121231116824/C10/tiles_224x224_s122",
        "/home/littlered/Documents/TYLNet2/datasets/A202511121231116824/C11/tiles_224x224_s122",
        "/home/littlered/Documents/TYLNet2/datasets/A202511121231116824/C12/tiles_224x224_s122",
        "/home/littlered/Documents/TYLNet2/datasets/A202511121231116824/C13/tiles_224x224_s122",
        "/home/littlered/Documents/TYLNet2/datasets/A202511121231116824/C14/tiles_224x224_s122"
    ] # <--- !! 修改这里为你自己的路径 !!
    
    # --- 4. 运行处理 ---
    # (确保 checkpoint 文件夹和 config['name'] 对应的文件夹存在)
    output_dir = os.path.join(configs['checkpoints_dir'], configs['name'])
    
    process_folder(model, device, positive_paths, norm_type, os.path.join(output_dir, "positive_scores.txt"))
    process_folder(model, device, negative_paths, norm_type, os.path.join(output_dir, "negative_scores.txt"))

    print("All done.")

if __name__ == '__main__':
    main()