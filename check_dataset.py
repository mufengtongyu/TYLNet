import torch
import cv2
import numpy as np
import yaml
import os
from torch.utils.data import DataLoader

# 确保 datasets.heatmap_dataset  已经被第1步的代码修复
from datasets.heatmap_dataset import heatmapDataset 

# 导入您的配置加载
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def tensor_to_cv_img(tensor_img):
    """将 [C, H, W] 的 PyTorch Tensor 转换为 [H, W, C] 的 OpenCV BGR 图像"""
    img = tensor_img.cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0)) # H, W, C
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy() 

def draw_validation_image(data_sample):
    """
    在图像上绘制所有标签：BBox, Heatmap, Center Point
    """
    
    # 1. 获取可视化底图 (IMAGE_SHOW)
    vis_img_tensor = data_sample['IMAGE_SHOW']
    vis_img = tensor_to_cv_img(vis_img_tensor) # [224, 224, 3], BGR, 0-255
    
    # 2. 获取热图 (HEATMAP_TARGET)
    heatmap_tensor = data_sample['HEATMAP_TARGET']
    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    heatmap = (heatmap * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 将热图和底图混合
    overlay_img = cv2.addWeighted(vis_img, 0.5, heatmap_color, 0.5, 0)
    
    # 3. 获取检测标签 (DETECTION_TARGET)
    detection_target = data_sample['DETECTION_TARGET']
    box = detection_target['boxes'][0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    
    # 在叠加图上绘制 BBox (绿色)
    cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # 绿色
    
    # 4. 获取真实坐标 (LABEL)
    label_coord = data_sample['LABEL'].cpu().numpy().astype(int)
    cx, cy = label_coord
    
    # 在叠加图上绘制中心点 (红色)
    cv2.circle(overlay_img, (cx, cy), 3, (0, 0, 255), -1) # 红色圆点
    
    return overlay_img

def main():
    print("--- 开始批量校验数据集修改效果 (前5张图) ---")
    
    # 1. 加载配置
    config_path = './configs/TCLNET_STAGE1.yaml' # 确保路径正确 [cite: 5]
    config = load_config(config_path)
    config['status'] = 'test' 
    
    # 2. 初始化您修改后的 Dataset
    print(f"加载数据集: {config['dataset_mode']}...") 
    try:
        dataset = heatmapDataset(config) # 
    except Exception as e:
        print(f"错误: 初始化 Dataset 失败: {e}")
        return

    if len(dataset) == 0:
        print("错误: 数据集长度为0，请检查 dataroot 路径。")
        return

    # 3. 【修改】循环绘制前5张图
    num_images_to_check = 50
    print(f"正在获取并绘制前 {num_images_to_check} 个样本...")
    
    for i in range(num_images_to_check):
        try:
            data_sample = dataset[i]
        except Exception as e:
            print(f"错误: __getitem__({i}) 失败: {e}")
            continue

        # 4. 检查返回的数据结构
        required_keys = ['IMAGE', 'IMAGE_SHOW', 'HEATMAP_TARGET', 'DETECTION_TARGET', 'LABEL']
        if not all(key in data_sample for key in required_keys):
            print(f"错误: 样本 {i} 返回的字典缺少必要的键。")
            continue
            
        print(f"  正在处理样本 {i} (PATH: {data_sample['PATH']})...")
        print(f"    LABEL (cx, cy): {data_sample['LABEL']}")

        # 5. 绘制可视化结果
        validation_image = draw_validation_image(data_sample)
        
        # 6. 保存图像
        output_path = f"dataset_validation_{i}.png"
        cv2.imwrite(output_path, validation_image)
        print(f"    已保存结果到: {output_path}")

    print(f"--- 批量校验完成！---")
    print(f"请检查生成的 {num_images_to_check} 张 'dataset_validation_*.png' 图像。")

if __name__ == "__main__":
    main()