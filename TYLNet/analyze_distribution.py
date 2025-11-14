import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# --- 1. 配置路径 ---
# (这必须与 generate_confidence.py 中的输出路径一致)
model_name = "TCLNET_cbam" # <--- !! 修改这里 !!
checkpoint_dir = "checkpointsTCLNet" # <--- !! 修改这里 !!

base_path = os.path.join(checkpoint_dir, model_name)
pos_scores_file = os.path.join(base_path, "positive_scores.txt")
neg_scores_file = os.path.join(base_path, "negative_scores.txt")
output_plot_file = os.path.join(base_path, "confidence_distribution.png")
# --------------------

def analyze():
    print("Loading scores...")
    try:
        pos_scores = np.loadtxt(pos_scores_file)
    except Exception as e:
        print(f"Error loading {pos_scores_file}: {e}")
        return

    try:
        neg_scores = np.loadtxt(neg_scores_file)
    except Exception as e:
        print(f"Error loading {neg_scores_file}: {e}")
        return

    print(f"Loaded {len(pos_scores)} positive scores (Mean: {np.mean(pos_scores):.4f})")
    print(f"Loaded {len(neg_scores)} negative scores (Mean: {np.mean(neg_scores):.4f})")

    # --- 2. 绘制直方图 ---
    print("Generating plot...")
    plt.figure(figsize=(12, 7))
    
    # (使用 'density=True' 可以使它们在同一尺度上)
    plt.hist(pos_scores, bins=50, alpha=0.7, label=f'Positive (Typhoon) (n={len(pos_scores)})', density=True, color='blue')
    plt.hist(neg_scores, bins=50, alpha=0.7, label=f'Negative (Non-Typhoon) (n={len(neg_scores)})', density=True, color='red')
    
    plt.xlabel('Confidence Score (Max Heatmap Value)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Confidence Score Distribution', fontsize=16)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 增加一条垂直线，帮助你目测阈值
    plt.axvline(np.mean(pos_scores), color='blue', linestyle='--', alpha=0.5)
    plt.axvline(np.mean(neg_scores), color='red', linestyle='--', alpha=0.5)
    
    # --- 3. 保存图像 ---
    plt.savefig(output_plot_file)
    print(f"Plot saved to {output_plot_file}")
    print("Please open the .png file to analyze the distribution and choose your threshold.")

if __name__ == '__main__':
    # 确保已安装 matplotlib (pip install matplotlib)
    analyze()


