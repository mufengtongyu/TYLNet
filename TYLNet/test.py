from datasets import create_dataset
from modules import create_model
from utils import startup
import os
import utils.tools as util
import numpy as np
import evaluation
import cv2

# --- 1. 新增导入 (包含之前的修复) ---
import torch
import copy # 用于 thop 修复
from thop import profile
# --------------------


def test(config):
    config['num_threads'] = 1                     # only <num_threads = 1> supported when testing_usr
    config['flip'] = False                        # not allowed to flip image
    config['status'] = 'test'
    config['crop_scale'] = 1.0

    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)

    # --- 2. 计算 GFLOPs 和 Params (最终修复版) ---
    print("Calculating model parameters and GFLOPs...")
    device = model.device  
    img_size = int(config['img_size']) 
    input_channels = 6 
    
    net_to_profile = copy.deepcopy(model.netG.module)
    net_to_profile.cpu()
    dummy_input_cpu = torch.randn(1, input_channels, img_size, img_size)
    net_to_profile.eval() 
    flops, params = profile(net_to_profile, inputs=(dummy_input_cpu, ), verbose=False)
    
    params_m = params / 1e6
    gflops = flops / 1e9
    
    print(f"Model Parameters: {params_m:.2f} M")
    print(f"Model GFLOPs: {gflops:.2f} G")
    # ---------------------------------------------

    result_root_path = os.path.join(config['checkpoints_dir'], config['name'], 'evaluation')
    util.mkdir(result_root_path)
    util.mkdir(os.path.join(result_root_path,'prediction_distance'))
    util.mkdir(os.path.join(result_root_path,'prediction_heatmap'))
    print(" create evaluate folder: " + result_root_path)

    # set module to testing_usr mode
    model.eval()

    # (np.float64 修复)
    save_npy = np.ndarray(shape=(dataset.__len__()+1,2),dtype=np.float64)
    save_npy[0][0],save_npy[0][1] = -1,-1
    
    # --- 3. 新增：检测逻辑和统计 ---
    CONF_THRESHOLD = 0.4  # Updated threshold for typhoon detection
    detected_count = 0
    missed_count = 0
    total_count = len(dataset)

    # --- Load category indices for per-class detection accuracy ---
    category_files = [
        ("TS", "Index2.npy"),
        ("Cat1", "Index3.npy"),
        ("Cat2", "Index4.npy"),
        ("Cat3", "Index5.npy"),
        ("Cat4", "Index6.npy"),
        ("Cat5", "Index7.npy"),
    ]

    category_sets = {}
    category_stats = {}
    for cat_name, filename in category_files:
        filepath = os.path.join(evaluation.data_dir, filename)
        if os.path.exists(filepath):
            indices = np.load(filepath)
            category_sets[cat_name] = set(int(idx) for idx in indices.tolist())
            category_stats[cat_name] = {"detected": 0, "total": len(indices)}
        else:
            category_sets[cat_name] = set()
            category_stats[cat_name] = {"detected": 0, "total": 0}
    # ---------------------------------

    for i, data in enumerate(dataset):
        model.set_input(data)  # push test datasets to module
        model.test()  # forward module

        # --- 4. 新增：集成检测逻辑 ---
        # 从 (File 11) 的 test_result 中获取我们添加的 'CONFIDENCE'
        confidence = model.test_result[3][1]
        index = int(data["PATH"].cpu().data.numpy()[0])

        if confidence >= CONF_THRESHOLD:
            # 检测成功 (True Positive)
            detected_count += 1

            # 记录类别检测结果
            for cat_name, idx_set in category_sets.items():
                if index in idx_set:
                    category_stats[cat_name]["detected"] += 1

            # --- 仅在检测成功时，才执行原有的保存逻辑 ---
            datapoints = (model.test_result[0][1]).cpu().data.numpy()
            save_npy[index][0],save_npy[index][1] = datapoints[0][0], datapoints[0][1]

            dist_img = model.test_result[1][1]
            util.save_image(util.tensor2im(dist_img), os.path.join(result_root_path,'prediction_distance', str(index) + ".png"))

            heatmap_img = model.test_result[2][1]
            util.save_image(util.tensor2im(heatmap_img),os.path.join(result_root_path, 'prediction_heatmap', str(index) + ".png"))
            
            print(f"Evaluate forward-- {i+1}/{total_count} -- DETECTED (Conf: {confidence:.4f})")
        else:
            # 检测失败 (False Negative)
            missed_count += 1
            print(f"Evaluate forward-- {i+1}/{total_count} -- MISSED (Conf: {confidence:.4f})")
        # ---------------------------------

    np.save(os.path.join(result_root_path,'regression.npy'),save_npy)
    
    # --- 5. 修复：解包 7 个返回值 ---
    l2_dist, dist1, dist2, dist3, dist4, dist5, dist6 = evaluation.evaluate_detailed(save_npy)
    print("Testing npy result have been saved!")
    
    # --- 6. 新增：在终端打印检测报告 ---
    detection_accuracy = (detected_count / total_count) * 100
    print("\n--- Detection Report (on Positive Test Set) ---")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print(f"Total Test Samples:   {total_count}")
    print(f"Detected (True Pos):  {detected_count}")
    print(f"Missed (False Neg):   {missed_count}")
    print(f"Detection Accuracy:   {detection_accuracy:.2f}%")
    print("\n--- Per-Category Detection Accuracy ---")
    for cat_name, stats in category_stats.items():
        if stats["total"] == 0:
            print(f"  {cat_name}: N/A (no samples)")
        else:
            cat_acc = (stats["detected"] / stats["total"]) * 100
            print(f"  {cat_name}: {stats['detected']}/{stats['total']} ({cat_acc:.2f}%)")
    print("\n--- Localization Report (for Detected Samples) ---")
    print(f"Overall MAE (px): {l2_dist:.4f}")
    print(f"  Dist1 (TS):   {dist1:.4f}")
    print(f"  Dist2 (Cat1): {dist2:.4f}")
    print(f"  Dist3 (Cat2): {dist3:.4f}")
    print(f"  Dist4 (Cat3): {dist4:.4f}")
    print(f"  Dist5 (Cat4): {dist5:.4f}")
    print(f"  Dist6 (Cat5): {dist6:.4f}")
    # ---------------------------------
    
    # --- 7. 新增：将所有结果写入 evaluation.txt ---
    eval_file_path = os.path.join(result_root_path, 'evaluation_summary.txt')
    with open(eval_file_path, "w") as text_file:
        text_file.write(f"--- Model Stats ---\n")
        text_file.write(f"Model: {config['name']}\n")
        text_file.write(f"Checkpoint: {config['test_epoch']}_net_G.pth\n")
        text_file.write(f"Parameters (M): {params_m:.2f}\n")
        text_file.write(f"GFLOPs (Input: 1x{input_channels}x{img_size}x{img_size}): {gflops:.2f}\n\n")

        text_file.write(f"--- Detection Report (on Positive Test Set) ---\n")
        text_file.write(f"Confidence Threshold: {CONF_THRESHOLD}\n")
        text_file.write(f"Total Test Samples:   {total_count}\n")
        text_file.write(f"Detected (True Pos):  {detected_count}\n")
        text_file.write(f"Missed (False Neg):   {missed_count}\n")
        text_file.write(f"Detection Accuracy:   {detection_accuracy:.2f}%\n\n")

        text_file.write(f"--- Per-Category Detection Accuracy ---\n")
        for cat_name, stats in category_stats.items():
            if stats["total"] == 0:
                text_file.write(f"{cat_name}: N/A (no samples)\n")
            else:
                cat_acc = (stats["detected"] / stats["total"]) * 100
                text_file.write(
                    f"{cat_name}: {stats['detected']}/{stats['total']} ({cat_acc:.2f}%)\n"
                )
        text_file.write("\n")

        text_file.write(f"--- Localization Report (for Detected Samples) ---\n")
        text_file.write(f"Overall MAE (px): {l2_dist:.4f}\n")
        text_file.write(f"Dist1 (TS): {dist1:.4f}\n")
        text_file.write(f"Dist2 (Cat1): {dist2:.4f}\n")
        text_file.write(f"Dist3 (Cat2): {dist3:.4f}\n")
        text_file.write(f"Dist4 (Cat3): {dist4:.4f}\n")
        text_file.write(f"Dist5 (Cat4): {dist5:.4f}\n")
        text_file.write(f"Dist6 (Cat5): {dist6:.4f}\n")
    
    print(f"\nModel stats and evaluation results saved to {eval_file_path}")
    # ---------------------------------------------


if __name__ == '__main__':
    configs = startup.SetupConfigs(config_path='configs/TCLNET_STAGE1.yaml')
    configs = configs.setup()
    
    # --- 确保 test.py 使用 'test' 状态 ---
    configs['status'] = 'test'
    # 您必须在 config 文件 (File 1) 中设置 'test_epoch'
    # 或者在这里手动设置
    # configs['test_epoch'] = 20 # 例如
    
    test(configs)


# from datasets import create_dataset
# from modules import create_model
# from utils import startup
# import os
# import utils.tools as util
# import numpy as np
# import evaluation
# import cv2



# def test(config):
#     config['num_threads'] = 1                     # only <num_threads = 1> supported when testing_usr
#     config['flip'] = False                        # not allowed to flip image
#     config['status'] = 'test'
#     config['crop_scale'] = 1.0

#     dataset = create_dataset(config)
#     model = create_model(config)
#     model.setup(config)

#     result_root_path = os.path.join(config['checkpoints_dir'], config['name'], 'evaluation') # 这里更改测试结果保存路径
#     util.mkdir(result_root_path)
#     util.mkdir(os.path.join(result_root_path,'prediction_distance'))
#     util.mkdir(os.path.join(result_root_path,'prediction_heatmap'))
#     print(" create evaluate folder: " + result_root_path)

#     # set module to testing_usr mode
#     model.eval()

#     save_npy = np.ndarray(shape=(dataset.__len__()+1,2),dtype=np.float32)
#     save_npy[0][0],save_npy[0][1] = -1,-1

#     for i, data in enumerate(dataset):
#         model.set_input(data)  # push test datasets to module
#         model.test()  # forward module

#         datapoints = (model.test_result[0][1]).cpu().data.numpy()
#         index = data["PATH"].cpu().data.numpy()[0]
#         save_npy[index][0],save_npy[index][1] = datapoints[0][0], datapoints[0][1]

#         dist_img = model.test_result[1][1]
#         util.save_image(util.tensor2im(dist_img), os.path.join(result_root_path,'prediction_distance', str(index) + ".png"))

#         heatmap_img = model.test_result[2][1]
#         util.save_image(util.tensor2im(heatmap_img),os.path.join(result_root_path, 'prediction_heatmap', str(index) + ".png"))

#         print("Evaluate forward-- complete:" + str(i + 1) + "  total:" + str(dataset.__len__()))

#     np.save(os.path.join(result_root_path,'regression.npy'),save_npy)
#     # l2_dist, easy_dist, hard_dist = evaluation.evaluate_detailed(save_npy)
#     # 修改后
#     l2_dist, dist1, dist2, dist3, dist4, dist5, dist6 = evaluation.evaluate_detailed(save_npy)
#     # print("Testing npy result have been saved! Evaluation distance: " + str(round(l2_dist)) + "(" + str(round(easy_dist)) + "," + str(round(hard_dist)) + ")")
#     # 修改后
#     print("Testing npy result have been saved! Evaluation distance: " + str(round(l2_dist, 4)) + '   ' +
#         str(round(dist1, 4)) + '   ' + str(round(dist2, 4)) + '   ' +
#         str(round(dist3, 4)) + '   ' + str(round(dist4, 4)) + '   ' +
#         str(round(dist5, 4)) + '   ' + str(round(dist6, 4)))


# if __name__ == '__main__':
#     configs = startup.SetupConfigs(config_path='configs/TCLNET_STAGE1.yaml') # 原先是STAGE2但是没提供
#     configs = configs.setup()
#     test(configs)


