from datasets import create_dataset
from modules import create_model
from utils import startup
import os
import utils.tools as util
import numpy as np
import evaluation

import torch
import copy
from thop import profile

# 每个像素对应的实际距离（km）
KM_PER_PIXEL = 4.0

# 置信度阈值（可根据直方图/ROC 调整）
CONF_THRESHOLD = 0.4

# 当前这次 test() 跑的是不是“台风测试集”
#  - 台风测试集：True  ->  对应论文中的正样本集（有台风）
#  - 非台风测试集：False ->  对应论文中的负样本集（无台风）
IS_TYPHOON_SET = True

# 与 evaluation.py 中 CATEGORY_FILES 一致，用于按强度统计检测率
CATEGORY_FILES = [
    ("TS",   "Index2.npy"),  # Dist1
    ("Cat1", "Index3.npy"),  # Dist2
    ("Cat2", "Index4.npy"),  # Dist3
    ("Cat3", "Index5.npy"),  # Dist4
    ("Cat4", "Index6.npy"),  # Dist5
    ("Cat5", "Index7.npy"),  # Dist6
]


def px2km(x: float) -> float:
    return x * KM_PER_PIXEL


def test(config):
    # ----------------- 1. 基本测试配置 -----------------
    config['num_threads'] = 1              # only <num_threads = 1> supported when testing
    config['flip'] = False                 # not allowed to flip image
    config['status'] = 'test'
    config['crop_scale'] = 1.0

    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)

    # ----------------- 2. 计算 GFLOPs 和 Params -----------------
    print("Calculating model parameters and GFLOPs...")
    img_size = int(config['img_size'])
    input_channels = 6  # 你当前网络输入通道数（若有变化请修改）

    # 取出真正的网络（DataParallel 包一层时需要 .module）
    netG = getattr(model, 'netG', None)
    if netG is None:
        raise AttributeError("model 中未找到属性 netG，用于 GFLOPs 统计。")

    if hasattr(netG, 'module'):
        net_to_profile = copy.deepcopy(netG.module)
    else:
        net_to_profile = copy.deepcopy(netG)

    net_to_profile.cpu()
    dummy_input_cpu = torch.randn(1, input_channels, img_size, img_size)
    net_to_profile.eval()

    flops, params = profile(net_to_profile, inputs=(dummy_input_cpu,), verbose=False)

    params_m = params / 1e6
    gflops = flops / 1e9

    print(f"Model Parameters: {params_m:.2f} M")
    print(f"Model GFLOPs: {gflops:.2f} G")

    # ----------------- 3. 准备结果保存路径 -----------------
    result_root_path = os.path.join(config['checkpoints_dir'], config['name'], 'evaluation')
    util.mkdir(result_root_path)
    util.mkdir(os.path.join(result_root_path, 'prediction_distance'))
    util.mkdir(os.path.join(result_root_path, 'prediction_heatmap'))
    print("Create evaluation folder:", result_root_path)

    # ----------------- 4. 模型设为 eval 模式 -----------------
    model.eval()

    # ----------------- 5. 预测结果数组 (regression) -----------------
    # 约定：save_npy[i] = [-1, -1] 表示第 i 张样本“未检测到”
    num_samples = len(dataset)
    save_npy = np.full((num_samples + 1, 2), -1.0, dtype=np.float64)
    save_npy[0] = [-1.0, -1.0]

    # ----------------- 6. 检测统计量 -----------------
    detected_count = 0       # 台风集: TP；非台风集: FP
    missed_count = 0         # 台风集: FN；非台风集: TN
    total_count = num_samples

    # 每个样本的置信度和索引（用于后续 ROC / PR 等分析）
    all_indices = []
    all_confidences = []

    # 仅对台风测试集统计各强度类别召回率
    category_sets = {}
    category_stats = {}

    if IS_TYPHOON_SET:
        for cat_name, filename in CATEGORY_FILES:
            path = os.path.join(evaluation.data_dir, filename)
            if os.path.exists(path):
                idxs = np.load(path).astype(int)
                idx_set = set(int(i) for i in idxs.tolist())
                category_sets[cat_name] = idx_set
                category_stats[cat_name] = {
                    "detected": 0,
                    "total": len(idx_set),
                }
            else:
                category_sets[cat_name] = set()
                category_stats[cat_name] = {
                    "detected": 0,
                    "total": 0,
                }

    # ----------------- 7. 遍历测试集，执行前向推理 -----------------
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        # 置信度（来自你网络的第 4 个输出）
        confidence = float(model.test_result[3][1])
        # 样本在 TEST_LABEL / IndexX.npy 中的索引
        index = int(data["PATH"].cpu().data.numpy()[0])

        all_indices.append(index)
        all_confidences.append(confidence)

        if confidence >= CONF_THRESHOLD:
            # 预测为“有台风”
            detected_count += 1

            if IS_TYPHOON_SET:
                # 记录各强度类别的命中次数
                for cat_name, idx_set in category_sets.items():
                    if index in idx_set:
                        category_stats[cat_name]["detected"] += 1

                # 仅在检测成功时，才写入回归结果 & 保存可视化
                datapoints = (model.test_result[0][1]).cpu().data.numpy()
                # datapoints: shape (1, 2) with normalized coords
                save_npy[index][0] = float(datapoints[0][0])
                save_npy[index][1] = float(datapoints[0][1])

                # 保存距离图 / 热力图
                dist_img = model.test_result[1][1]
                dist_path = os.path.join(result_root_path, 'prediction_distance', f"{index}.png")
                util.save_image(util.tensor2im(dist_img), dist_path)

                heatmap_img = model.test_result[2][1]
                heatmap_path = os.path.join(result_root_path, 'prediction_heatmap', f"{index}.png")
                util.save_image(util.tensor2im(heatmap_img), heatmap_path)

            print(f"Evaluate forward-- {i+1}/{total_count} -- DETECTED (Conf: {confidence:.4f})")
        else:
            # 预测为“无台风”
            missed_count += 1
            # save_npy[index] 保持为 [-1, -1]，表示未检测
            print(f"Evaluate forward-- {i+1}/{total_count} -- MISSED (Conf: {confidence:.4f})")

    # ----------------- 8. 保存检测结果 (conf + index) -----------------
    det_results = {
        "indices": np.array(all_indices, dtype=np.int32),
        "confidences": np.array(all_confidences, dtype=np.float32),
        "threshold": float(CONF_THRESHOLD),
        "is_typhoon_set": bool(IS_TYPHOON_SET),
    }
    np.save(os.path.join(result_root_path, "detection_results.npy"), det_results)

    # ----------------- 9. 打印检测报告 -----------------
    detection_accuracy = 100.0 * detected_count / max(1, total_count)

    print("\n--- Detection Report ---")
    print(f"Confidence Threshold: {CONF_THRESHOLD:.3f}")
    print(f"Total Test Samples:   {total_count}")
    print(f"Predicted Positive:  {detected_count}")
    print(f"Predicted Negative:  {missed_count}")

    if IS_TYPHOON_SET:
        recall = detected_count / max(1, total_count)
        print(f"[Typhoon set] Recall (TPR) = {recall:.4f}  ({detected_count}/{total_count})")

        print("\n--- Per-Category Detection Accuracy (Typhoon set) ---")
        for cat_name, stats in category_stats.items():
            if stats["total"] == 0:
                print(f"  {cat_name}: N/A (no samples)")
            else:
                cat_acc = 100.0 * stats["detected"] / stats["total"]
                print(f"  {cat_name}: {stats['detected']}/{stats['total']} ({cat_acc:.2f}%)")
    else:
        # 非台风集：detected_count = FP, missed_count = TN
        fp = detected_count
        tn = missed_count
        fpr = fp / max(1, fp + tn)
        tnr = tn / max(1, fp + tn)
        print(f"[Non-typhoon set] FPR = {fpr:.4f}  (FP={fp}, TN={tn})")
        print(f"[Non-typhoon set] TNR = {tnr:.4f}")

    # ----------------- 10. 定位误差评估（仅台风集） -----------------
    if IS_TYPHOON_SET:
        # 保存回归结果
        np.save(os.path.join(result_root_path, 'regression.npy'), save_npy)

        # 调用 evaluation.evaluate_detailed，只统计“检测成功样本”的误差
        overall_px, d1_px, d2_px, d3_px, d4_px, d5_px, d6_px = evaluation.evaluate_detailed(save_npy)

        print("\n--- Localization Report (for DETECTED typhoon samples) ---")
        print(f"Overall MAE: {overall_px:.4f} px ({px2km(overall_px):.2f} km)")
        print(f"  Dist1 (TS):   {d1_px:.4f} px ({px2km(d1_px):.2f} km)")
        print(f"  Dist2 (Cat1): {d2_px:.4f} px ({px2km(d2_px):.2f} km)")
        print(f"  Dist3 (Cat2): {d3_px:.4f} px ({px2km(d3_px):.2f} km)")
        print(f"  Dist4 (Cat3): {d4_px:.4f} px ({px2km(d4_px):.2f} km)")
        print(f"  Dist5 (Cat4): {d5_px:.4f} px ({px2km(d5_px):.2f} km)")
        print(f"  Dist6 (Cat5): {d6_px:.4f} px ({px2km(d6_px):.2f} km)")
    else:
        overall_px = d1_px = d2_px = d3_px = d4_px = d5_px = d6_px = 0.0
        print("\nNon-typhoon set: skip localization evaluation.")

    # ----------------- 11. 将关键结果写入文本文件 -----------------
    eval_file_path = os.path.join(result_root_path, 'evaluation_summary.txt')
    with open(eval_file_path, 'w') as f:
        f.write("--- Model Stats ---\n")
        f.write(f"Model: {config['name']}\n")
        f.write(f"Checkpoint: {config.get('test_epoch', 'latest')}_net_G.pth\n")
        f.write(f"Parameters (M): {params_m:.2f}\n")
        f.write(f"GFLOPs (Input: 1x{input_channels}x{img_size}x{img_size}): {gflops:.2f}\n\n")

        f.write("--- Detection Report ---\n")
        f.write(f"Confidence Threshold: {CONF_THRESHOLD:.3f}\n")
        f.write(f"Total Test Samples:   {total_count}\n")
        f.write(f"Predicted Positive:  {detected_count}\n")
        f.write(f"Predicted Negative:  {missed_count}\n\n")

        if IS_TYPHOON_SET:
            f.write("[Typhoon set]\n")
            f.write(f"Recall (TPR): {detection_accuracy:.2f}% ({detected_count}/{total_count})\n\n")
            f.write("Per-Category Detection Accuracy:\n")
            for cat_name, stats in category_stats.items():
                if stats['total'] == 0:
                    f.write(f"  {cat_name}: N/A (no samples)\n")
                else:
                    cat_acc = 100.0 * stats['detected'] / stats['total']
                    f.write(f"  {cat_name}: {stats['detected']}/{stats['total']} ({cat_acc:.2f}%)\n")
            f.write("\n")

            f.write("--- Localization Report (for DETECTED typhoon samples) ---\n")
            f.write(f"Overall MAE: {overall_px:.4f} px ({px2km(overall_px):.2f} km)\n")
            f.write(f"Dist1 (TS):   {d1_px:.4f} px ({px2km(d1_px):.2f} km)\n")
            f.write(f"Dist2 (Cat1): {d2_px:.4f} px ({px2km(d2_px):.2f} km)\n")
            f.write(f"Dist3 (Cat2): {d3_px:.4f} px ({px2km(d3_px):.2f} km)\n")
            f.write(f"Dist4 (Cat3): {d4_px:.4f} px ({px2km(d4_px):.2f} km)\n")
            f.write(f"Dist5 (Cat4): {d5_px:.4f} px ({px2km(d5_px):.2f} km)\n")
            f.write(f"Dist6 (Cat5): {d6_px:.4f} px ({px2km(d6_px):.2f} km)\n")
        else:
            f.write("[Non-typhoon set]\n")
            fp = detected_count
            tn = missed_count
            fpr = fp / max(1, fp + tn)
            tnr = tn / max(1, fp + tn)
            f.write(f"FPR = {fpr:.4f} (FP={fp}, TN={tn})\n")
            f.write(f"TNR = {tnr:.4f}\n")

    print(f"\nModel stats and evaluation results saved to {eval_file_path}")


if __name__ == '__main__':
    configs = startup.SetupConfigs(config_path='configs/TCLNET_STAGE1.yaml')
    configs = configs.setup()
    # 可在此处根据需要修改 configs，例如：
    # configs['test_epoch'] = 20
    test(configs)


# from datasets import create_dataset
# from modules import create_model
# from utils import startup
# import os
# import utils.tools as util
# import numpy as np
# import evaluation
# import cv2
#
# import torch
# import copy # 用于 thop 修复
# from thop import profile
#
# KM_PER_PIXEL = 4.0
#
# def px2km(x):
#     return x * KM_PER_PIXEL
#
# def test(config):
#     config['num_threads'] = 1                     # only <num_threads = 1> supported when testing_usr
#     config['flip'] = False                        # not allowed to flip image
#     config['status'] = 'test'
#     config['crop_scale'] = 1.0
#
#     dataset = create_dataset(config)
#     model = create_model(config)
#     model.setup(config)
#
#     # --- 2. 计算 GFLOPs 和 Params (最终修复版) ---
#     print("Calculating model parameters and GFLOPs...")
#     device = model.device
#     img_size = int(config['img_size'])
#     input_channels = 6
#
#     net_to_profile = copy.deepcopy(model.netG.module)
#     net_to_profile.cpu()
#     dummy_input_cpu = torch.randn(1, input_channels, img_size, img_size)
#     net_to_profile.eval()
#     flops, params = profile(net_to_profile, inputs=(dummy_input_cpu, ), verbose=False)
#
#     params_m = params / 1e6
#     gflops = flops / 1e9
#
#     print(f"Model Parameters: {params_m:.2f} M")
#     print(f"Model GFLOPs: {gflops:.2f} G")
#     # ---------------------------------------------
#
#     result_root_path = os.path.join(config['checkpoints_dir'], config['name'], 'evaluation')
#     util.mkdir(result_root_path)
#     util.mkdir(os.path.join(result_root_path,'prediction_distance'))
#     util.mkdir(os.path.join(result_root_path,'prediction_heatmap'))
#     print(" create evaluate folder: " + result_root_path)
#
#     # set module to testing_usr mode
#     model.eval()
#
#     # 用 -1 填满，约定：[-1, -1] 代表“未检测到”
#     save_npy = np.full((len(dataset) + 1, 2), -1.0, dtype=np.float64)
#     save_npy[0] = [-1.0, -1.0]
#
#     # --- 3. 新增：检测逻辑和统计 ---
#     CONF_THRESHOLD = 0.4  # Updated threshold for typhoon detection
#     is_typhoon_set = True  # 跑台风测试集时 True；跑非台风集时改成 False
#     detected_count = 0
#     missed_count = 0
#     total_count = len(dataset)
#
#     # 保存每个样本的置信度 & 索引（后面做 ROC / PR 用）
#     all_indices = []
#     all_confidences = []
#
#     # --- Load category indices for per-class detection accuracy ---
#     category_sets = {}
#     category_stats = {}
#     category_files = [
#         ("TS", "Index2.npy"),
#         ("Cat1", "Index3.npy"),
#         ("Cat2", "Index4.npy"),
#         ("Cat3", "Index5.npy"),
#         ("Cat4", "Index6.npy"),
#         ("Cat5", "Index7.npy"),
#     ]
#
#     for cat_name, filename in category_files:
#         filepath = os.path.join(evaluation.data_dir, filename)
#         if os.path.exists(filepath):
#             indices = np.load(filepath)
#             category_sets[cat_name] = set(int(idx) for idx in indices.tolist())
#             category_stats[cat_name] = {"detected": 0, "total": len(indices)}
#         else:
#             category_sets[cat_name] = set()
#             category_stats[cat_name] = {"detected": 0, "total": 0}
#     # ---------------------------------
#
#     for i, data in enumerate(dataset):
#         model.set_input(data)  # push test datasets to module
#         model.test()  # forward module
#
#         # --- 4. 新增：集成检测逻辑 ---
#         # 从 (File 11) 的 test_result 中获取我们添加的 'CONFIDENCE'
#         confidence = model.test_result[3][1]
#
#         index = int(data["PATH"].cpu().data.numpy()[0])
#         # 记录 sample-level 信息
#         all_indices.append(index)
#         all_confidences.append(confidence)
#
#         if confidence >= CONF_THRESHOLD:
#             # 检测成功 (True Positive)
#             detected_count += 1
#
#             # 记录类别检测结果
#             for cat_name, idx_set in category_sets.items():
#                 if index in idx_set:
#                     category_stats[cat_name]["detected"] += 1
#
#             # --- 仅在检测成功时，才执行原有的保存逻辑 ---
#             datapoints = (model.test_result[0][1]).cpu().data.numpy()
#             # 1126
#             save_npy[index][0],save_npy[index][1] = datapoints[0][0], datapoints[0][1]
#
#             dist_img = model.test_result[1][1]
#             util.save_image(util.tensor2im(dist_img), os.path.join(result_root_path,'prediction_distance', str(index) + ".png"))
#
#             heatmap_img = model.test_result[2][1]
#             util.save_image(util.tensor2im(heatmap_img),os.path.join(result_root_path, 'prediction_heatmap', str(index) + ".png"))
#
#             print(f"Evaluate forward-- {i+1}/{total_count} -- DETECTED (Conf: {confidence:.4f})")
#         else:
#             # 检测失败 (False Negative)
#             missed_count += 1
#             print(f"Evaluate forward-- {i+1}/{total_count} -- MISSED (Conf: {confidence:.4f})")
#         # ---------------------------------
#
#     np.save(os.path.join(result_root_path,'regression.npy'),save_npy)
#
#     # === 保存检测结果（用于后处理） ===
#     det_results = {
#         "indices": np.array(all_indices, dtype=np.int32),
#         "confidences": np.array(all_confidences, dtype=np.float32),
#         "threshold": float(CONF_THRESHOLD),
#         "is_typhoon_set": bool(is_typhoon_set),
#     }
#     np.save(os.path.join(result_root_path, 'detection_results.npy'), det_results)
#
#     # --- 5. 修复：解包 7 个返回值 ---
#     l2_dist, dist1, dist2, dist3, dist4, dist5, dist6 = evaluation.evaluate_detailed(save_npy)
#     print("Testing npy result have been saved!")
#
#     # --- 6. 新增：在终端打印检测报告 ---
#     detection_accuracy = (detected_count / total_count) * 100
#     print("\n--- Detection Report (on Positive Test Set) ---")
#     print(f"Confidence Threshold: {CONF_THRESHOLD}")
#     print(f"Total Test Samples:   {total_count}")
#     print(f"Detected (True Pos):  {detected_count}")
#     if is_typhoon_set:
#         # 对台风测试集：TP / FN
#         recall = detected_count / max(1, total_count)
#         print(f"[Typhoon set] Recall (TPR) = {recall:.4f} "
#               f"({detected_count}/{total_count})")
#
#         print("\nPer-intensity recall:")
#         for cat_name, stats in category_stats.items():
#             if stats["total"] > 0:
#                 r = stats["detected"] / stats["total"]
#                 print(f"  {cat_name}: {stats['detected']}/{stats['total']} = {r:.4f}")
#             else:
#                 print(f"  {cat_name}: no samples")
#     else:
#         # 对非台风测试集：FP / TN
#         fp = detected_count
#         tn = missed_count
#         fpr = fp / max(1, fp + tn)
#         tnr = tn / max(1, fp + tn)
#         print(f"[Non-typhoon set] FPR = {fpr:.4f}  (FP={fp}, TN={tn})")
#         print(f"[Non-typhoon set] TNR = {tnr:.4f}")
#     if is_typhoon_set:
#         np.save(os.path.join(result_root_path, 'regression.npy'), save_npy)
#         l2_dist, dist1, dist2, dist3, dist4, dist5, dist6 = evaluation.evaluate_detailed(save_npy)
#         print("Testing npy result have been saved! Evaluation distance: "
#               f"{l2_dist:.4f}  {dist1:.4f}  {dist2:.4f}  {dist3:.4f}  "
#               f"{dist4:.4f}  {dist5:.4f}  {dist6:.4f}")
#     else:
#         print("Non-typhoon set: skip localization evaluation.")
#
#
# if __name__ == '__main__':
#     configs = startup.SetupConfigs(config_path='configs/TCLNET_STAGE1.yaml')
#     configs = configs.setup()
#
#
#     configs['status'] = 'test'
#
#
#     test(configs)
#
