import os
import numpy as np

# 路径说明：
# 请把 data_dir 改成包含 TEST_LABEL.npy、Index2.npy~Index7.npy 的目录。
# 例如：r"D:\Improved-TCLNet\Improved-TCLNet\Improved-TCLNet\datasets\data\TCLDwithTSsame\13"
data_dir = r"D:\Improved-TCLNet\Improved-TCLNet\Improved-TCLNet\datasets\data\TCLDwithTSsame\13"


def _load_npy(name: str) -> np.ndarray:
    """从 data_dir 加载 npy 文件。"""
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


# 各强度类别对应的 index 文件（与论文中的 TS / Cat1–Cat5 对应）
CATEGORY_FILES = [
    ("TS",   "Index2.npy"),  # Dist1
    ("Cat1", "Index3.npy"),  # Dist2
    ("Cat2", "Index4.npy"),  # Dist3
    ("Cat3", "Index5.npy"),  # Dist4
    ("Cat4", "Index6.npy"),  # Dist5
    ("Cat5", "Index7.npy"),  # Dist6
]


def calculate_l2_distance(predictions: np.ndarray) -> float:
    """
    兼容旧接口：返回所有“检测成功样本”的平均像素误差（欧氏距离）。
    本质上就是 evaluate_detailed(predictions)[0]。
    """
    mean_overall_px, *_ = evaluate_detailed(predictions)
    return mean_overall_px


def evaluate_detailed(predictions: np.ndarray):
    """
    计算整体和各强度类别的平均像素距离（欧氏距离）。

    参数
    ----
    predictions : np.ndarray
        形状 (N+1, 2)，第 0 行不用。
        约定：若某个样本未检测到，则其坐标为 [-1, -1]（由 test.py 初始化）。

    返回
    ----
    mean_overall_px, dist1_px, dist2_px, dist3_px, dist4_px, dist5_px, dist6_px
        全部单位为“像素”，由 test.py 再换算为 km。
    """
    predictions = predictions.astype(np.float64)
    num_pred = predictions.shape[0]

    if num_pred <= 1:
        raise ValueError("predictions 至少需要包含 1 行有效数据（含第 0 行 dummy）。")

    # 有效样本：x,y >= 0 视为“检测成功”
    valid_mask = (predictions[:, 0] >= 0.0) & (predictions[:, 1] >= 0.0)
    valid_mask[0] = False  # 第 0 行是 dummy

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError("没有检测成功的样本（all predictions are invalid, e.g., [-1, -1]）。")

    # 加载 GT：归一化坐标 (0~1)，长度应 >= num_pred-1
    targets = _load_npy("TEST_LABEL.npy").astype(np.float64)

    if targets.shape[0] < num_pred:
        # 安全截断到最短长度
        num_effective = targets.shape[0]
        predictions = predictions[:num_effective]
        valid_mask = valid_mask[:num_effective]
        valid_indices = np.where(valid_mask)[0]
    else:
        num_effective = num_pred

    # 将归一化坐标转换为像素坐标（224x224 图）
    preds_px = predictions[:num_effective] * 224.0
    targets_px = targets[:num_effective] * 224.0

    # 每个样本的像素距离
    diffs = preds_px - targets_px
    dists_px = np.sqrt(np.sum(diffs ** 2, axis=1))

    # ---- 整体平均：只对检测成功样本求均值 ----
    mean_overall_px = float(np.mean(dists_px[valid_indices]))

    # ---- 各强度类别：只对“该类 ∩ 检测成功”的样本求均值 ----
    dist_list_px = []

    for cat_name, filename in CATEGORY_FILES:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            # 若该类没有 index 文件，则记为 0（或你可以改成 np.nan）
            dist_list_px.append(0.0)
            continue

        idxs = _load_npy(filename).astype(int)
        # 保证 index 在有效范围内，并且对应样本已检测成功
        idxs_valid = [idx for idx in idxs if 0 <= idx < num_effective and valid_mask[idx]]
        if len(idxs_valid) == 0:
            dist_list_px.append(0.0)
            continue

        mean_px = float(np.mean(dists_px[idxs_valid]))
        dist_list_px.append(mean_px)

    # 若某些类别缺失，补齐到 6 个
    while len(dist_list_px) < 6:
        dist_list_px.append(0.0)

    dist1_px, dist2_px, dist3_px, dist4_px, dist5_px, dist6_px = dist_list_px[:6]

    return mean_overall_px, dist1_px, dist2_px, dist3_px, dist4_px, dist5_px, dist6_px



# import numpy as np
# import os
# data_dir = r'D:\Improved-TCLNet\Improved-TCLNet\Improved-TCLNet\datasets\data\TCLDwithTSsame\13'
#
# print("data_dir =", data_dir)
# print("dir exists:", os.path.exists(data_dir))
# print("label exists:", os.path.exists(os.path.join(data_dir, 'TEST_LABEL.npy')))
#
# def _load_npy(name: str):
#     """辅助函数：自动拼路径并打印出来，方便你确认。"""
#     path = os.path.join(data_dir, name)
#     print("Loading:", path)
#     return np.load(path)
#
# def calculate_l2_distance(predictions):
#     # targets = np.load(data_dir + 'TEST_LABEL.npy').astype(np.float64)
#     targets = _load_npy('TEST_LABEL.npy').astype(np.float64)
#     predictions = predictions.astype(np.float64)
#     targets = targets[1:,:]
#     predictions = predictions[1:,:]
#     targets = targets * 224.
#     predictions = predictions * 224.
#
#     dist = np.power((predictions-targets),2)
#     dist = np.sum(dist,axis=1)
#     dist = np.sqrt(dist)
#     dist = np.mean(dist)
#
#     return dist
#
# def evaluate_detailed(predictions):
#     predictions = predictions * 224.
#     if data_dir == r'D:\Improved-TCLNet\Improved-TCLNet\Improved-TCLNet\datasets\data\TCLDwithTSsame\13':
#
#         targets = _load_npy('TEST_LABEL.npy').astype(np.float64) * 224.
#
#         points1 = _load_npy('Index2.npy')
#         points2 = _load_npy('Index3.npy')
#         points3 = _load_npy('Index4.npy')
#         points4 = _load_npy('Index5.npy')
#         points5 = _load_npy('Index6.npy')
#         points6 = _load_npy('Index7.npy')
#
#         mean1 = 0.0
#         mean2 = 0.0
#         mean3 = 0.0
#         mean4 = 0.0
#         mean5 = 0.0
#         mean6 = 0.0
#         mean_overall = 0.0
#
#         for i in range(len(points1)):
#             mean_easypont = np.power((targets[points1[i]] - predictions[points1[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean1 = mean1 + mean_easypont
#         for i in range(len(points2)):
#             mean_easypont = np.power((targets[points2[i]] - predictions[points2[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean2 = mean2 + mean_easypont
#         for i in range(len(points3)):
#             mean_easypont = np.power((targets[points3[i]] - predictions[points3[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean3 = mean3 + mean_easypont
#         for i in range(len(points4)):
#             mean_easypont = np.power((targets[points4[i]] - predictions[points4[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean4 = mean4 + mean_easypont
#         for i in range(len(points5)):
#             mean_easypont = np.power((targets[points5[i]] - predictions[points5[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean5 = mean5 + mean_easypont
#         for i in range(len(points6)):
#             mean_hardpoint = np.power((targets[points6[i]] - predictions[points6[i]]), 2)
#             mean_hardpoint = np.sqrt(np.sum(mean_hardpoint))
#             mean6 = mean6 + mean_hardpoint
#
#         print(targets[0].shape,predictions[0].shape)
#         for k in range(3200):
#             # print(targets[k+1],predictions[k+1],targets[k+1] - predictions[k+1])
#             mean_overpoint = np.power((targets[k+1] - predictions[k+1]), 2)
#             mean_overpoint = np.sqrt(np.sum(mean_overpoint))
#             mean_overall = mean_overall + mean_overpoint
#
#         mean1 = mean1 / len(points1)
#         mean2 = mean2 / len(points2)
#         mean3 = mean3 / len(points3)
#         mean4 = mean4 / len(points4)
#         mean5 = mean5 / len(points5)
#         mean6 = mean6 / len(points6)
#         mean_overall = mean_overall / (len(predictions)-1)
#
#         return mean_overall,mean1,mean2,mean3,mean4,mean5,mean6
#     else:
#
#         targets = _load_npy('TEST_LABEL.npy').astype(np.float64) * 224.
#
#         # points1 = _load_npy('Index2.npy')
#         points2 = _load_npy('Index3.npy')
#         points3 = _load_npy('Index4.npy')
#         points4 = _load_npy('Index5.npy')
#         points5 = _load_npy('Index6.npy')
#         points6 = _load_npy('Index7.npy')
#
#         mean1 = 0.0
#         mean2 = 0.0
#         mean3 = 0.0
#         mean4 = 0.0
#         mean5 = 0.0
#         mean6 = 0.0
#         mean_overall = 0.0
#
#         # for i in range(len(points1)):
#         #     mean_easypont = np.power((targets[points1[i]] - predictions[points1[i]]), 2)
#         #     mean_easypont = np.sqrt(np.sum(mean_easypont))
#         #     mean1 = mean1 + mean_easypont
#         for i in range(len(points2)):
#             mean_easypont = np.power((targets[points2[i]] - predictions[points2[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean2 = mean2 + mean_easypont
#         for i in range(len(points3)):
#             mean_easypont = np.power((targets[points3[i]] - predictions[points3[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean3 = mean3 + mean_easypont
#         for i in range(len(points4)):
#             mean_easypont = np.power((targets[points4[i]] - predictions[points4[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean4 = mean4 + mean_easypont
#         for i in range(len(points5)):
#             mean_easypont = np.power((targets[points5[i]] - predictions[points5[i]]), 2)
#             mean_easypont = np.sqrt(np.sum(mean_easypont))
#             mean5 = mean5 + mean_easypont
#         for i in range(len(points6)):
#             mean_hardpoint = np.power((targets[points6[i]] - predictions[points6[i]]), 2)
#             mean_hardpoint = np.sqrt(np.sum(mean_hardpoint))
#             mean6 = mean6 + mean_hardpoint
#
#         print(targets.shape, predictions.shape)
#         for k in range(len(predictions) - 1):
#             mean_overpoint = np.power((targets[k + 1] - predictions[k + 1]), 2)
#             mean_overpoint = np.sqrt(np.sum(mean_overpoint))
#             mean_overall = mean_overall + mean_overpoint
#
#         # mean1 = mean1 / len(points1)
#         mean2 = mean2 / len(points2)
#         mean3 = mean3 / len(points3)
#         mean4 = mean4 / len(points4)
#         mean5 = mean5 / len(points5)
#         mean6 = mean6 / len(points6)
#         mean_overall = mean_overall / (len(predictions) - 1)
#
#         return mean_overall, mean1, mean2, mean3, mean4, mean5, mean6