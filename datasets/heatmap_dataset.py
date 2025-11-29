import os.path
from datasets.cdataset import cdataset
import cv2

import random
import numpy as np
import torch
from torchvision import transforms


def normalization(data, norm_type="standard"):
    if norm_type == "standard":
        mean = [0.5]
        std = [0.5]
    elif norm_type == 'imagenet':
        mean = [0.485]  # 可以根据实际情况调整
        std = [0.229]   # 可以根据实际情况调整
    else:
        raise NotImplementedError("norm_type mismatched!")

    transform = transforms.Normalize(mean, std, inplace=False)
    return transform(data)


class heatmapDataset(cdataset):
    def __init__(self, config):
        cdataset.__init__(self, config)
        self.input_paths = [os.path.join(config['dataroot'], f"{i}/",str.upper(config['status']) + "_INPUT")
                            for i in range(10, 16)]
        self.heatmap_paths = os.path.join(config['dataroot'], "10/",str.upper(config['status'])+"_HEATMAP")
        self.labels = np.load(os.path.join(config['dataroot'],"10/",str.upper(config['status'])+"_LABEL"+".npy"))
        self.dataset_len = len(os.listdir(self.input_paths[0]))
        self.config = config

    # def __getitem__(self, index):
    #     # 1. 图像加载
    #     image_input_show = cv2.imread(os.path.join(self.input_paths[5], str(index + 1) + ".png")).astype(np.float32)
    #     image_inputs_list = [cv2.imread(os.path.join(path, str(index + 1) + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #                    for path in self.input_paths]
        
    #     # 2. 原始热图加载 (类型: numpy array)
    #     image_heatmap_np = cv2.imread(os.path.join(self.heatmap_paths,str(index+1)+".png"),cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
    #     # --- 3. 【核心修复】: 不再使用 kpoint，而是从热图中反向查找中心 ---
    #     # 查找热图中的最大值及其坐标
    #     # maxLoc 返回的是 (x, y) 坐标，即 (列, 行)
    #     (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image_heatmap_np)
        
    #     # 这就是我们需要的、与热图100%对齐的中心点！
    #     center_x = maxLoc[0]
    #     center_y = maxLoc[1]
    #     # --- 修复结束 ---

    #     # 4. 图像预处理
    #     image_input_list_norm = [normalization(torch.from_numpy(img).unsqueeze(0).float() / 255,str(self.config['norm_type'])) for img in image_inputs_list]
    #     image_heatmap_tensor = torch.from_numpy(image_heatmap_np).unsqueeze(0).float() / 255.0
    #     image_input_show_tensor = torch.from_numpy(image_input_show).permute(2, 0, 1).float() / 255.

    #     # 合并12个通道
    #     image_input_tensor = torch.cat(image_input_list_norm, dim=0) 

    #     # 5. 保留原始坐标 (用于评估)
    #     # 现在我们使用从热图中新找到的坐标
    #     label_coord_tensor = torch.Tensor([center_x, center_y]).float()

    #     # 6. [新增] 自动生成 BBox 标签 (用于检测头)
    #     BOX_SIZE = 128.0  # 128x128 的框大小
    #     IMG_SIZE = 224.0
        
    #     half_box = BOX_SIZE / 2.0
        
    #     # 使用我们新找到的 center_x, center_y
    #     x1 = max(0.0, center_x - half_box)
    #     y1 = max(0.0, center_y - half_box)
    #     x2 = min(IMG_SIZE, center_x + half_box) 
    #     y2 = min(IMG_SIZE, center_y + half_box)

    #     target_bbox = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
    #     target_labels = torch.tensor([0], dtype=torch.int64) # 0 = "台风" 类别
        
    #     target_detection = {
    #         "boxes": target_bbox,
    #         "labels": target_labels
    #     }
        
    #     # 7. [修改] 返回新的多任务字典
    #     return {'IMAGE': image_input_tensor,        # [12, 224, 224] Tensor
    #             'IMAGE_SHOW': image_input_show_tensor, # [3, 224, 224] Tensor
    #             'HEATMAP_TARGET': image_heatmap_tensor, # [1, 224, 224] Tensor
    #             'DETECTION_TARGET': target_detection,   # 字典 {'boxes': ..., 'labels': ...}
    #             "LABEL": label_coord_tensor,      # [2] Tensor
    #             "PATH": index+1
    #            }


    # def __getitem__(self, index):
    #     image_input_show = cv2.imread(os.path.join(self.input_paths[5], str(index + 1) + ".png")).astype(np.float32)
    #     print("img")
    #     image_inputs = [cv2.imread(os.path.join(path, str(index + 1) + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #                    for path in self.input_paths]
    #     # print(image_inputs[0])
    #     image_heatmap = cv2.imread(os.path.join(self.heatmap_paths,str(index+1)+".png"),cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #     kpoint = self.labels[index+1]

    #     label = torch.Tensor([kpoint[0],kpoint[1]]).float()

    #     image_input = [normalization(torch.from_numpy(img).unsqueeze(0).float() / 255,str(self.config['norm_type'])) for img in image_inputs]
    #     image_heatmap = torch.from_numpy(image_heatmap).unsqueeze(0).float()
    #     image_input_show = torch.from_numpy(image_input_show).permute(2, 0, 1).float() / 255.
    #     print("tensor")
    #     image_heatmap = image_heatmap / 255.

        

        
    #     # 旧代码
    #     # image_input = torch.cat(image_input,dim=0)
    #     # # print(image_input[0])
    #     # label = torch.Tensor([kpoint[0],kpoint[1]]).float()

    #     # return {'IMAGE': image_input,
    #     #         'IMAGE_SHOW':image_input_show,
    #     #         'HEATMAP':image_heatmap,
    #     #         "LABEL": label,
    #     #         "PATH": index+1
    #     #         }

    def __getitem__(self, index):
        image_input_show = cv2.imread(os.path.join(self.input_paths[5], str(index + 1) + ".png")).astype(np.float32)
        # print("img")
        image_inputs = [cv2.imread(os.path.join(path, str(index + 1) + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                       for path in self.input_paths]
        # print(image_inputs[0])
        image_heatmap = cv2.imread(os.path.join(self.heatmap_paths,str(index+1)+".png"),cv2.IMREAD_GRAYSCALE).astype(np.float32)
        kpoint = self.labels[index+1]

        image_input = [normalization(torch.from_numpy(img).unsqueeze(0).float() / 255,str(self.config['norm_type'])) for img in image_inputs]
        image_heatmap = torch.from_numpy(image_heatmap).unsqueeze(0).float()
        image_input_show = torch.from_numpy(image_input_show).permute(2, 0, 1).float() / 255.
        # print("tensor")
        image_heatmap = image_heatmap / 255.
        image_input = torch.cat(image_input,dim=0)
        # print(image_input[0])
        label = torch.Tensor([kpoint[0],kpoint[1]]).float()

        return {'IMAGE': image_input,
                'IMAGE_SHOW':image_input_show,
                'HEATMAP':image_heatmap,
                "LABEL": label,
                "PATH": index+1
                }


    def __len__(self):
        return self.dataset_len
