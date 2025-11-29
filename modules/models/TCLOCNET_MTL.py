# 文件: Improved-TCLNet - 副本/modules/models/TCLOCNET_MTL.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入我们刚创建的模型架构
from modules.components.TCLOCNET_MTL_COMPONENT import TCLocNet_MTL
import math

# --- 损失函数帮助器 ---

def bce_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """BCE Focal Loss (用于分类)"""
    inputs = inputs.sigmoid()
    pt = torch.where(targets == 1, inputs, 1 - inputs)
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == 'sum':
        return focal_loss.sum()
    return focal_loss.mean()

def diou_loss(preds, targets, eps=1e-7):
    """DIoU Loss (用于BBox回归)"""
    # preds, targets: [N, 4] (x1, y1, x2, y2)
    
    # IoU
    ix1 = torch.max(preds[:, 0], targets[:, 0])
    iy1 = torch.max(preds[:, 1], targets[:, 1])
    ix2 = torch.min(preds[:, 2], targets[:, 2])
    iy2 = torch.min(preds[:, 3], targets[:, 3])
    iw = torch.clamp(ix2 - ix1, min=0.)
    ih = torch.clamp(iy2 - iy1, min=0.)
    i_area = iw * ih
    
    pa = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    ta = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    u_area = pa + ta - i_area + eps
    iou = i_area / u_area
    
    # DIoU 惩罚项
    cx_p = (preds[:, 0] + preds[:, 2]) / 2
    cy_p = (preds[:, 1] + preds[:, 3]) / 2
    cx_t = (targets[:, 0] + targets[:, 2]) / 2
    cy_t = (targets[:, 1] + targets[:, 3]) / 2
    
    dist_sq = (cx_p - cx_t)**2 + (cy_p - cy_t)**2
    
    bx1 = torch.min(preds[:, 0], targets[:, 0])
    by1 = torch.min(preds[:, 1], targets[:, 1])
    bx2 = torch.max(preds[:, 2], targets[:, 2])
    by2 = torch.max(preds[:, 3], targets[:, 3])
    c_diag_sq = (bx2 - bx1)**2 + (by2 - by1)**2 + eps
    
    diou_penalty = dist_sq / c_diag_sq
    return (1 - iou + diou_penalty).sum()

# --- 主模型类 (用于 run.py) ---

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_classes = 1 # "typhoon"
        
        # 1. 构建网络
        self.network = TCLocNet_MTL(config)

        # 2. 定义多任务损失函数
        
        # 任务1: 热图损失
        # (TCLNet 使用 L1+L2，这里我们简化为 MSE，对峰值更敏感)
        self.heatmap_loss_func = nn.MSELoss(reduction='sum')
        
        # 任务2: 检测损失
        self.det_cls_loss_func = bce_focal_loss
        self.det_reg_loss_func = diou_loss
        self.det_ctr_loss_func = nn.BCEWithLogitsLoss(reduction='sum')
        
        # 3. 定义损失权重 (超参数，可调)
        self.w_heatmap = config['TRAIN'].get('W_HEATMAP', 1.0)
        self.w_cls = config['TRAIN'].get('W_CLS', 1.0)
        self.w_reg = config['TRAIN'].get('W_REG', 1.0)
        self.w_ctr = config['TRAIN'].get('W_CTR', 1.0)

    def _compute_detection_loss(self, det_preds, gt_targets, coords_batch):
        """计算 FCOS 检测损失"""
        cls_preds, reg_preds, ctr_preds = det_preds
        
        # [B, H*W, C] -> [N, C] (N = B*H*W)
        cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        reg_preds = reg_preds.permute(0, 2, 3, 1).reshape(-1, 4)
        ctr_preds = ctr_preds.permute(0, 2, 3, 1).reshape(-1, 1)
        
        B, C, H, W = det_preds[0].shape
        
        all_cls_targets = []
        all_reg_targets = []
        all_ctr_targets = []
        all_pos_mask = []

        for b_idx in range(B):
            # 获取当前图像的 GT
            gt_boxes = gt_targets[b_idx]['boxes']   # [M, 4]
            gt_labels = gt_targets[b_idx]['labels'] # [M]
            grid_xy = coords_batch[b_idx]         # [H*W, 2]
            
            if gt_boxes.shape[0] == 0:
                # --- 没有目标 (全负样本) ---
                N = grid_xy.shape[0]
                cls_target = torch.zeros(N, self.num_classes, device=cls_preds.device)
                reg_target = torch.zeros(N, 4, device=reg_preds.device)
                ctr_target = torch.zeros(N, 1, device=ctr_preds.device)
                pos_mask = torch.zeros(N, dtype=torch.bool, device=cls_preds.device)
            else:
                # --- 有目标 (FCOS 匹配) ---
                N = grid_xy.shape[0]
                M = gt_boxes.shape[0]
                
                # [N, M]
                x = grid_xy[:, 0].unsqueeze(1).expand(N, M)
                y = grid_xy[:, 1].unsqueeze(1).expand(N, M)
                
                gt_x1 = gt_boxes[:, 0].unsqueeze(0).expand(N, M)
                gt_y1 = gt_boxes[:, 1].unsqueeze(0).expand(N, M)
                gt_x2 = gt_boxes[:, 2].unsqueeze(0).expand(N, M)
                gt_y2 = gt_boxes[:, 3].unsqueeze(0).expand(N, M)

                # 计算 (l, t, r, b)
                l = x - gt_x1
                t = y - gt_y1
                r = gt_x2 - x
                b = gt_y2 - y
                
                ltrb = torch.stack([l, t, r, b], dim=-1) # [N, M, 4]
                
                # 正样本：落入GT Box内的点
                is_in_box = ltrb.min(dim=-1)[0] > 0 # [N, M]
                
                # (由于我们是单尺度，不限制范围)
                # ... (省略了FCOS复杂的多尺度分配策略) ...
                
                # --- 分配策略 ---
                # (由于台风只有一个，我们简化：所有在框内的点都负责预测)
                # (在多目标重叠时，这里需要更复杂的分配，但对台风任务足够了)
                
                gt_areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1) # [N, M]
                # 选择面积最小的GT (在多台风重叠时有用)
                min_area_idx = torch.argmin(gt_areas, dim=1) # [N]
                
                # [N, 4]
                reg_target_per_img = ltrb[torch.arange(N), min_area_idx]
                # [N]
                is_pos = is_in_box[torch.arange(N), min_area_idx]
                
                # 1. 分类目标
                cls_target = torch.zeros(N, self.num_classes, device=cls_preds.device)
                gt_label_per_img = gt_labels[min_area_idx] # [N]
                cls_target[is_pos, gt_label_per_img[is_pos]] = 1.0

                # 2. 回归目标
                reg_target = torch.zeros(N, 4, device=reg_preds.device)
                reg_target[is_pos] = reg_target_per_img[is_pos]
                
                # 3. 中心度目标
                ctr_target = torch.zeros(N, 1, device=ctr_preds.device)
                l, t, r, b = reg_target[is_pos].T
                ctr_val = torch.sqrt( (torch.min(l,r) / (torch.max(l,r)+1e-7)) *
                                      (torch.min(t,b) / (torch.max(t,b)+1e-7)) )
                ctr_target[is_pos, 0] = ctr_val
                pos_mask = is_pos # [N]

            all_cls_targets.append(cls_target)
            all_reg_targets.append(reg_target)
            all_ctr_targets.append(ctr_target)
            all_pos_mask.append(pos_mask)
            
        # 组合 Batch
        cls_targets = torch.cat(all_cls_targets, dim=0)
        reg_targets = torch.cat(all_reg_targets, dim=0)
        ctr_targets = torch.cat(all_ctr_targets, dim=0)
        pos_mask = torch.cat(all_pos_mask, dim=0)
        
        # --- 计算损失 ---
        num_pos = pos_mask.sum().item()
        if num_pos == 0: num_pos = 1.0 # 避免除0
            
        # 1. 分类损失 (Focal Loss)
        loss_cls = self.det_cls_loss_func(cls_preds, cls_targets) / num_pos
        
        # 2. 回归损失 (DIoU Loss) - 仅对正样本
        if pos_mask.sum() > 0:
            pos_coords = coords_batch.reshape(-1, 2)[pos_mask] # [num_pos, 2]
            pos_reg_preds = reg_preds[pos_mask] # [num_pos, 4] (l,t,r,b)
            pos_reg_targets = reg_targets[pos_mask] # [num_pos, 4] (l,t,r,b)

            # (l,t,r,b) -> (x1,y1,x2,y2)
            preds_x1 = pos_coords[:, 0] - pos_reg_preds[:, 0]
            preds_y1 = pos_coords[:, 1] - pos_reg_preds[:, 1]
            preds_x2 = pos_coords[:, 0] + pos_reg_preds[:, 2]
            preds_y2 = pos_coords[:, 1] + pos_reg_preds[:, 3]
            preds_boxes = torch.stack([preds_x1, preds_y1, preds_x2, preds_y2], dim=-1)
            
            targets_x1 = pos_coords[:, 0] - pos_reg_targets[:, 0]
            targets_y1 = pos_coords[:, 1] - pos_reg_targets[:, 1]
            targets_x2 = pos_coords[:, 0] + pos_reg_targets[:, 2]
            targets_y2 = pos_coords[:, 1] - pos_reg_targets[:, 3]
            targets_boxes = torch.stack([targets_x1, targets_y1, targets_x2, targets_y2], dim=-1)
            
            loss_reg = self.det_reg_loss_func(preds_boxes, targets_boxes) / num_pos
            
            # 3. 中心度损失
            loss_ctr = self.det_ctr_loss_func(ctr_preds[pos_mask], ctr_targets[pos_mask]) / num_pos
        else:
            loss_reg = torch.tensor(0.0, device=cls_preds.device)
            loss_ctr = torch.tensor(0.0, device=cls_preds.device)
            
        return loss_cls, loss_reg, loss_ctr


    def forward(self, data_batch):
        """
        data_batch: 从您修改后的 heatmap_dataset 返回的字典
        """
        images = data_batch['IMAGE']
        device = images.device
        
        # 1. 正向传播
        det_preds, pred_heatmap = self.network(images)
        
        if self.training:
            # --- 训练模式：计算总损失 ---
            
            # 1.1 获取热图标签
            gt_heatmap = data_batch['HEATMAP_TARGET'] # [B, 1, 224, 224]
            
            # 1.2 计算热图损失
            loss_heatmap = self.heatmap_loss_func(pred_heatmap, gt_heatmap)
            
            # 2.1 获取检测标签
            gt_detection = data_batch['DETECTION_TARGET'] # 字典列表
            # 将字典列表转为 batch 形式 (B x [M, 4] -> [B, M, 4])
            # (注意：我们的dataloader没有collate_fn，所以gt_detection是list)
            gt_targets_batch = []
            for b_idx in range(len(gt_detection['boxes'])):
                gt_targets_batch.append({
                    'boxes': gt_detection['boxes'][b_idx].to(device),
                    'labels': gt_detection['labels'][b_idx].to(device)
                })

            # 2.2 生成网格坐标 (用于FCOS匹配)
            B, C, H, W = det_preds[0].shape
            y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            coords = torch.stack([x_coords, y_coords], dim=-1).to(device) # [H, W, 2]
            coords_batch = coords.reshape(1, H*W, 2).repeat(B, 1, 1) # [B, H*W, 2]
            
            # 2.3 计算检测损失
            loss_cls, loss_reg, loss_ctr = self._compute_detection_loss(
                det_preds, gt_targets_batch, coords_batch
            )
            
            # 3. 加权总损失
            loss = (self.w_heatmap * loss_heatmap) + \
                   (self.w_cls * loss_cls) + \
                   (self.w_reg * loss_reg) + \
                   (self.w_ctr * loss_ctr)
                   
            return {
                "loss": loss,
                "l_heat": loss_heatmap,
                "l_cls": loss_cls,
                "l_reg": loss_reg,
                "l_ctr": loss_ctr
            }

        else:
            # --- 推理模式：解码 ---
            
            # 1. 解码热图 (已是 0-1)
            # pred_heatmap 已经是 [B, 1, 224, 224]
            
            # 2. 解码检测 (FCOS)
            cls_preds, reg_preds, ctr_preds = det_preds
            cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes).sigmoid()
            reg_preds = reg_preds.permute(0, 2, 3, 1).reshape(B, -1, 4) # (l,t,r,b)
            ctr_preds = ctr_preds.permute(0, 2, 3, 1).reshape(B, -1, 1).sigmoid()
            
            # (生成坐标网格)
            B, C, H, W = det_preds[0].shape
            y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            coords = torch.stack([x_coords, y_coords], dim=-1).to(device) # [H, W, 2]
            coords_batch = coords.reshape(1, H*W, 2).repeat(B, 1, 1) # [B, H*W, 2]

            # (l,t,r,b) -> (x1,y1,x2,y2)
            x1 = coords_batch[..., 0] - reg_preds[..., 0]
            y1 = coords_batch[..., 1] - reg_preds[..., 1]
            x2 = coords_batch[..., 0] + reg_preds[..., 2]
            y2 = coords_batch[..., 1] + reg_preds[..., 3]
            
            boxes = torch.stack([x1, y1, x2, y2], dim=-1) # [B, N, 4]
            
            # 得分 = 分类得分 * 中心度得分
            scores = (cls_preds * ctr_preds).squeeze(-1) # [B, N]
            
            # ... (此处应有 NMS) ...
            
            # 为了 test.py 简单，我们只返回得分最高的 k 个
            k = 10 
            top_scores, top_indices = torch.topk(scores, k, dim=1)
            
            detection_outputs = []
            for b_idx in range(B):
                detection_outputs.append({
                    'boxes': boxes[b_idx][top_indices[b_idx]],
                    'scores': top_scores[b_idx],
                    'labels': torch.zeros(k, device=device, dtype=torch.long) # 全是 0 (台风)
                })

            # 返回: (检测结果[字典列表], 热图[张量])
            return detection_outputs, pred_heatmap
