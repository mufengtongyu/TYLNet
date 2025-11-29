from modules.comdel import cmodel
import torch
from modules import backbone
from torch.nn import functional as F
import torchvision

class network(cmodel):
    def __init__(self, config):
        super().__init__(config)
        
        self.loss_names = ['LOC', 'CLS', 'BBOX'] # 定位、分类、BBox回归损失
        self.model_names = ["G"]
        
        # 使用 backbone 工厂创建新模型
        self.netG = backbone.create_backbone(
            net_name='ADVANCED_TCLNET',
            init_type=config['init_type'],
            init_gain=float(config['init_gain']),
            gpu_ids=config['gpu_ids']
        )

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=float(config['learning_rate']))
        self.optimizers.append(['G', self.optimizer_G])

        # 定义损失函数
        self.criterion_loc = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        # BBox 回归通常使用 Smooth L1 Loss
        self.criterion_bbox = torch.nn.SmoothL1Loss()
        
        # 损失权重
        self.lambda_loc = 1.0
        self.lambda_cls = 1.0
        self.lambda_bbox = 1.0

    def set_input(self, inputs):
        self.INPUT = inputs['IMAGE'].to(self.device)
        # 定位目标
        self.TARGET_HEATMAP_224 = inputs['HEATMAP_TARGET'].to(self.device)
        self.TARGET_HEATMAP_56 = F.max_pool2d(F.max_pool2d(self.TARGET_HEATMAP_224, 2), 2)
        # 检测目标
        self.TARGET_DETECTION = [{k: v.to(self.device) for k, v in t.items()} for t in inputs['DETECTION_TARGET']]

    def forward(self):
        self.outputs = self.netG.forward(self.INPUT)

    def backward_G(self):
        # 1. 计算定位损失
        self.LOSS_LOC = self.criterion_loc(self.outputs['heatmap'], self.TARGET_HEATMAP_56)

        # 2. 计算检测损失
        # 注意: 这里的 target 格式需要根据 torchvision 的要求进行匹配
        # 这里是一个简化的示例, 实际中可能需要更复杂的匹配逻辑
        target_labels = torch.cat([t['labels'] for t in self.TARGET_DETECTION], dim=0)
        target_boxes = torch.cat([t['boxes'] for t in self.TARGET_DETECTION], dim=0)
        
        self.LOSS_CLS = self.criterion_cls(self.outputs['class_logits'], target_labels)
        self.LOSS_BBOX = self.criterion_bbox(self.outputs['box_regression'], target_boxes)
        
        # 3. 计算总损失
        total_loss = (self.lambda_loc * self.LOSS_LOC +
                      self.lambda_cls * self.LOSS_CLS +
                      self.lambda_bbox * self.LOSS_BBOX)
        total_loss.backward()
    
    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


