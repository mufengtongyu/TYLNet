import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from modules import network

class cmodel(ABC):
    def __init__(self, config):

        # get device name: cpu/gpu
        self.device = torch.device('cuda:{}'.format(config['gpu_ids'][0])) if config['gpu_ids'] else torch.device('cpu')
        self.save_dir = os.path.join(config['checkpoints_dir'], config['name'])
        os.makedirs(self.save_dir, exist_ok=True)
        self.schedulers = None
        self.gpu_ids = config['gpu_ids']

        torch.backends.cudnn.benchmark = True

        # define init lists
        self.loss_names,self.model_names,self.visual_names,self.optimizers,self.test_result = [],[],[],[],[]

        # learning rate parameter when policy is 'plateau'
        self.metric = None



    @abstractmethod
    def set_input(self, inputs):
        pass



    @abstractmethod
    def forward(self):
        pass


    @abstractmethod
    def test_forward(self):
        pass



    @abstractmethod
    def optimize_parameters(self):
        pass



    def setup(self, config):

        # if training mode, set schedulers by default options
        if config['status'] == "train":
            self.schedulers = [network.get_scheduler(optimizer[1],config) for optimizer in self.optimizers]

        # if testing_usr mode, load weights from saved module
        if config['status'] == "test":
            self.load_networks(int(config['test_epoch']))
        self.print_networks(bool(config['verbose']))



    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()



    def train(self):
        for name in self.model_names:
            if isinstance(name,str):
                net = getattr(self,'net'+name)
                net.train()



    def test(self):
        with torch.no_grad():
            self.test_forward()



    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
        lr_generator = self.optimizers[0][1].param_groups[0]['lr']
        print('update learning rate = %.7f' % lr_generator)



    def resume_scheduler(self,epoch):
        for scheduler in self.schedulers:
            scheduler.last_epoch = epoch
        print("scheduler have resumed from epoch %d" % epoch)



    def get_current_visuals(self):
        """"""
        """ rteurn visualize images, run.py call this function to show and save in visdom"""

        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret



    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'LOSS_' + name))
        return errors_ret



    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)



    def save_optimizers(self, epoch):
        for optim in self.optimizers:
            save_filename = '%s_optim_%s.pth' % (epoch, optim[0])
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(optim[1].state_dict(), save_path)



    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):

        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)



    # def load_networks(self, epoch):
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             load_filename = '%s_net_%s.pth' % (epoch, name)
    #             load_path = os.path.join(self.save_dir, load_filename)
    #             net = getattr(self, 'net' + name)
    #             if isinstance(net, torch.nn.DataParallel):
    #                 net = net.module
    #             print('loading the model from %s' % load_path)

    #             state_dict = torch.load(load_path, map_location=str(self.device))
    #             if hasattr(state_dict, '_metadata'):
    #                 del state_dict._metadata

    #             for key in list(state_dict.keys()):
    #                 self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    #             net.load_state_dict(state_dict)\

    # def load_networks(self, epoch):
    #     """Load all the networks from the disk.
    #     Parameters:
    #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    #     """
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             load_filename = '%s_net_%s.pth' % (epoch, name)
    #             load_path = os.path.join(self.save_dir, load_filename)
    #             net = getattr(self, 'net' + name)
                
    #             # 如果模型本身是 DataParallel 实例, 获取其内部的 module
    #             if isinstance(net, torch.nn.DataParallel):
    #                 net = net.module
                    
    #             print('loading the model from %s' % load_path)
                
    #             # 加载 state_dict
    #             state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True) # 建议加上 weights_only=True 更安全

    #             # 检查是否需要移除 'module.' 前缀
    #             # 如果 state_dict 的第一个 key 是以 'module.' 开头的
    #             if list(state_dict.keys())[0].startswith('module.'):
    #                 # 创建一个新的 OrderedDict 来存放处理后的 key
    #                 new_state_dict = OrderedDict()
    #                 for k, v in state_dict.items():
    #                     name_new = k[7:] # 移除 'module.' 前缀 (7个字符)
    #                     new_state_dict[name_new] = v
    #                 # 使用处理后的 state_dict 加载模型
    #                 net.load_state_dict(new_state_dict)
    #             else:
    #                 # 如果没有 'module.' 前缀, 直接加载
    #                 net.load_state_dict(state_dict)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % save_path)

                # 采纳 PyTorch 警告，使用 weights_only=True
                state_dict = torch.load(save_path, map_location=str(self.device), weights_only=True)
                
                # --- BEGIN FINAL FIX: 清理 thop 和 'module.' 前缀 ---
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # 1. 过滤掉 thop 键
                    if "total_ops" in k or "total_params" in k:
                        continue
                    
                    # 2. 剥离 'module.' 前缀 (如果存在)
                    if k.startswith('module.'):
                        clean_key = k[7:]  # remove 'module.'
                        new_state_dict[clean_key] = v
                    else:
                        new_state_dict[k] = v # 保持原样
                # --- END FINAL FIX ---

                if hasattr(new_state_dict, '_metadata'):
                    del new_state_dict._metadata

                # 现在 new_state_dict 中的键 (e.g., "pre.0.conv.weight") 
                # 与解包后的 'net' 对象的层级完全匹配
                for key in list(new_state_dict.keys()):
                    self.__patch_instance_norm_state_dict(new_state_dict, net, key.split('.'))
                
                # 使用 strict=False 加载
                # 因为 __patch_instance_norm_state_dict 可能会 pop 掉 InstanceNorm 的键
                net.load_state_dict(new_state_dict, strict=False)


    def load_optimizers(self, epoch):
        for optim in self.optimizers:
            load_filename = '%s_optim_%s.pth' % (epoch, optim[0])
            load_path = os.path.join(self.save_dir, load_filename)
            print('loading the optimizers from %s' % load_path)

            state_dict = torch.load(load_path, map_location=str(self.device))
            optim[1].load_state_dict(state_dict)



    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')



    @staticmethod
    def set_requires_grad(nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad