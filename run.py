import time
from datasets import create_dataset
from modules import create_model
from utils import startup
import utils.tools as util
import os
import numpy as np
import evaluation
import torch
import torch.nn as nn

from thop import profile

def train(config):
    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)

# --- 新增代码：计算 GFLOPs 和 Params (已修复) ---
    print("Calculating model parameters and GFLOPs...")
    device = model.device  # 从模型获取设备 (e.g., 'cuda:0')
    
    # 从 config (File 1) 获取 img_size
    img_size = int(config['img_size']) 
    
    # 确定输入通道 (来自 File 4, 6 个输入)
    input_channels = 6 
    
    # 1. 从 DataParallel 包装器中获取真实模型
    net_to_profile = model.netG.module
    
    # 2. 将真实模型移至 CPU
    net_to_profile.cpu()
    
    # 3. 创建一个 CPU 上的虚拟张量
    dummy_input_cpu = torch.randn(1, input_channels, img_size, img_size)

    # 4. 在 CPU 上进行分析 (更稳定)
    net_to_profile.eval() # 必须设为 eval 模式
    flops, params = profile(net_to_profile, inputs=(dummy_input_cpu, ), verbose=False)
    
    # 5. 将模型移回原始设备 (e.g., 'cuda:0') 以便继续训练
    net_to_profile.to(device)
    model.train() # 将主模型包装器恢复为训练模式

    params_m = params / 1e6
    gflops = flops / 1e9
    # --------------------------------------------------
    
    dataset_size = len(dataset)  # get the size of dataset
    print('The number of training images = %d' % dataset_size)
    test_config = config.copy()
    test_config['status'] = 'test'
    test_config['num_threads'] = 1
    test_dataset = create_dataset(test_config)
    test_dataset_size = len(test_dataset)
    print('The number of testing images = %d' % test_dataset_size)
    total_iters = 0  # total iteration for datasets points

    if int(config['resume_epoch']) > 0:
        print("\n resume traing from rpoch " + str(int(config['resume_epoch']))+" ...")
        model.resume_scheduler(int(config['resume_epoch']))
        model.load_networks(config['resume_epoch'])
        model.load_optimizers(config['resume_epoch'])
    # outter iteration for differtent epoch; we save module via <epoch_count> and <epoch_count>+<save_latest_freq> options
    for epoch in range(int(config['resume_epoch'])+1, int(config['epoch']) +1):
        epoch_start_time = time.time()  # note the starting time for current epoch
        epoch_iter = 0  # iteration times for current epoch, reset to 0 for each epoch

        #innear iteration for single epoch
        for i, data in enumerate(dataset):
            total_iters = total_iters + int(config['train_batch_size'])
            epoch_iter = epoch_iter + int(config['train_batch_size'])
            model.set_input(data)  # push loading image to the module
            model.optimize_parameters()  # calculate loss, gradient and refresh module parameter

        if epoch % int(config['save_epoch_freq']) == 0:  # save module each <save_epoch_freq> epoch iterations
            print('saving the module at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
            model.save_optimizers(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, int(config['epoch']), time.time() - epoch_start_time))

        val(config=test_config, epoch=epoch, dataset=test_dataset,model=model)

        # update learning rate after each epoch
        model.update_learning_rate()


def val(config,epoch,dataset,model):

    result_root_path = os.path.join(config['checkpoints_dir'], config['name'], config['results_dir'],"epoch"+str(epoch))
    util.mkdir(result_root_path)

    model.eval()
    save_npy = np.ndarray(shape=(dataset.__len__() + 1, 2), dtype=np.float64)
    save_npy[0][0], save_npy[0][1] = -1, -1
    print("Start evaluating epoch "+str(epoch)+"...")

    for i, data in enumerate(dataset):
        model.set_input(data)  # push test datasets to module
        model.test()  # forward module
        datapoints = (model.test_result[0][1]).cpu().data.numpy()
        index = data["PATH"].cpu().data.numpy()[0]
        save_npy[index][0], save_npy[index][1] = datapoints[0][0], datapoints[0][1]
        dist_img = model.test_result[1][1]
        util.save_image(util.tensor2im(dist_img), os.path.join(result_root_path, str(index) + ".png"))
    model.train()

    np.save(os.path.join(result_root_path, 'regression.npy'), save_npy)
    l2_dist, dist1, dist2, dist3, dist4, dist5, dist6 = evaluation.evaluate_detailed(save_npy)
    text = open(os.path.join(config['checkpoints_dir'], config['name'], config['results_dir'], "evaluation.txt"), "a+")
    text.writelines(
        "EPOCH " + str(epoch) + ": " + str(round(l2_dist, 4)) + "   " + str(round(dist1, 4)) + '   ' +
        str(round(dist2, 4)) + "   " + str(round(dist3, 4)) + '   ' +
        str(round(dist4, 4)) + "   " +str(round(dist5, 4)) + '   ' +
        str(round(dist6, 4)) + "\n")
    text.close()
    print("Testing npy result have been saved! Evaluation distance: "+str(round(l2_dist))+"   "+ str(round(dist1, 4)) + '   ' +
        str(round(dist2, 4)) + "   "+str(round(dist3, 4)) + '   ' +
        str(round(dist4, 4)) + "   "+str(round(dist5, 4)) + '   ' +
        str(round(dist6, 4)))

if __name__ == '__main__':
    configs_stage1 = startup.SetupConfigs(config_path='configs/TCLNET_STAGE1.yaml')
    configs_stage1 = configs_stage1.setup()

    if (configs_stage1['status'] == "train") :
        train(configs_stage1)


