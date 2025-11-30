import time
from datasets import create_dataset
from modules import create_model
from utils import startup
import utils.tools as util
import os
import numpy as np
import evaluation
import torch

from thop import profile


def train(config):
    dataset = create_dataset(config)
    model = create_model(config)
    model.setup(config)

    print("Calculating model parameters and GFLOPs...")
    device = model.device

    img_size = int(config['img_size'])
    input_channels = 6

    net_to_profile = model.netG.module if hasattr(model.netG, 'module') else model.netG

    net_to_profile.cpu()

    dummy_input_cpu = torch.randn(1, input_channels, img_size, img_size)

    net_to_profile.eval()
    flops, params = profile(net_to_profile, inputs=(dummy_input_cpu, ), verbose=False)

    net_to_profile.to(device)
    model.train()

    params_m = params / 1e6
    gflops = flops / 1e9

    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    test_config = config.copy()
    test_config['status'] = 'test'
    test_config['num_threads'] = 1
    test_dataset = create_dataset(test_config)
    test_dataset_size = len(test_dataset)
    print('The number of testing images = %d' % test_dataset_size)
    total_iters = 0

    best_metrics = None
    best_metric_value = float('inf')
    patience = int(config.get('early_stop_patience', 10))
    min_delta = float(config.get('early_stop_delta', 0.0))
    epochs_no_improve = 0
    finished_epoch = int(config['resume_epoch'])

    if int(config['resume_epoch']) > 0:
        print("\n resume traing from rpoch " + str(int(config['resume_epoch']))+" ...")
        model.resume_scheduler(int(config['resume_epoch']))
        model.load_networks(config['resume_epoch'])
        model.load_optimizers(config['resume_epoch'])
    for epoch in range(int(config['resume_epoch'])+1, int(config['epoch']) +1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            total_iters = total_iters + int(config['train_batch_size'])
            epoch_iter = epoch_iter + int(config['train_batch_size'])
            model.set_input(data)
            model.optimize_parameters()

        if epoch % int(config['save_epoch_freq']) == 0:
            print('saving the module at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
            model.save_optimizers(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, int(config['epoch']), time.time() - epoch_start_time))

        metrics = val(config=test_config, epoch=epoch, dataset=test_dataset,model=model)

        if metrics is not None:
            current_metric = metrics['l2_dist']
            if best_metrics is None or current_metric + min_delta < best_metric_value:
                best_metrics = metrics
                best_metric_value = current_metric
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        model.update_learning_rate()

        finished_epoch = epoch
        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} evaluations)")
            break

    return {
        'params_m': params_m,
        'gflops': gflops,
        'best_metrics': best_metrics,
        'epochs_ran': finished_epoch,
        'attention_module': config.get('attention_module', 'se'),
        'use_skip_connection': bool(config.get('use_skip_connection', False)),
    }


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

    return {
        'l2_dist': l2_dist,
        'dist1': dist1,
        'dist2': dist2,
        'dist3': dist3,
        'dist4': dist4,
        'dist5': dist5,
        'dist6': dist6,
    }


def run_full_module_suite(base_config):
    module_variants = [
        ("baseline", {"attention_module": "baseline", "use_skip_connection": False}),
        ("baseline_skip", {"attention_module": "baseline", "use_skip_connection": True}),
        ("se", {"attention_module": "se"}),
        ("skattention", {"attention_module": "skattention"}),
        ("cbam", {"attention_module": "cbam"}),
        ("resblock_cbam", {"attention_module": "resblock+cbam"}),
        ("ca", {"attention_module": "ca"}),
        ("eca", {"attention_module": "eca"}),
        ("simam", {"attention_module": "simam"}),
    ]

    summary_dir = os.path.join(base_config['checkpoints_dir'], base_config['name'])
    util.mkdir(summary_dir)
    summary_path = os.path.join(summary_dir, 'ablation_summary.txt')
    summary_records = []

    for variant_name, overrides in module_variants:
        variant_config = base_config.copy()
        variant_config.update(overrides)
        variant_config['full_module_test_mode'] = False
        variant_config['name'] = f"{base_config['name']}_{variant_name}"

        print(f"\n================ Running variant: {variant_name} ================")
        result = train(variant_config)
        result['variant'] = variant_name
        result['config_snapshot'] = {
            'attention_module': variant_config.get('attention_module', 'se'),
            'use_skip_connection': bool(variant_config.get('use_skip_connection', False))
        }
        summary_records.append(result)

    with open(summary_path, 'w') as fp:
        fp.write('variant\tattention_module\tuse_skip_connection\tparams(M)\tGFLOPS\tbest_l2\tran_epochs\n')
        for record in summary_records:
            best_l2 = record.get('best_metrics', {}).get('l2_dist', 'N/A') if record.get('best_metrics') else 'N/A'
            fp.write(f"{record['variant']}\t{record['config_snapshot']['attention_module']}\t{record['config_snapshot']['use_skip_connection']}\t{record['params_m']:.4f}\t{record['gflops']:.4f}\t{best_l2}\t{record['epochs_ran']}\n")


if __name__ == '__main__':
    configs_stage1 = startup.SetupConfigs(config_path='configs/TCLNET_STAGE1.yaml')
    configs_stage1 = configs_stage1.setup()

    if configs_stage1.get('full_module_test_mode', False):
        run_full_module_suite(configs_stage1)
    elif (configs_stage1['status'] == "train") :
        train(configs_stage1)


