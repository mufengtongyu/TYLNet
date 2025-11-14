from modules import network
from modules.components import Improved_TCLNET

def create_backbone(net_name, init_type, init_gain, gpu_ids):
    net = None
    if net_name == 'TCLNET':
        net = Improved_TCLNET.net()
    else:
        raise NotImplementedError("model not found")
    return network.init_net(net, init_type, init_gain, gpu_ids)
