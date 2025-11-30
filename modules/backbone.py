from modules import network
from modules.components import Improved_TCLNET


def create_backbone(net_name, init_type, init_gain, gpu_ids, config=None):
    net = None
    if net_name == 'TCLNET':
        attention_type = 'se'
        use_skip_connection = False

        if config is not None:
            attention_type = str(config.get('attention_module', 'se')).lower()
            use_skip_connection = bool(config.get('use_skip_connection', False))

            if attention_type == 'baseline_skip':
                use_skip_connection = True
                attention_type = 'baseline'

        net = Improved_TCLNET.net(attention_type=attention_type, use_skip_connection=use_skip_connection)
    else:
        raise NotImplementedError("model not found")
    return network.init_net(net, init_type, init_gain, gpu_ids)
