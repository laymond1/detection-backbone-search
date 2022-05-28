import json
import numpy as np
import torch
import torch.nn as nn

from ofa.utils.layers import (
    set_layer_from_config,
    MBConvLayer,
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    ResidualBlock,
)
from ofa.utils import (
    download_url,
    make_divisible,
    val2list,
    MyNetwork,
    MyGlobalAvgPool2d,
)

from proxyless_nets import ProxylessNASNets
from search_space import Blocks, blocks_key


class MobileNetV2(ProxylessNASNets):
    def __init__(
        self,
        n_classes=1000,
        width_mult=1.0,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.2,
        expand_ratio=None,
        depth_param=None,
        stage_width_list=None,
    ):

        expand_ratio = 6 if expand_ratio is None else expand_ratio

        input_channel = 32
        last_channel = 1280

        input_channel = make_divisible( # check divisible by 8
            input_channel * width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        last_channel = (
            make_divisible(last_channel * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            if width_mult > 1.0
            else last_channel
        )

        inverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],
            [expand_ratio, 64, 2, 2], # 2-P2/4 - 2th
            [expand_ratio, 128, 3, 2], # 3-P3/8 - 4th
            [expand_ratio, 256, 4, 2], # 4-P4/16 - 7th
            [expand_ratio, 512, 3, 1], 
            [expand_ratio, 512, 3, 2], # 6-P5/32 - 14th
            [expand_ratio, 1024, 1, 1],
        ]
        self.inverted_residual_setting = np.array(inverted_residual_setting)

        if depth_param is not None:
            assert isinstance(depth_param, int)
            for i in range(1, len(inverted_residual_setting) - 1):
                inverted_residual_setting[i][2] = depth_param

        if stage_width_list is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = stage_width_list[i]

        # first conv layer
        first_conv = ConvLayer(
            3,
            input_channel,
            kernel_size=3,
            stride=2,
            use_bn=True,
            act_func="relu6",
            ops_order="weight_bn_act",
        )
        # inverted residual blocks
        blocks = []
        stage_ends_idx = []
        i_th = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

            for i in range(n):
                # stride size
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # Skip-connection
                if stride == 1:
                    if input_channel == output_channel:
                        shortcut = IdentityLayer(input_channel, input_channel)
                    else:
                        shortcut = None
                else:
                    shortcut = None
                # Search Space
                _ops = nn.ModuleList()
                for k in blocks_key:
                    mobile_inverted_conv = Blocks[k](
                            in_channels=input_channel,
                            out_channels=output_channel,
                            stride=stride,
                            expand_ratio=t,
                    )
                    _ops.append(
                        ResidualBlock(mobile_inverted_conv, shortcut)
                    )
                blocks.append(_ops)
                input_channel = output_channel
                i_th += 1

        stage_ends_idx.append(i_th-1)

        # 1x1_conv before global average pooling
        feature_mix_layer = ConvLayer(
            input_channel,
            last_channel,
            kernel_size=1,
            use_bn=True,
            act_func="relu6",
            ops_order="weight_bn_act",
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(MobileNetV2, self).__init__(
            first_conv, blocks, stage_ends_idx, feature_mix_layer, classifier
        )

        # set bn param
        self.set_bn_param(*bn_param)

    # Override config function
    def config(self, arch=None):
        if arch is None:
            unzip = lambda ops : [i.config for i in ops]
            return {
                "name": MobileNetV2.__name__,
                "bn": self.get_bn_param(),
                "first_conv": self.first_conv.config,
                "blocks": list(map(unzip, self.blocks)),
                "stage_ends_idx": self.stage_ends_idx,
                "feature_mix_layer": None
                    if self.feature_mix_layer is None
                    else self.feature_mix_layer.config,
                "classifier": self.classifier.config,
            }
        else:
            assert len(arch) == len(self.blocks) 
            return {
                "name": MobileNetV2.__name__,
                "bn": self.get_bn_param(),
                "first_conv": self.first_conv.config,
                "blocks": [self.blocks[i][k].config for i, k in enumerate(arch)],
                "stage_ends_idx": self.stage_ends_idx,
                "feature_mix_layer": None
                    if self.feature_mix_layer is None
                    else self.feature_mix_layer.config,
                "classifier": self.classifier.config,
            }

    # Override build_from_config function
    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        stage_ends_idx = config['stage_ends_idx']
        feature_mix_layer = set_layer_from_config(config["feature_mix_layer"])
        classifier = set_layer_from_config(config["classifier"])

        blocks = []
        for ops_config in config["blocks"]:
            if isinstance(ops_config, list):
                _ops = nn.ModuleList()
                for block_config in ops_config:
                    _ops.append(ResidualBlock.build_from_config(block_config))
                blocks.append(_ops)
            else:
                block_config = ops_config
                blocks.append(ResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, stage_ends_idx, feature_mix_layer, classifier)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    # Network Sampling
    def sample_from_net(self, arch=None):
        assert arch is not None
        assert len(arch) == len(self.blocks)

        # sample_net = nn.ModuleList()
        sample_net = []
        sample_net.append(self.first_conv)
        for i, ops in enumerate(self.blocks):
            sample_net.append(ops[arch[i]])
        sample_net.append(self.feature_mix_layer)
        sample_net.append(self.global_avg_pool)
        sample_net.append(self.classifier)
        return nn.Sequential(*sample_net)

    # Override load_state_dict function
    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()

        # Only extract stated_dict from Supernet for Object detection backbone
        for key in state_dict:
            if key not in current_state_dict:
                continue
            else:
                new_key = key
            current_state_dict[new_key] = state_dict[key]
        super(MyNetwork, self).load_state_dict(current_state_dict)

    # Override forward function
    def forward(self, x, rngs=None):
        if rngs is None:
            rngs = tuple(np.random.randint(len(blocks_key)) for i in range(self.inverted_residual_setting[:,2].sum()))
        assert self.inverted_residual_setting[:,2].sum() == len(rngs)

        x = self.first_conv(x)
        for i, select_block in enumerate(self.blocks):
            x = select_block[rngs[i]](x)

        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    import sys
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from latency_lookup_table import MBv2LatencyTable
    from backbone import BackBoneMobileNetV2

    net = MobileNetV2()
    input = torch.randn(1, 3, 224, 224)
    net(input).shape

    net_config = net.config()
    net.build_from_config(net_config)

    net_config = net.config(arch = [0,1,2,1,1, 1,2,2,2,2, 1,2,0,2,1, 1,2])
    net.build_from_config(net_config)

    sample_net = net.sample_from_net(arch = [0,1,2,1,1, 1,2,2,2,2, 1,2,0,2,1, 1,2])

    estimator = MBv2LatencyTable(url='latency_lookup_table/mobile_lut.yaml')

    estimator.count_flops_given_config(net_config)
    latency = estimator.predict_network_latency_given_config(net_config)
    # estimator.predict_network_latency(net) # can not use

    net.eval()
    torch.save(net.state_dict(), './models/latest_model2.pt')

    backbone = BackBoneMobileNetV2()
    state_dict = torch.load('./models/latest_model2.pt')
    backbone.load_state_dict(state_dict)

    print(net.config)

    estimator = MBv2LatencyTable(url='latency_lookup_table/mobile_lut.yaml')
    # lat = estimator.predict_network_latency(net)
    # print(lat)



