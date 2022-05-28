import json
from collections import OrderedDict
import numpy as np
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

from ofa.imagenet_classification.networks import ProxylessNASNets
from supernet.search_space import Blocks, blocks_key

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN


class BackBoneMobileNetV2(MyNetwork):
    def __init__(
        self,
        arch=None,
        n_classes=1000,
        width_mult=1.0,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.2,
        ks=None, # not used
        expand_ratio=None,
        depth_param=None,
        stage_width_list=None,
    ):
        super(BackBoneMobileNetV2, self).__init__()

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
            [1, 16, 1, 1],
            [expand_ratio, 24, 2, 2],
            [expand_ratio, 32, 3, 2],
            [expand_ratio, 64, 4, 2],
            [expand_ratio, 96, 3, 1],
            [expand_ratio, 160, 3, 2],
            [expand_ratio, 320, 1, 1],
        ]
        self.inverted_residual_setting = np.array(inverted_residual_setting)
        

        if depth_param is not None:
            assert isinstance(depth_param, int)
            for i in range(1, len(inverted_residual_setting) - 1):
                inverted_residual_setting[i][2] = depth_param

        if stage_width_list is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = stage_width_list[i]

        mode = 'search'
        if 'search' in mode:
            architecture = None # ex)  [0,1,2,1,1, 1,2,2,2,2, 1,2,0,2,1, 1,2]
            self.blocks_key = blocks_key
            self.num_states = sum(self.inverted_residual_setting[:,2])

        # first conv layer
        self.first_conv = ConvLayer(
            3,
            input_channel,
            kernel_size=3,
            stride=2,
            use_bn=True,
            act_func="relu6",
            ops_order="weight_bn_act",
        )
        # inverted residual blocks
        self.blocks = []
        self.stage_ends_idx = []
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
                
                if architecture is None:
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
                else:
                    # Architecture
                    mobile_inverted_conv = Blocks[architecture[i_th]](
                                in_channels=input_channel,
                                out_channels=output_channel,
                                stride=stride,
                                expand_ratio=t,
                        )
                    _ops.append(ResidualBlock(mobile_inverted_conv, shortcut)
                        )
                self.blocks.append(_ops)
                input_channel = output_channel
                i_th += 1
        self.blocks = nn.Sequential(*self.blocks)
        self.stage_ends_idx.append(i_th-1)

        self.stage_indices = [i for i, blocks in enumerate(self.blocks) if blocks[0].conv.stride > 1]
        self.stage_channels = [i for i, blocks in enumerate(self.blocks) if blocks[0].conv.stride > 1]
        self.return_layers = None

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        return _str

    @property
    def config(self, arch=None):
        if arch is None:
            unzip = lambda ops : [i.config for i in ops]
            return {
                "name": BackBoneMobileNetV2.__name__,
                "bn": self.get_bn_param(),
                "first_conv": self.first_conv.config,
                "blocks": list(map(unzip, self.blocks)),
                "stage_ends_idx": self.stage_ends_idx,
            }
        else:
            assert len(arch) == len(self.blocks) 
            return {
                "name": BackBoneMobileNetV2.__name__,
                "bn": self.get_bn_param(),
                "first_conv": self.first_conv.config,
                "blocks": [self.blocks[i][k].config for i, k in enumerate(arch)],
                "stage_ends_idx": self.stage_ends_idx,
            }
    
    @staticmethod # TODO Fix this method to 
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        stage_ends_idx = config['stage_ends_idx']

        blocks = []
        for ops_config in config["blocks"]:
            for block_config in ops_config:
                blocks.append(ResidualBlock.build_from_config(block_config))

        net = BackBoneMobileNetV2(first_conv, blocks, stage_ends_idx)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    @staticmethod
    def sample_from_net(self, arch=None):
        assert arch is not None
        assert len(arch) == len(self.blocks)

        sample_net = []
        sample_net.append(self.first_conv)
        for i, ops in enumerate(self.blocks):
            sample_net.append(ops[arch[i]])
        return nn.Sequential(*sample_net)

    def set_backbone_fpn(self, returned_layers):
        """
            Specifying features for FPN & return in_channels_list
        """
        # [1, 3, 6, 13] except first conv
        stage_indices = [i for i, block in enumerate(self.blocks) if block[0].conv.stride > 1]
        num_stages = len(stage_indices)

        # freeze backbone
        for parameter in self.blocks.parameters():
            parameter.requires_grad_(False)

        self.out_channels = 256
        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
        # self.return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}
        self.return_layers = [stage_indices[i] for i in returned_layers]

        in_channels_list = [self.blocks[stage_indices[i]][0].conv.out_channels if i != 0 else self.blocks[stage_indices[i]].out_channels for i in returned_layers]
        return in_channels_list

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

    def forward(self, x, rngs=None):
        outputs = []
        outputs = OrderedDict()
        x = self.first_conv(x)

        for i, select_block in enumerate(self.blocks):
            x = select_block(x) if rngs is None else select_block[rngs[i]](x)
            if self.return_layers is not None and i in self.return_layers:
                outputs[str(i)] = x
                # outputs.append(x) # P2/4, P3/8, P4/16, P5/32
            elif self.return_layers is None and i in self.stage_ends_idx:
                outputs[str(i)] = x
                # outputs.append(x)
        return outputs

if __name__ == "__main__":


    net = BackBoneMobileNetV2()
    input = torch.randn(1, 3, 224, 224)
    arch = [0,1,2,1,1, 1,2,2,2,2, 1,2,0,2,1, 1,2]
    output = net(input, rngs=arch)
    len(output)


    in_channels_list = net.set_backbone_fpn(returned_layers=[1,2,3])
    net.return_layers
    output = net(input, rngs=arch)
    len(output)
    output['3'].shape
    output['6'].shape
    output['13'].shape

    outputs = [output['3'], output['6'], output['13']]

    from torchvision.ops import FeaturePyramidNetwork
    fpn = FeaturePyramidNetwork(in_channels_list, 256)
    fpn = FeaturePyramidNetwork(in_channels_list, net.out_channels)
    output = fpn(output)

    len(output)

    output['3'].shape
    output['6'].shape
    output['13'].shape

    anchors = [
    [10,13, 16,30, 33,23],  # P3/8
    [30,61, 62,45, 59,119], # P4/16
    [116,90, 156,198, 373,326] # P5/32
    ]
    len(anchors)

    from yolov5.models.yolo import Detect
    detector = Detect(nc=80, anchors=anchors, ch=in_channels_list)
    detector
    detector(output)

    state_dict = torch.load('./models/latest_model.pt')
    net.load_state_dict(state_dict)


# TODO make FPN & HEAD


# def build_mobilenetv2_fpn_backbone(
#     backbone,
#     pretrained=False,
#     fpn=True,
#     norm_layer=misc_nn_ops.FrozenBatchNorm2d,
#     trainable_layers=2,
#     returned_layers=None
#     extra_blocks=None
# ):
#     backbone.extract_features_backbone()

#     stage_indices = [0] + [i for i, blocks in enumerate(backbone.blocks) if blocks[0].conv.stride > 1]
#     num_stages = len(stage_indices)

#     backbone = backbone.features
#     assert 0 <= trainable_layers <= num_stages
#     freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

#     for b in backbone[:freeze_before]:
#         for parameter in b.parameters():
#             parameter.requires_grad_(False)

#     out_channels = 256
#     if fpn:
#         if extra_blocks is None:
#             extra_blocks = LastLevelMaxPool()

#         if returned_layers is None:
#             returned_layers = [num_stages - 2, num_stages - 1]
#         assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
#         return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

#         in_channels_list = [backbone[stage_indices[i]][0].conv.out_channels if i != 0 else backbone[stage_indices[i]].out_channels for i in returned_layers]
        
#         return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

#     else:
#         m = nn.Sequential(
#             backbone,
#             # depthwise linear combination of channels to reduce their size
#             nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
#         )
#         m.out_channels = out_channels
#         return m