# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import yaml
from ofa.utils import download_url, make_divisible, MyNetwork

from supernet import MobileNetV2

__all__ = [
    "count_conv_flop",
    "MBv2LatencyTable",
]


def count_conv_flop(out_size, in_channels, out_channels, kernel_size, groups):
    out_h = out_w = out_size
    delta_ops = (
        in_channels * out_channels * kernel_size * kernel_size * out_h * out_w / groups
    )
    return delta_ops


class LatencyTable(object):
    def __init__(
        self,
        local_dir="~/.ofa/latency_tools/",
        url="https://hanlab.mit.edu/files/proxylessNAS/LatencyTools/mobile_trim.yaml",
    ):
        if url.startswith("http"):
            fname = download_url(url, local_dir, overwrite=True)
        else:
            fname = url
        with open(fname, "r") as fp:
            self.lut = yaml.full_load(fp)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return "x".join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def query(self, **kwargs):
        raise NotImplementedError

    def predict_network_latency(self, net, image_size):
        raise NotImplementedError

    def predict_network_latency_given_config(self, net_config, image_size):
        raise NotImplementedError

    @staticmethod
    def count_flops_given_config(net_config, image_size=224):
        raise NotImplementedError


class MBv2LatencyTable(LatencyTable):
    def query(
        self,
        l_type: str,
        input_shape,
        output_shape,
        mid=None,
        ks=None,
        stride=None,
        id_skip=None,
        se=None,
        h_swish=None,
    ):
        """
        :param l_type:
                Layer type must be one of the followings
                1. `Conv`: The initial 3x3 conv with stride 2.
                2. `Conv_1`: feature_mix_layer
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param input_shape: input shape (h, w, #channels)
        :param output_shape: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param ks: kernel size
        :param stride:
        :param id_skip: indicate whether has the residual connection
        """ 
        infos = [
            l_type,
            "input:%s" % self.repr_shape(input_shape),
            "output:%s" % self.repr_shape(output_shape),
        ]

        if l_type in ("expanded_conv",):
            assert None not in (mid, ks, stride, id_skip, se, h_swish)
            infos += [
                "expand:%d" % mid,
                "kernel:%d" % ks,
                "stride:%d" % stride,
                "idskip:%d" % id_skip,
                # "se:%d" % se,
                # "hs:%d" % h_swish,
            ]
        key = "-".join(infos)
        return self.lut[key]["mean"]

    # It cannot be used because it is supernet's latency
    def predict_network_latency(self, net, image_size=224):
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [image_size, image_size, 3], # input shape
            [(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels], # output shape
        )
        # blocks
        fsize = (image_size + 1) // 2
        for ops in net.blocks:
            for block in ops:
                mb_conv = block.conv
                shortcut = block.shortcut

                if mb_conv is None:
                    continue
                if shortcut is None:
                    idskip = 0
                else:
                    idskip = 1
                out_fz = int((fsize - 1) / mb_conv.stride + 1)
                block_latency = self.query(
                    "expanded_conv",
                    [fsize, fsize, mb_conv.in_channels],
                    [out_fz, out_fz, mb_conv.out_channels],
                    mid=mb_conv.expand_ratio,
                    ks=mb_conv.kernel_size,
                    stride=mb_conv.stride,
                    id_skip=idskip,
                    se=1 if mb_conv.use_se else 0,
                    h_swish=1 if mb_conv.act_func == "h_swish" else 0,
                )
                predicted_latency += block_latency
                fsize = out_fz
        # final expand layer
        predicted_latency += self.query(
            "Conv_1",
            [fsize, fsize, net.final_expand_layer.in_channels],
            [fsize, fsize, net.final_expand_layer.out_channels],
        )
        # global average pooling
        predicted_latency += self.query(
            "AvgPool2D",
            [fsize, fsize, net.final_expand_layer.out_channels],
            [1, 1, net.final_expand_layer.out_channels],
        )
        # feature mix layer
        predicted_latency += self.query(
            "Conv_2",
            [1, 1, net.feature_mix_layer.in_channels],
            [1, 1, net.feature_mix_layer.out_channels],
        )
        # classifier
        predicted_latency += self.query(
            "Logits", [1, 1, net.classifier.in_features], [net.classifier.out_features]
        )
        return predicted_latency

    # If you want to calculate specific network's latency, you should use it
    def predict_network_latency_given_config(self, net_config, image_size=224):
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [image_size, image_size, 3],
            [
                (image_size + 1) // 2,
                (image_size + 1) // 2,
                net_config["first_conv"]["out_channels"],
            ],
        )
        # blocks
        fsize = (image_size + 1) // 2
        for block in net_config["blocks"]:
            mb_conv = (
                block["mobile_inverted_conv"]
                if "mobile_inverted_conv" in block
                else block["conv"]
            )
            shortcut = block["shortcut"]

            if mb_conv is None:
                continue
            if shortcut is None:
                idskip = 0
            else:
                idskip = 1
            out_fz = int((fsize - 1) / mb_conv["stride"] + 1)
            if mb_conv["mid_channels"] is None:
                mb_conv["mid_channels"] = round(
                    mb_conv["in_channels"] * mb_conv["expand_ratio"]
                )
            block_latency = self.query(
                "expanded_conv",
                [fsize, fsize, mb_conv["in_channels"]],
                [out_fz, out_fz, mb_conv["out_channels"]],
                mid=mb_conv["expand_ratio"],
                ks=mb_conv["kernel_size"],
                stride=mb_conv["stride"],
                id_skip=idskip,
                se=1 if mb_conv["use_se"] else 0,
                h_swish=1 if mb_conv["act_func"] == "h_swish" else 0,
            )
            predicted_latency += block_latency
            fsize = out_fz
        # # final expand layer
        # predicted_latency += self.query(
        #     "Conv_1",
        #     [fsize, fsize, net_config["final_expand_layer"]["in_channels"]],
        #     [fsize, fsize, net_config["final_expand_layer"]["out_channels"]],
        # )
        # global average pooling
        # predicted_latency += self.query(
            # "AvgPool2D",
            # [fsize, fsize, net_config["final_expand_layer"]["out_channels"]],
            # [1, 1, net_config["final_expand_layer"]["out_channels"]],
        # )
        # feature mix layer
        # predicted_latency += self.query(
        #     "Conv_2",
        #     [1, 1, net_config["feature_mix_layer"]["in_channels"]],
        #     [1, 1, net_config["feature_mix_layer"]["out_channels"]],
        # )
        # classifier
        predicted_latency += self.query(
            "Logits",
            [1, 1, net_config["classifier"]["in_features"]],
            [net_config["classifier"]["out_features"]],
        )
        return predicted_latency

    @staticmethod
    def count_flops_given_config(net_config, image_size=224):
        flops = 0
        # first conv
        flops += count_conv_flop(
            (image_size + 1) // 2, 3, net_config["first_conv"]["out_channels"], 3, 1
        )
        # blocks
        fsize = (image_size + 1) // 2
        for block in net_config["blocks"]:
            mb_conv = (
                block["mobile_inverted_conv"]
                if "mobile_inverted_conv" in block
                else block["conv"]
            )
            if mb_conv is None:
                continue
            out_fz = int((fsize - 1) / mb_conv["stride"] + 1)
            if mb_conv["mid_channels"] is None:
                mb_conv["mid_channels"] = round(
                    mb_conv["in_channels"] * mb_conv["expand_ratio"]
                )
            if mb_conv["expand_ratio"] != 1:
                # inverted bottleneck
                flops += count_conv_flop(
                    fsize, mb_conv["in_channels"], mb_conv["mid_channels"], 1, 1
                )
            # depth conv
            flops += count_conv_flop(
                out_fz,
                mb_conv["mid_channels"],
                mb_conv["mid_channels"],
                mb_conv["kernel_size"],
                mb_conv["mid_channels"],
            )
            if mb_conv["use_se"]:
                # SE layer
                se_mid = make_divisible(
                    mb_conv["mid_channels"] // 4, divisor=MyNetwork.CHANNEL_DIVISIBLE
                )
                flops += count_conv_flop(1, mb_conv["mid_channels"], se_mid, 1, 1)
                flops += count_conv_flop(1, se_mid, mb_conv["mid_channels"], 1, 1)
            # point linear
            flops += count_conv_flop(
                out_fz, mb_conv["mid_channels"], mb_conv["out_channels"], 1, 1
            )
            fsize = out_fz
        # final expand layer
        # flops += count_conv_flop(
            # fsize,
            # net_config["final_expand_layer"]["in_channels"],
            # net_config["final_expand_layer"]["out_channels"],
            # 1,
            # 1,
        # )
        # feature mix layer
        flops += count_conv_flop(
            7,
            net_config["feature_mix_layer"]["in_channels"],
            net_config["feature_mix_layer"]["out_channels"],
            1,
            1,
        )
        # classifier
        flops += count_conv_flop(
            1,
            net_config["classifier"]["in_features"],
            net_config["classifier"]["out_features"],
            1,
            1,
        )
        return flops / 1e6  # MFLOPs


class BackBoneMBv2LatencyTable(LatencyTable):
    def query(
        self,
        l_type: str,
        input_shape,
        output_shape,
        mid=None,
        ks=None,
        stride=None,
        id_skip=None,
        se=None,
        h_swish=None,
    ):
        """
        :param l_type:
                Layer type must be one of the followings
                1. `Conv`: The initial 3x3 conv with stride 2.
                2. `Conv_1`: feature_mix_layer
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param input_shape: input shape (h, w, #channels)
        :param output_shape: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param ks: kernel size
        :param stride:
        :param id_skip: indicate whether has the residual connection
        """ 
        infos = [
            l_type,
            "input:%s" % self.repr_shape(input_shape),
            "output:%s" % self.repr_shape(output_shape),
        ]

        if l_type in ("expanded_conv",):
            assert None not in (mid, ks, stride, id_skip, se, h_swish)
            infos += [
                "expand:%d" % mid,
                "kernel:%d" % ks,
                "stride:%d" % stride,
                "idskip:%d" % id_skip,
                # "se:%d" % se,
                # "hs:%d" % h_swish,
            ]
        key = "-".join(infos)
        return self.lut[key]["mean"]

    def predict_network_latency(self, net, image_size=224):
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [image_size, image_size, 3], # input shape
            [(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels], # output shape
        )
        # blocks
        fsize = (image_size + 1) // 2
        for ops in net.blocks:
            for block in ops:
                mb_conv = block.conv
                shortcut = block.shortcut

                if mb_conv is None:
                    continue
                if shortcut is None:
                    idskip = 0
                else:
                    idskip = 1
                out_fz = int((fsize - 1) / mb_conv.stride + 1)
                block_latency = self.query(
                    "expanded_conv",
                    [fsize, fsize, mb_conv.in_channels],
                    [out_fz, out_fz, mb_conv.out_channels],
                    mid=mb_conv.expand_ratio,
                    ks=mb_conv.kernel_size,
                    stride=mb_conv.stride,
                    id_skip=idskip,
                    se=1 if mb_conv.use_se else 0,
                    h_swish=1 if mb_conv.act_func == "h_swish" else 0,
                )
                predicted_latency += block_latency
                fsize = out_fz
        # final expand layer
        predicted_latency += self.query(
            "Conv_1",
            [fsize, fsize, net.final_expand_layer.in_channels],
            [fsize, fsize, net.final_expand_layer.out_channels],
        )
        # global average pooling
        predicted_latency += self.query(
            "AvgPool2D",
            [fsize, fsize, net.final_expand_layer.out_channels],
            [1, 1, net.final_expand_layer.out_channels],
        )
        # feature mix layer
        predicted_latency += self.query(
            "Conv_2",
            [1, 1, net.feature_mix_layer.in_channels],
            [1, 1, net.feature_mix_layer.out_channels],
        )
        # classifier
        predicted_latency += self.query(
            "Logits", [1, 1, net.classifier.in_features], [net.classifier.out_features]
        )
        return predicted_latency

    def predict_network_latency_given_config(self, net_config, image_size=224):
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [image_size, image_size, 3],
            [
                (image_size + 1) // 2,
                (image_size + 1) // 2,
                net_config["first_conv"]["out_channels"],
            ],
        )
        # blocks
        fsize = (image_size + 1) // 2
        for block in net_config["blocks"]:
            mb_conv = (
                block["mobile_inverted_conv"]
                if "mobile_inverted_conv" in block
                else block["conv"]
            )
            shortcut = block["shortcut"]

            if mb_conv is None:
                continue
            if shortcut is None:
                idskip = 0
            else:
                idskip = 1
            out_fz = int((fsize - 1) / mb_conv["stride"] + 1)
            if mb_conv["mid_channels"] is None:
                mb_conv["mid_channels"] = round(
                    mb_conv["in_channels"] * mb_conv["expand_ratio"]
                )
            block_latency = self.query(
                "expanded_conv",
                [fsize, fsize, mb_conv["in_channels"]],
                [out_fz, out_fz, mb_conv["out_channels"]],
                mid=mb_conv["expand_ratio"],
                ks=mb_conv["kernel_size"],
                stride=mb_conv["stride"],
                id_skip=idskip,
                se=1 if mb_conv["use_se"] else 0,
                h_swish=1 if mb_conv["act_func"] == "h_swish" else 0,
            )
            predicted_latency += block_latency
            fsize = out_fz
        # # final expand layer
        # predicted_latency += self.query(
        #     "Conv_1",
        #     [fsize, fsize, net_config["final_expand_layer"]["in_channels"]],
        #     [fsize, fsize, net_config["final_expand_layer"]["out_channels"]],
        # )
        # # global average pooling
        # predicted_latency += self.query(
        #     "AvgPool2D",
        #     [fsize, fsize, net_config["final_expand_layer"]["out_channels"]],
        #     [1, 1, net_config["final_expand_layer"]["out_channels"]],
        # )
        # feature mix layer
        # predicted_latency += self.query(
        #     "Conv_2",
        #     [1, 1, net_config["feature_mix_layer"]["in_channels"]],
        #     [1, 1, net_config["feature_mix_layer"]["out_channels"]],
        # )
        # classifier
        # predicted_latency += self.query(
        #     "Logits",
        #     [1, 1, net_config["classifier"]["in_features"]],
        #     [net_config["classifier"]["out_features"]],
        # )
        return predicted_latency

    @staticmethod
    def count_flops_given_config(net_config, image_size=224):
        flops = 0
        # first conv
        flops += count_conv_flop(
            (image_size + 1) // 2, 3, net_config["first_conv"]["out_channels"], 3, 1
        )
        # blocks
        fsize = (image_size + 1) // 2
        for block in net_config["blocks"]:
            mb_conv = (
                block["mobile_inverted_conv"]
                if "mobile_inverted_conv" in block
                else block["conv"]
            )
            if mb_conv is None:
                continue
            out_fz = int((fsize - 1) / mb_conv["stride"] + 1)
            if mb_conv["mid_channels"] is None:
                mb_conv["mid_channels"] = round(
                    mb_conv["in_channels"] * mb_conv["expand_ratio"]
                )
            if mb_conv["expand_ratio"] != 1:
                # inverted bottleneck
                flops += count_conv_flop(
                    fsize, mb_conv["in_channels"], mb_conv["mid_channels"], 1, 1
                )
            # depth conv
            flops += count_conv_flop(
                out_fz,
                mb_conv["mid_channels"],
                mb_conv["mid_channels"],
                mb_conv["kernel_size"],
                mb_conv["mid_channels"],
            )
            if mb_conv["use_se"]:
                # SE layer
                se_mid = make_divisible(
                    mb_conv["mid_channels"] // 4, divisor=MyNetwork.CHANNEL_DIVISIBLE
                )
                flops += count_conv_flop(1, mb_conv["mid_channels"], se_mid, 1, 1)
                flops += count_conv_flop(1, se_mid, mb_conv["mid_channels"], 1, 1)
            # point linear
            flops += count_conv_flop(
                out_fz, mb_conv["mid_channels"], mb_conv["out_channels"], 1, 1
            )
            fsize = out_fz
        # # final expand layer
        # flops += count_conv_flop(
        #     fsize,
        #     net_config["final_expand_layer"]["in_channels"],
        #     net_config["final_expand_layer"]["out_channels"],
        #     1,
        #     1,
        # )
        # # feature mix layer
        # flops += count_conv_flop(
        #     7,
        #     net_config["feature_mix_layer"]["in_channels"],
        #     net_config["feature_mix_layer"]["out_channels"],
        #     1,
        #     1,
        # )
        # # classifier
        # flops += count_conv_flop(
        #     1,
        #     net_config["classifier"]["in_features"],
        #     net_config["classifier"]["out_features"],
        #     1,
        #     1,
        # )
        return flops / 1e6  # MFLOPs

if __name__ == "__main__":

    net = MobileNetV2()
    estimator = MBv2LatencyTable()
    # lat = estimator.predict_network_latency(net)
    # print(lat)