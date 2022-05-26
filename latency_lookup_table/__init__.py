# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import copy
from .latency_lookup_table import MBv2LatencyTable
from .latency_lookup_table import *


class BaseEfficiencyModel:
    def __init__(self, ofa_net):
        self.ofa_net = ofa_net

    def get_active_subnet_config(self, arch_dict):
        arch_dict = copy.deepcopy(arch_dict)
        image_size = arch_dict.pop("image_size")
        self.ofa_net.set_active_subnet(**arch_dict)
        active_net_config = self.ofa_net.get_active_net_config()
        return active_net_config, image_size

    def get_efficiency(self, arch_dict):
        raise NotImplementedError


class Mbv2FLOPsModel(BaseEfficiencyModel):
    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        return MBv2LatencyTable.count_flops_given_config(active_net_config, image_size)


class Mbv2LatencyModel(BaseEfficiencyModel):
    def __init__(self, ofa_net, lookup_table_path_dict):
        super(MBv2LatencyTable, self).__init__(ofa_net)
        self.latency_tables = {}
        for image_size, path in lookup_table_path_dict.items():
            self.latency_tables[image_size] = MBv2LatencyTable(
                local_dir="/tmp/.ofa_latency_tools/",
                url=os.path.join(path, "%d_lookup_table.yaml" % image_size),
            )

    def get_efficiency(self, arch_dict):
        active_net_config, image_size = self.get_active_subnet_config(arch_dict)
        return self.latency_tables[image_size].predict_network_latency_given_config(
            active_net_config, image_size
        )
