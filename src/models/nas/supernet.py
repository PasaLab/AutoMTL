import copy

import torch
import torch.nn as nn

from src.models.basic.layers import FM, MLP, EmbeddingLayer
from src.models.nas import BasicNetwork, ExpertModule, MixedExpert, MixFeature, MixedOp


class SuperNet(BasicNetwork):
    def __init__(
        self,
        features,
        embedding_dim,
        task_types,
        n_experts,
        n_expert_layers,
        n_layers,
        in_features,
        out_features,
        tower_layers,
        dropout,
        expert_candidate_ops,
    ):
        """
        Args:
            features (list): the list of `Feature Class`
            embedding_dim (int): ...
            task_types (List[str]): list of task types.
            n_experts (int): number of experts per MoE layer.
            n_expert_layers (int): number of MoE layers.
            n_layers (int): number of layers per expert.
            in_features (int): block input feature dimension.
            out_features (int): block output feature dimension.
            dropout (float): dropout ratio.
            tower_layers (List[int]): hidden sizes of tower layers.
            expert_candidate_ops (List[str]): ...
        """
        super().__init__()

        self._exported_arch = dict()

        self._redundant_modules = None
        self._unused_modules = None

        # embedding layer and feature preprocessing
        self.features = features
        self.embedding = EmbeddingLayer(features)
        self.embedding_dim = embedding_dim
        self.n_feilds = len(self.embedding.embed_dict)
        self.n_tasks = len(task_types)

        self.input_dim = (
            self.embedding_dim * (self.n_feilds + 1) + self.embedding.n_dense
        )

        self.n_experts = n_experts
        self.n_expert_layers = n_expert_layers

        self.interaction_layer = FM()  # [bs, nfields, emb] -> [bs, emb]
        self.feature_modules = nn.ModuleList(
            [
                MixFeature(self.n_feilds, self.interaction_layer)
                for _ in range(self.n_experts)
            ]
        )   # add another MixFeature for gating network

        # multi-layer experts
        expert_input_dims = [self.input_dim] + [out_features] * (
            self.n_expert_layers - 1
        )  # input dimensions of mix of experts layers
        self.experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ExpertModule(
                            input_dim,
                            in_features,
                            out_features,
                            num_layers=n_layers,
                            candidate_ops=expert_candidate_ops,
                            dropout=dropout,
                        )
                        for _ in range(self.n_experts)
                    ]
                )
                for input_dim in expert_input_dims
            ]
        )

        # set mixed expert module parameters
        # The gating network was putted outside the Mixed Expert module
        self.mixed_experts = nn.ModuleList(
            nn.ModuleList(
                [
                    MixedExpert(self.input_dim, n_choices=self.n_experts)
                    for _ in range(
                        self.n_experts if i < self.n_expert_layers - 1
                        else self.n_tasks
                    )
                ]
            )
            for i in range(self.n_expert_layers)
        )

        self.towers = nn.ModuleList(
            [
                MLP(
                    [out_features] + tower_layers,
                    dropout=dropout,
                    contain_output_layer=True,
                )
                for _ in range(self.n_tasks)
            ]
        )

    def forward(self, x):
        embs, dense_fea = self.embedding(
            x, self.features, squeeze_dim=False
        )  # [B, N, E], [B, n_dense_fields]
        mix_features = [
            feature_module(embs, dense_fea) for feature_module in self.feature_modules
        ]
        mix_features.append(
            nn.functional.pad(embs.view(embs.size(0), -1), (0, self.embedding_dim + self.embedding.n_dense))
        )       # raw feature input for gate
        
        temp = []
        for i in range(self.n_expert_layers - 1):
            for j in range(self.n_experts):
                mix_features[j] = self.experts[i][j](mix_features[j])   # len(mix_features) = n+1
            for j in range(self.n_experts):
                temp.append(self.mixed_experts[i][j](mix_features))     # len(temp) = n+1
            temp.append(mix_features[-1])
            mix_features = temp
            temp = []
            
        for j in range(self.n_experts):
            mix_features[j] = self.experts[-1][j](mix_features[j])
        for i in range(self.n_tasks):
            temp.append(self.mixed_experts[-1][i](mix_features))
        mix_features = temp

        outs = [self.towers[i](mix_features[i]) for i in range(self.n_tasks)]
        outs = torch.cat(outs, dim=1)

        return outs

    @property
    def exported_arch(self):
        return self._exported_arch

    @staticmethod
    def build_from_config(config):
        raise ValueError("not needed")

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if "alpha" in name or "beta" in name:
                yield param

    def alpha_parameters(self):
        for name, param in self.named_parameters():
            if "alpha" in name:
                yield param

    def beta_parameters(self):
        for name, param in self.named_parameters():
            if "beta" in name:
                yield param

    def weight_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if "alpha" not in name and "beta" not in name:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                if "alpha" not in name and "beta" not in name:
                    flag = False
                    for key in keys:
                        if key in name:
                            flag = True
                            break
                    if flag:
                        yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                if "alpha" not in name and "beta" not in name:
                    flag = True
                    for key in keys:
                        if key in name:
                            flag = False
                            break
                    if flag:
                        yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def all_parameters(self):
        for _, param in self.named_parameters():
            yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith("MixedFeature") or m.__str__().startswith(
                    "MixedExpert"
                ):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def init_arch_params(self, init_type="normal", init_ratio=1e-3):
        for param in self.alpha_parameters():
            if init_type == "normal":
                param.data.normal_(0, init_ratio)
            elif init_type == "uniform":
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

        for param in self.beta_parameters():
            param.data.zero_()
            # param.data.fill_(4.6)

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            print(m, level="debug")
            unused = {}
            involved_index = m.active_index + m.inactive_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), " do not support `set_chosen_op_active()`")

    def set_active_via_net(self, net):
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def discretize_one_op(self):
        mix_ops = []
        for name, module in self.named_modules():
            if isinstance(module, MixedOp):
                mix_ops.append((name, module, module.entropy()))

        if len(mix_ops) == 0:
            print("There are no mixed ops to be discretized.")
            return

        print(f"Remainding {len(mix_ops)} mix ops.")

        op_name, discretize_op, _ = min(mix_ops, key=lambda x: x[2])
        tokens = op_name.split(".")
        pre_name = ".".join(tokens[:-1])
        cur_module = self.get_submodule(pre_name)
        idx, op = discretize_op.discretize()
        self._exported_arch[op_name] = idx
        cur_module.add_module(tokens[-1], op)

    def export_architecture(self):
        mix_ops = []
        for name, module in self.named_modules():
            if isinstance(module, MixedOp):
                mix_ops.append((name, module))

        print(f"Remainding {len(mix_ops)} mix ops.")

        if len(mix_ops) > 0:
            for op_name, discretize_op in mix_ops:
                tokens = op_name.split(".")
                pre_name = ".".join(tokens[:-1])
                cur_module = self.get_submodule(pre_name)
                idx, op = discretize_op.discretize()
                self._exported_arch[op_name] = idx
                cur_module.add_module(tokens[-1], op)

        print("Export mixed feature and mixed expert.")
        for name, module in self.named_modules():
            if isinstance(module, (MixFeature, MixedExpert)):
                self._exported_arch[name] = module.export_arch()

    def convert_to_normal_net(self, arch_config):
        """Covert a supernet to normal net, used in the revoer/retain stage

        Args:
            arch_config (dict): selection of each mixed op/feature/expert
        """
        for name, module in self.named_modules():
            if isinstance(module, MixedOp):
                op = module.discretize(chosen_idx=arch_config[name])
                tokens = name.split(".")
                pre_name = ".".join(tokens[:-1])
                cur_module = self.get_submodule(pre_name)
                cur_module.add_module(tokens[-1], op)
            elif isinstance(module, MixFeature):
                module.export_arch(*arch_config[name])
            elif isinstance(module, MixedExpert):
                module.export_arch(arch_config[name])


    def print_arch(self, epoch_idx):
        print(f"Epoch-({epoch_idx}): " + "-" * 30 + f"Current Architecture" + "-" * 30)
        for i, m in enumerate(self.feature_modules):
            print(f"Epoch-({epoch_idx}): " + f"Mixed Feature {i}, {m.module_str}\t")
        for i, es in enumerate(self.experts):
            for j, e in enumerate(es):
                print(f"Epoch-({epoch_idx}): " + f"Expert {i}, {j}: {e.module_str}\t")
        for i, ms in enumerate(self.mixed_experts):
            for j, m in enumerate(ms):
                print(f"Epoch-({epoch_idx}): " + f"Mixed Expert {i}, {m.module_str}\t")
        print("-" * 60)
