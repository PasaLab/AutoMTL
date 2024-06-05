import types

import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.nas.layers import *
from src.models.nas.modules import GateFunc


class MixedExpert(BasicUnit):
    def __init__(self, input_dim, n_choices, for_gating=False):
        """Mixed Expert Layer

        Args:
            input_dim (int): dimension of input feature
            n_choices (int): number of experts
            for_gating (bool): True means the output will feed to the gating network
                of next MoE layer.
        """
        super(MixedExpert, self).__init__()

        self.for_gating = for_gating
        self.n_choices = n_choices

        if not for_gating:
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, self.n_choices, bias=False),
                nn.Softmax(dim=1),
            )

        self.beta = Parameter(torch.Tensor(self.n_choices))  # architecture parameters

        self.selection_gate = GateFunc.apply

        self.in_warmup = True

    def set_chosen_op_active(self):
        self.in_warmup = False

    def forward(self, xs):
        """
        Args:
            xs (List[Tensor]): List of input tensor from mixed feature module (after experts)
                the last one is for gating network
        """
        expert_selection = self.selection_gate(F.sigmoid(self.beta), self.in_warmup)

        expert_outputs = xs[:-1]

        if self.for_gating:
            return sum(
                [expert_outputs[i] * expert_selection[i] for i in range(self.n_choices)]
            )

        gate_input = xs[-1]
        gate_value = self.gate_network(gate_input)

        out = 0
        for i in range(self.n_choices):
            out += (
                gate_value[:, i].unsqueeze(1) * expert_outputs[i] * expert_selection[i]
            )

        return out

    @property
    def module_str(self):
        select_probs = F.sigmoid(self.beta.detach()).cpu().numpy()
        return f"Mixed expert select probs: {select_probs}"

    @property
    def config(self):
        raise ValueError("not needed")

    @staticmethod
    def build_from_config(config):
        raise ValueError("not needed")

    def get_flops(self, x):
        """Only active paths taken into consideration when calculating FLOPs"""
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.experts[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    def export_arch(self, beta=None):
        def _forward(self, xs):
            expert_outputs = xs[:-1]

            if self.for_gating:
                return sum(
                    [expert_outputs[i] * self.beta[i] for i in range(self.n_choices)]
                )

            gate_input = xs[-1]
            gate_value = self.gate_network(gate_input)

            out = 0
            for i in range(self.n_choices):
                out += gate_value[:, i].unsqueeze(1) * expert_outputs[i] * self.beta[i]

            return out

        if beta:
            self.beta.data = torch.tensor(beta)
        else:
            self.beta.data = (F.sigmoid(self.beta.data) > 0.5).float()  
        self.beta.requires_grad = False

        self.forward = types.MethodType(_forward, self)

        return self.beta.tolist()
