import numpy as np
import torch.nn.functional as F

from src.models.nas.layers import *


def build_candidate_ops(
    candidate_ops, in_features, out_features, ops_order="weight_bn_act", dropout=0
):
    if candidate_ops is None:
        raise ValueError("please specify a candidate set")

    name2ops = {
        "Identity": lambda in_F, out_F: IdentityLayer(in_F, out_F, ops_order=ops_order),
        "Zero": lambda in_F, out_F: ZeroLayer(in_F, out_F),
        "MLP-16": lambda in_F, out_F: MLP(
            in_F, out_F, hidden_size=16, dropout_rate=dropout
        ),
        "MLP-32": lambda in_F, out_F: MLP(
            in_F, out_F, hidden_size=32, dropout_rate=dropout
        ),
        "MLP-64": lambda in_F, out_F: MLP(
            in_F, out_F, hidden_size=64, dropout_rate=dropout
        ),
        "MLP-128": lambda in_F, out_F: MLP(
            in_F, out_F, hidden_size=128, dropout_rate=dropout
        ),
        "MLP-256": lambda in_F, out_F: MLP(
            in_F, out_F, hidden_size=256, dropout_rate=dropout
        ),
        "MLP-512": lambda in_F, out_F: MLP(
            in_F, out_F, hidden_size=512, dropout_rate=dropout
        ),
        "MLP-1024": lambda in_F, out_F: MLP(
            in_F, out_F, hidden_size=1024, dropout_rate=dropout
        ),
    }

    return [name2ops[name](in_features, out_features) for name in candidate_ops]


class MixedOp(BasicUnit):
    def __init__(self, candidate_ops):
        super(MixedOp, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.alpha = nn.Parameter(
            torch.Tensor(self.n_choices)
        )  # architecture parameters

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    def is_zero_layer(self):
        return self.chosen_op.is_zero_layer()

    def forward(self, x):
        op_results = torch.stack([op(x) for op in self.candidate_ops])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * self.probs_over_ops.reshape(*alpha_shape), dim=0)

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return "MixOp(%s, %.3f)" % (self.candidate_ops[chosen_index].module_str, probs)

    @property
    def config(self):
        raise ValueError("not needed")

    @staticmethod
    def build_from_config(config):
        raise ValueError("not needed")

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(torch.mul(probs, log_probs)).item()
        return entropy

    def get_flops(self, x):
        """Only active paths taken into consideration when calculating FLOPs"""
        flops = 0
        flops += self.chosen_op.get_flops(x)
        return flops, self.forward(x)

    def discretize(self, chosen_idx=None):
        """Discretization, keeping only one operator"""
        if chosen_idx is not None:
            return self.candidate_ops[chosen_idx]
        return self.chosen_index[0], self.chosen_op
