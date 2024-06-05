import types

import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.nas.layers import *
from src.models.nas.modules import GateFunc


class MixFeature(BasicUnit):
    def __init__(self, n_fields, interaction_layer):
        """
        Args:
            n_fields (int): number of sparse feature fields
            interaction_layer (nn.Module): interaction layer
        """
        super(MixFeature, self).__init__()

        self.n_fields = n_fields
        self.interaction_layer = interaction_layer

        # architecture parameters for single feature selection
        self.single_beta = Parameter(torch.Tensor(self.n_fields))
        # architecture parameters for interactin feature selection
        self.inter_beta = Parameter(torch.Tensor(self.n_fields))

        self.selection_gate = GateFunc.apply

        self.in_warmup = True

    @property
    def chosen_indexes(self):
        single_probs = F.sigmoid(self.single_beta).cpu().detach().numpy()
        inter_probs = F.sigmoid(self.inter_beta).cpu().detach().numpy()
        single_indexes = np.where(single_probs > 0.6)[0].tolist()
        inter_indexes = np.where(inter_probs > 0.6)[0].tolist()
        return single_indexes, inter_indexes

    def set_chosen_op_active(self):
        self.in_warmup = False

    def forward(self, embs, dense_fea):
        """
        Args:
            embs (Tensor): features after embedding
                [batch size, nfields, embedding size]
            dense_fea (Tensor): dense features.
                [batch size, n_dense_fields]

        Returns:
            out (Tensor): feature after selection and interaction
                [batch size, (nfields + 1) * embedding size + n_dense_fields]
        """
        single_selection = self.selection_gate(
            F.sigmoid(self.single_beta), self.in_warmup
        ).view(1, self.n_fields, 1)
        inter_selection = self.selection_gate(
            F.sigmoid(self.inter_beta), self.in_warmup
        ).view(1, self.n_fields, 1)
        single_feature = single_selection * embs

        inter_feature = self.interaction_layer(inter_selection * embs)

        single_feature = single_feature.view(single_feature.size(0), -1)

        if dense_fea is not None:
            return torch.cat((single_feature, inter_feature, dense_fea), dim=1)
        else:
            return torch.cat((single_feature, inter_feature), dim=1)

    @property
    def module_str(self):
        single_probs = F.sigmoid(self.single_beta).cpu().detach().numpy()
        inter_probs = F.sigmoid(self.inter_beta).cpu().detach().numpy()
        return (
            f"Single-fea select probs: {single_probs}, "
            f"Inter-fea select probs: {inter_probs}"
        )

    @property
    def config(self):
        raise ValueError("not needed")

    @staticmethod
    def build_from_config(config):
        raise ValueError("not needed")

    def export_arch(self, single_beta=None, inter_beta=None):
        def _forward(self, embs, dense_fea):
            single_feature = self.single_beta.view(1, self.n_fields, 1) * embs
            inter_feature = self.interaction_layer(
                self.inter_beta.view(1, self.n_fields, 1) * embs
            )

            single_feature = single_feature.view(single_feature.size(0), -1)

            if dense_fea is not None:
                return torch.cat((single_feature, inter_feature, dense_fea), dim=1)
            else:
                return torch.cat((single_feature, inter_feature), dim=1)

        if single_beta is None or inter_beta is None:
            self.single_beta.data = (F.sigmoid(self.single_beta.data) > 0.5).float()
            self.inter_beta.data = (F.sigmoid(self.inter_beta.data) > 0.5).float()
        else:
            self.single_beta.data = torch.tensor(single_beta)
            self.inter_beta.data = torch.tensor(single_beta)
        self.single_beta.requires_grad = False
        self.inter_beta.requires_grad = False

        self.forward = types.MethodType(_forward, self)

        return self.single_beta.tolist(), self.inter_beta.tolist()
