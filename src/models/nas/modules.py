from src.models.nas.mix_op import *
import math


class GateFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, in_warmup):
        if in_warmup:
            return torch.ones_like(input)
        if torch.enable_grad():
            return torch.bernoulli(input)
        else:
            return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    

class BasicNetwork(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    "momentum": m.momentum,
                    "eps": m.eps,
                }
        return None

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_parameters(self):
        return self.get_parameters()


class ExpertModule(BasicUnit):
    def __init__(
        self, input_dim, in_features, out_features, num_layers, candidate_ops, dropout=0
    ):
        """The Expert Module.

        Args:
            input_dim (int): input feature dimension
            in_features (int|List[int]): input dimension
            out_features (int|List[int]): output dimension
            num_layers (int): number of layers
            candidate_ops (List[str]): candidate operations
            dropout (float): dropout rate.
        """
        super(ExpertModule, self).__init__()

        if isinstance(in_features, list):
            self.in_features = [input_dim] + in_features
        else:
            self.in_features = [input_dim] + [in_features] * (num_layers - 1)

        if isinstance(out_features, list):
            self.out_features = out_features
        else:
            self.out_features = [out_features] * num_layers

        self.num_layers = num_layers

        blocks = []
        for i in range(num_layers):
            op = MixedOp(
                candidate_ops=build_candidate_ops(
                    candidate_ops,
                    self.in_features[i],
                    self.out_features[i],
                    dropout=dropout,
                )
            )
            blocks.append(op)

        self.blocks = nn.ModuleList(blocks)
        

    def forward(self, x):
        """
        Args:
            x (Tensor): List of feature tensors

        Returns:
            out (Tensor)
        """
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def module_str(self):
        ret = "Expert module: "
        for block in self.blocks:
            ret += block.module_str + " | "
        return ret

    @property
    def config(self):
        raise ValueError("not needed")

    @staticmethod
    def build_from_config(config):
        raise ValueError("not needed")
