from collections import OrderedDict

import torch
import torch.nn as nn


class BasicUnit(nn.Module):
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


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MLP.__name__: MLP,
    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class BasicLayer(BasicUnit):
    def __init__(
        self,
        in_features,
        out_features,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(BasicLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm1d(in_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = None
        # activation
        if act_func == "relu":
            if self.ops_list[0] == "act":
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == "tanh":
            self.activation = nn.Tanh()
        elif act_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == "weight":
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == "bn":
                if self.bn is not None:
                    x = self.bn(x)
            elif op == "act":
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError("Unrecognized op: %s" % op)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class IdentityLayer(BasicLayer):
    def __init__(
        self,
        in_features,
        out_features,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(IdentityLayer, self).__init__(
            in_features, out_features, use_bn, act_func, dropout_rate, ops_order
        )
        if self.in_features != self.out_features:
            self.projection = nn.Linear(in_features, out_features, bias=False)

    def weight_call(self, x):
        if self.in_features != self.out_features:
            x = self.projection(x)
        return x

    @property
    def module_str(self):
        if self.in_features == self.out_features:
            return "Identity"
        else:
            return f"{self.in_features}->{self.out_features}_Project"

    @property
    def config(self):
        config = {
            "name": IdentityLayer.__name__,
        }
        config.update(super(IdentityLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class LinearLayer(BasicLayer):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(LinearLayer, self).__init__(
            in_features, out_features, use_bn, act_func, dropout_rate, ops_order
        )

        self.bias = bias
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    def weight_call(self, x):
        return self.linear(x)

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        config = {
            "name": LinearLayer.__name__,
            "bias": self.bias,
        }
        config.update(super(LinearLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(BasicUnit):
    def __init__(self, in_features, out_features):
        super(ZeroLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor):
        out_shape = list(x.shape)
        out_shape[-1] = self.out_features
        padding = torch.zeros(*out_shape, device=x.device)
        padding = torch.autograd.Variable(padding, requires_grad=False)
        return padding

    @property
    def module_str(self):
        return "Zero"

    @property
    def config(self):
        return {"name": ZeroLayer.__name__}

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True


class MLP(BasicLayer):
    """Multple Layer Perceptron"""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_size,
        use_bn=False,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(MLP, self).__init__(
            in_features, out_features, use_bn, act_func, dropout_rate, ops_order
        )

        self.hidden_size = hidden_size

        # activation
        if act_func == "relu":
            self.hidden_activation = nn.ReLU()
        elif act_func == "tanh":
            self.hidden_activation = nn.Tanh()
        elif act_func == "sigmoid":
            self.hidden_activation = nn.Sigmoid()
        else:
            raise ValueError

        self._hidden_linear = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.hidden_activation,
        )

        self.out_projection = nn.Linear(hidden_size, out_features)

    def weight_call(self, x):
        x = self._hidden_linear(x)
        x = self.out_projection(x)
        return x

    @property
    def module_str(self):
        return f"{self.in_features}->{self.hidden_size}->{self.out_features}_MLP"

    @property
    def config(self):
        config = {"name": MLP.__name__, "hidden_size": self.hidden_size}
        config.update(super(MLP, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return MLP(**config)

    def get_flops(self, x):
        return (self.in_features + self.out_features) * self.hidden_size, self.forward(
            x
        )

    @staticmethod
    def is_zero_layer():
        return False
