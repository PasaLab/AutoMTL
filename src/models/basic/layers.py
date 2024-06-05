import torch
import torch.nn as nn

from src.models.basic.features import DenseFeature, SparseFeature


def activate_layer(activate_name="relu", emb_dim=None):
    """Construct activation layer.

    Args:
        activate_name (str): name of activation function. Defaults to 'relu'.
        emb_dim (int, optional): used for Dice activation. Defaults to None.

    Returns:
        activation: activation layer
    """
    if activate_name is None:
        return None
    if activate_name == "sigmoid":
        activation = nn.Sigmoid()
    elif activate_name == "tanh":
        activation = nn.Tanh()
    elif activate_name == "relu":
        activation = nn.ReLU()
    elif activate_name == "leakyrelu":
        activation = nn.LeakyReLU()
    elif activate_name == "none":
        activate_name = None
    else:
        raise NotImplementedError(
            f"activation function {activate_name} is not implemented."
        )

    return activation


class EmbeddingLayer(nn.Module):
    """General Embedding Layer.
    We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.

    
    Args:
        features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.

    Shape:
        - Input: 
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output: 
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  # exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False

        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with == None:
                    sparse_emb.append(
                        self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1)
                    )
                else:
                    sparse_emb.append(
                        self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(
                            1
                        )
                    )
            else:
                dense_values.append(
                    x[fea.name].float().unsqueeze(1)
                )  # .unsqueeze(1).unsqueeze(1)

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = torch.cat(
                sparse_emb, dim=1
            )  # [batch_size, num_features, embed_dim]

        if (
            squeeze_dim
        ):  # Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  # only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                return sparse_emb.flatten(
                    start_dim=1
                )  # squeeze dim to : [batch_size, num_features*embed_dim]
            elif dense_exists and sparse_exists:
                return torch.cat(
                    (sparse_emb.flatten(start_dim=1), dense_values), dim=1
                )  # concat dense value with sparse embedding
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                if dense_exists:
                    return (
                        sparse_emb,
                        dense_values,
                    )  # [batch_size, num_features, embed_dim]
                else:
                    return sparse_emb, None
            else:
                raise ValueError(
                    "If keep the original shape:[batch_size, num_features, embed_dim],"
                    " expected %s in feature list, got %s"
                    % ("SparseFeatures", features)
                )


class MLP(nn.Module):
    r"""MLP Layers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activate="relu",
        bn=True,
        init_method=None,
        contain_output_layer=False,
    ):
        super(MLP, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activate = activate
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for i, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activate_func = activate_layer(activate_name=activate)
            if activate_func is not None:
                mlp_modules.append(activate_func)
            mlp_modules.append(nn.Dropout(p=self.dropout))

        if contain_output_layer:
            mlp_modules.append(nn.Linear(self.layers[-1], 1))

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                nn.init.normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_x):
        return self.mlp_layers(input_x)


class FM(nn.Module):
    """Factor Machine

    This block will not use, since its output dimensions are weird.

    Args:
        embedding_dim
        use_first_order: whether use first order features.
    """

    def __init__(self, use_first_order=False, reduce_sum=False):
        super().__init__()

        self._use_first_order = use_first_order
        self._reduce_sum = reduce_sum

        if use_first_order and reduce_sum:
            self._first_linear = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        """
        Args:
            inputs (tensor): shape: [batch_size, nfields, embedding_dim]
                B = batch size, N = num fields, E = embedding dim
        Returns:
            shape: [batch_size, batch_out_dim]
        """
        sum_squared = torch.pow(torch.sum(x, dim=1), 2)  # [B, E]
        squared_sum = torch.sum(torch.pow(x, 2), dim=1)  # [B, E]
        second_order = torch.sub(sum_squared, squared_sum)  # [B, E]

        if self._reduce_sum:
            second_order = torch.sum(second_order, dim=1, keepdim=True)  # [B, 1]
            output = second_order * 0.5
            if self._use_first_order:
                first_out = self._first_linear(x)  # [B, N, 1]
                first_out = torch.sum(first_out, dim=1)  # [B, 1]
                output += first_out
            return output

        output = second_order * 0.5  # [B, E]

        return output
