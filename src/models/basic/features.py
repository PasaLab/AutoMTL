import torch


class SparseFeature:
    """The Feature Class for Sparse feature.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(
        self, name, vocab_size, embed_dim=16, shared_with=None, padding_idx=None
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.shared_with = shared_with
        self.padding_idx = padding_idx

    def __repr__(self):
        return (
            f"<SparseFeature {self.name} with Embedding shape ({self.vocab_size},"
            f" {self.embed_dim})>"
        )

    def get_embedding_layer(self):
        if not hasattr(self, "embed"):
            self.embed = torch.nn.Embedding(self.vocab_size, self.embed_dim)
        return self.embed


class DenseFeature:
    """The Feature Class for Dense feature.

    Args:
        name (str): feature's name.
        embed_dim (int): embedding vector's length, the value fixed `1`.
    """

    def __init__(self, name):
        self.name = name
        self.embed_dim = 1

    def __repr__(self):
        return f"<DenseFeature {self.name}>"
