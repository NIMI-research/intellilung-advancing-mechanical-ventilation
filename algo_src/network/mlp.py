from typing import List, Literal

from pydantic import BaseModel, field_validator
from torch import nn
from torch.nn import ReLU, Sequential, ELU
import torch

from network.layer import VectorizedLinear


def create_mlp(input_dim: int, hidden_dims: List[int], output_dim: int = None, activation = ReLU,
               output_activation=None, layer_norm_hidden=False, layer_norm_out=False, activation_kwargs=None):
    if activation_kwargs is None:
        activation_kwargs = {}
    layers = torch.nn.Sequential()

    last_layer_out = input_dim
    for dim in hidden_dims:
        layers.append(torch.nn.Linear(last_layer_out, dim))
        if layer_norm_hidden:
            layers.append(nn.LayerNorm(dim))
        layers.append(activation(**activation_kwargs))
        last_layer_out = dim
    if output_dim:
        layers.append(torch.nn.Linear(last_layer_out, output_dim))
        last_layer_out = output_dim
    if layer_norm_out:
        layers.append(nn.LayerNorm(last_layer_out))
    if output_activation:
        layers.append(output_activation())
    return layers


class VectorizedLayerNorm(nn.Module):
    def __init__(self, d: int, k: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.d, self.k, self.eps, self.affine = d, k, eps, affine
        if affine:
            self.weight = nn.Parameter(torch.ones(k, d))
            self.bias = nn.Parameter(torch.zeros(k, d))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [K,B,D] -> norm over D per (K,B)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            y = y * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return y


def create_vectorized_ensemble_mlp(input_dim: int, n_ensembles: int, hidden_dims: List[int], output_dim: int = None,
                                   activation: nn.Module=ReLU, layer_norm_hidden=False, layer_norm_out=False,
                                   activation_kwargs=None):
    if activation_kwargs is None:
        activation_kwargs = {}
    layers = torch.nn.Sequential()

    last_layer_out = input_dim
    for dim in hidden_dims:
        layers.append(VectorizedLinear(last_layer_out, dim, n_ensembles))
        if layer_norm_hidden:
            layers.append(VectorizedLayerNorm(dim, n_ensembles))
        layers.append(activation(**activation_kwargs))
        last_layer_out = dim
    if output_dim:
        layers.append(VectorizedLinear(last_layer_out, output_dim, n_ensembles))
        last_layer_out = output_dim
    if layer_norm_out:
        layers.append(VectorizedLayerNorm(last_layer_out, n_ensembles))
    return layers


def create_layer_norm_mlp(input_dim: int, output_dim: int, hidden_dims: List[int],
                          output_activation=None) -> Sequential:
    """
    MLP which uses LayerNorm (with a tanh normalizer) on the
  first layer and non-linearities (elu) on all but the last remaining layers.

    """

    return Sequential(
        torch.nn.Linear(input_dim, hidden_dims[0]),
        torch.nn.LayerNorm(hidden_dims[0], elementwise_affine=True, bias=True),
        torch.nn.Tanh(),
        create_mlp(input_dim=hidden_dims[0], hidden_dims=hidden_dims[1:], output_dim=output_dim, activation=ELU,
                   output_activation=output_activation)

    )


def create_layer_norm_vec_mlp(input_dim: int, n_ensembles: int, hidden_dims: List[int], output_dim: int = None,
                              activation=ELU) -> Sequential:
    """
    MLP which uses LayerNorm (with a tanh normalizer) on the
  first layer and non-linearities (elu) on all but the last remaining layers.

    """

    return Sequential(
        VectorizedLinear(input_dim, hidden_dims[0], n_ensembles),
        torch.nn.LayerNorm(hidden_dims[0], elementwise_affine=True, bias=True),
        torch.nn.Tanh(),
        create_vectorized_ensemble_mlp(
            input_dim=hidden_dims[0],
            hidden_dims=hidden_dims[1:],
            output_dim=output_dim,
            activation=activation,
            n_ensembles=n_ensembles
        )

    )


ActivationName = Literal[
    "relu", "gelu", "silu", "swish", "tanh", "sigmoid", "leaky_relu", "elu", "mish"
]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,  # aka swish in many libs
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "mish": nn.Mish,
}


class MLPConfig(BaseModel):
    hidden_dims: List[int]
    layer_norm_hidden: bool = False
    layer_norm_output: bool = False
    activation: ActivationName = "relu"
    activation_kwargs: dict = {}

    @field_validator("activation")
    @classmethod
    def normalize_activation(cls, v: str) -> str:
        v = v.lower()
        if v not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{v}'. Options: {sorted(_ACTIVATIONS)}")
        return v

    def get_network(self, input_dim: int, output_dim: int = None) -> nn.Module:
        return create_mlp(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            activation=_ACTIVATIONS[self.activation],
            layer_norm_hidden=self.layer_norm_hidden,
            layer_norm_out=self.layer_norm_output,
            activation_kwargs=self.activation_kwargs
        )


class EnsembleMLPConfig(MLPConfig):
    k_ensemble: int

    def get_activation(self) -> nn.Module:
        act_cls = _ACTIVATIONS[self.activation]
        return act_cls(**self.activation_kwargs)

    def get_network(self, input_dim: int, output_dim: int = None) -> nn.Module:
        return create_vectorized_ensemble_mlp(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            activation=_ACTIVATIONS[self.activation],
            layer_norm_hidden=self.layer_norm_hidden,
            layer_norm_out=self.layer_norm_output,
            activation_kwargs=self.activation_kwargs,
            n_ensembles=self.k_ensemble
        )
