from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, get_laplacian

from ..inits import glorot, zeros


class EnhancedGATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 K,
                 normalization='sym',
                 cached=True,
                 negative_slope: float = 0.2,
                 bias: bool = True,
                 dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(EnhancedGATConv, self).__init__(node_dim=0, **kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cached = cached
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # --------------------------------------------------------------------------------------------------------------
        self.heads = heads = 1
        self.negative_slope = negative_slope
        self.dropout = dropout

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin_src = Linear(in_channels, heads * out_channels,
                              bias=False, weight_initializer='glorot')
        # self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, 1, 1, out_channels))
        # self.att_dst = Parameter(torch.Tensor(1, 1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # chebyshev convolution
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

        # graph attention
        self.lin_src.reset_parameters()
        # self.lin_dst.reset_parameters()
        glorot(self.att_src)
        # glorot(self.att_dst)

    @staticmethod
    def norm(edge_index, num_nodes, edge_attr, normalization, lambda_max, dtype=None, batch=None):
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        edge_index, edge_weight = get_laplacian(edge_index, edge_attr,
                                                normalization, dtype,
                                                num_nodes)

        if batch is not None and torch.is_tensor(lambda_max):
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight[edge_weight == float('inf')] = 0

        # edge_index, edge_weight = add_self_loops(edge_index,
        #                                          edge_weight,
        #                                          fill_value=-1,
        #                                          num_nodes=num_nodes)

        return edge_index, edge_weight

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch=None,
                lambda_max=None,
                **kwargs):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        lambda_max = 2.0 if lambda_max is None else lambda_max

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index,
                                         x.size(1),
                                         edge_attr,
                                         self.normalization,
                                         lambda_max,
                                         dtype=x.dtype,
                                         batch=batch)

            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        # --------------------------------------------------------------------------------------------------------------
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        assert isinstance(x, Tensor) and len(x.shape) == 3

        num_batches, num_nodes = x.shape[0], x.shape[1]
        # x_src = x_dst = self.lin_src(x).view(num_batches, num_nodes, H, C)
        x_src = self.lin_src(x).view(num_batches, num_nodes, H, C)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x * self.att_src).sum(-1)

        # x = (x_src.permute(1, 2, 0, 3), x_dst.permute(1, 2, 0, 3))
        alpha = (alpha_src.permute(1, 2, 0), None)

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=None)

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])  # (num_nodes)(torch.Tensor(K, in_channels, out_channels))

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, alpha=alpha, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, alpha=alpha, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        out = out.permute(2, 0, 1, 3).sqeeze(2)
        print(out.shape)
        exit(0)
        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(self, alpha_j: Tensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def message(self, x_j: Tensor, alpha: Tensor, norm: Tensor) -> Tensor:
        print(x_j.shape, alpha.shape, norm.shape)  # torch.Size([1090, 166, 32]) torch.Size([1090, 166, 1]) torch.Size([1090])

        return norm.reshape(-1, 1, 1) * alpha.unsqueeze(-1) * x_j

    # def message(self, x_j, norm):
    #     return norm.view(1, -1, 1) * x_j  # torch.Size([1, 1256, 1]) torch.Size([16, 1256, 32])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
