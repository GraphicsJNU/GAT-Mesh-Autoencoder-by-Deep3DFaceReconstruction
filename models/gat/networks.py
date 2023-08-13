import torch
from torch import Tensor, index_select
from torch.nn import Module, ModuleList, Linear, ELU, init, BatchNorm1d, Conv2d, Sequential, ReLU, Flatten, AvgPool2d
import torch.nn.functional as F
from models.gat.conv import ChebConv, GATConv
import torchvision.models as models
import torchvision.models._utils as _utils

from scipy.io import loadmat

import numpy as np
from torch_scatter import scatter_add

from models.gat.utils import utils


def pool(x, transform, dim=1):
    row, col = transform._indices()
    value = transform._values().unsqueeze(-1)
    x = index_select(x, dim, col) * value
    x = scatter_add(x, row, dim, dim_size=transform.size(0))

    return x


class Encoder(Module):
    def __init__(self, filters: list, edge_index_list: list, down_transform_list: list,
                 latent_channels: int = 8, k: int = 6, lambda_max: float = 2.0, conv: str = 'cheb'):
        super(Encoder, self).__init__()

        self.filters = filters
        self.edge_index_list = edge_index_list
        self.down_transform_list = down_transform_list
        self.lambda_max = lambda_max
        self.gcn_cnt = len(filters) - 2
        self.conv_list = ModuleList()
        self.batchnorm_list = ModuleList()
        for i in range(self.gcn_cnt):
            s, e = i, i + 1
            if conv == 'cheb':
                self.conv_list.append(ChebConv(filters[s], filters[e], K=k))
            elif conv == 'gat':
                self.conv_list.append(GATConv(filters[s], filters[e], heads=k))
            else:
                raise RuntimeError('You can use only cheb or gat or arma.')

            self.batchnorm_list.append(BatchNorm1d(down_transform_list[i].shape[1]))

        self.activation = ELU()
        self.fc = Linear(down_transform_list[-1].size(0) * filters[-1], latent_channels)

        self.reset_parameters()

    def forward(self, x):
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x, self.edge_index_list[i], lambda_max=self.lambda_max)
            x = self.batchnorm_list[i](x)
            x = self.activation(x)
            x = pool(x, self.down_transform_list[i])

        x = x.view(x.shape[0], self.down_transform_list[-1].size(0) * self.filters[-1])
        x = self.fc(x)

        return x

    def reset_parameters(self):
        for conv in self.conv_list:
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0)
                else:
                    init.xavier_uniform_(param)

        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)


class DecoderLayer(Module):
    def __init__(self, filters: list, edge_index_list: list, up_transform_list: list,
                 latent_channels: int = 8, k: int = 6, lambda_max: float = 2.0, conv: str = 'cheb'):
        super(DecoderLayer, self).__init__()

        self.filters = filters
        self.edge_index_list = edge_index_list
        self.up_transform_list = up_transform_list
        self.lambda_max = lambda_max
        self.gcn_cnt = len(filters) - 2
        self.fc = Linear(latent_channels, up_transform_list[-1].size(1) * filters[-1])
        self.conv_list = ModuleList()
        self.batchnorm_list = ModuleList()
        for i in range(self.gcn_cnt):
            s, e = -i - 1, -i - 2
            if conv == 'cheb':
                self.conv_list.append(ChebConv(filters[s], filters[e], K=k))
                self.recon_conv = ChebConv(filters[1], filters[0], k)
            elif conv == 'gat':
                self.conv_list.append(GATConv(filters[s], filters[e], heads=k))
                self.recon_conv = GATConv(filters[1], filters[0], heads=k)
            else:
                raise RuntimeError('You can use only cheb or gat or arma.')

            self.batchnorm_list.append(BatchNorm1d(up_transform_list[self.gcn_cnt - i - 1].shape[0]))

        self.activation = ELU()

        self.reset_parameters()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.up_transform_list[-1].size(1), self.filters[-1])
        for i in range(len(self.conv_list)):
            x = pool(x, self.up_transform_list[self.gcn_cnt - i - 1])
            x = self.conv_list[i](x, self.edge_index_list[self.gcn_cnt - i - 1], lambda_max=self.lambda_max)
            x = self.batchnorm_list[i](x)
            x = self.activation(x)
        x = self.recon_conv(x, self.edge_index_list[0])

        return x

    def reset_parameters(self):
        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)

        for conv in self.conv_list:
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0)
                else:
                    init.xavier_uniform_(param)

        for name, param in self.recon_conv.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)


class AutoEncoder(Module):
    def __init__(self, feature_channels: int, latent_channels: int, a: list, d: list, u: list, k: int = 6,
                 lambda_max: float = 2.3, enc_conv='cheb', dec_conv='cheb', device='cpu'):
        super(AutoEncoder, self).__init__()

        self.filters = [feature_channels, 16, 16, 16, 32, 32]

        edge_index_list = [utils.to_edge_index(adj).to(device) for adj in a]
        down_transform_list = [utils.to_sparse(down_transform).to(device) for down_transform in d]
        up_transform_list = [utils.to_sparse(up_transform).to(device) for up_transform in u]

        self.encoder = Encoder(self.filters, edge_index_list, down_transform_list, latent_channels, k,
                               lambda_max=lambda_max, conv=enc_conv).to(device)
        self.decoder = DecoderLayer(self.filters, edge_index_list, up_transform_list, latent_channels, k,
                                    lambda_max=lambda_max, conv=dec_conv).to(device)

    def forward(self, x, is_vec=False):
        if x.shape[-1] != self.filters[0]:
            raise Exception('(batch_size, vertex_cnt, 3)이 아닌 데이터는 처리할 수 없습니다.')

        encoded_vec = self.encoder(x)
        pred = self.decoder(encoded_vec)

        if is_vec:
            return pred, encoded_vec
        else:
            return pred


class RefineAutoEncoder(Module):
    def __init__(self, feature_channels: int, a: list, d: list, u: list, k: int = 6, lambda_max: float = 2.3,
                 enc_conv='cheb', dec_conv='cheb', device='cpu'):
        super(RefineAutoEncoder, self).__init__()

        self.filters = [feature_channels, 8, 16, 32]

        self.edge_index_list = [utils.to_edge_index(adj).to(device) for adj in a]
        self.down_transform_list = [utils.to_sparse(down_transform).to(device) for down_transform in d]
        self.up_transform_list = [utils.to_sparse(up_transform).to(device) for up_transform in u]

        self.lambda_max = lambda_max
        self.conv_list = ModuleList()
        self.gcn_cnt = len(self.filters) - 1
        for i in range(self.gcn_cnt):
            s, e = i, i + 1
            if enc_conv == 'cheb':
                self.conv_list.append(ChebConv(self.filters[s], self.filters[e], K=k))
            elif enc_conv == 'gat':
                self.conv_list.append(GATConv(self.filters[s], self.filters[e], heads=k))
            else:
                raise RuntimeError('You can use only cheb or gat or arma.')

        for i in range(self.gcn_cnt, 1, -1):
            s, e = i, i - 1
            if dec_conv == 'cheb':
                self.conv_list.append(ChebConv(self.filters[s], self.filters[e], K=k))
                self.recon_conv = ChebConv(self.filters[1], self.filters[0] // 2, k)
            elif dec_conv == 'gat':
                self.conv_list.append(GATConv(self.filters[s], self.filters[e], heads=k))
                self.recon_conv = GATConv(self.filters[1], self.filters[0] // 2, heads=k)
            else:
                raise RuntimeError('You can use only cheb or gat or arma.')


        self.activation = ELU()

        self.reset_parameters()

    def forward(self, x):
        if x.shape[-1] != self.filters[0]:
            raise Exception('(batch_size, vertex_cnt, 6)이 아닌 데이터는 처리할 수 없습니다.')

        for i in range(self.gcn_cnt):
            x = self.conv_list[i](x, self.edge_index_list[i], lambda_max=self.lambda_max)
            x = self.activation(x)
            if i:
                x = pool(x, self.down_transform_list[i - 1])

        for i in range(self.gcn_cnt - 1):
            x = self.conv_list[self.gcn_cnt + i](x, self.edge_index_list[self.gcn_cnt - i], lambda_max=self.lambda_max)
            x = self.activation(x)
            x = pool(x, self.up_transform_list[self.gcn_cnt - i])

        pred = self.recon_conv(x, self.edge_index_list[0])

        return pred

    def reset_parameters(self):
        for conv in self.conv_list:
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0)
                else:
                    init.xavier_uniform_(param)

        for name, param in self.recon_conv.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)


# https://discuss.pytorch.org/t/variational-auto-encoder-not-able-to-reconstruct-the-surveillance-image/176472
class VariationalAutoEncoder(Module):
    def __init__(self, feature_channels: int, latent_channels: int, a: list,
                 d: list, u: list, enc_k: int = 6, dec_k: int = 6, lambda_max: float = 2.3,
                 enc_conv='cheb', dec_conv='cheb', device='cpu'):
        super(VariationalAutoEncoder, self).__init__()

        self.filters = [feature_channels, 16, 16, 16, 32, 32]
        self.device = device
        edge_index_list = [utils.to_edge_index(adj).to(device) for adj in a]
        down_transform_list = [utils.to_sparse(down_transform).to(device) for down_transform in d]
        up_transform_list = [utils.to_sparse(up_transform).to(device) for up_transform in u]

        self.encoder = Encoder(self.filters, edge_index_list, down_transform_list, latent_channels * 2,
                               enc_k, lambda_max=lambda_max, conv=enc_conv).to(device)
        self.decoder = DecoderLayer(self.filters, edge_index_list, up_transform_list, latent_channels,
                                    dec_k, lambda_max=lambda_max, conv=dec_conv).to(device)

        self.mean_fc = Linear(latent_channels * 2, latent_channels).to(device)
        self.log_var_fc = Linear(latent_channels * 2, latent_channels).to(device)

        self.reset_parameters()

    def _parameterize(self, x):
        mean = self.mean_fc(x)
        log_var = self.log_var_fc(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z, mean, log_var

    def forward(self, x, is_loss=False):
        if x.shape[-1] != self.filters[0]:
            raise Exception('(batch_size, vertex_cnt, 3)이 아닌 데이터는 처리할 수 없습니다.')

        x = self.encoder(x)
        z, mean, log_var = self._parameterize(x)
        pred = self.decoder(z)

        if is_loss:
            return pred, mean, log_var
        else:
            return pred

    def reset_parameters(self):
        fc_list = [self.mean_fc, self.log_var_fc]
        for fc in fc_list:
            for name, param in fc.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0)
                else:
                    init.xavier_uniform_(param)
