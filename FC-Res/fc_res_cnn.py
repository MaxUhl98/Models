"""
Implementation of https://www.researchgate.net/publication/345176073_An_Intelligent_EEG_Classification_Methodology_Based_on_Sparse_Representation_Enhanced_Deep_Learning_Networks
"""
import torch
from torch import nn
from typing import Callable, Dict, Any, Union
from _config import Config
import numpy as np
import pandas as pd
from utils.helpers import get_logger
from utils.engine import k_fold_train
from data_generator import prepare_train, GenericData
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pprint import pformat


class ResidualConvolution(nn.Module):
    def __init__(self, input_channels: int, kernel_size: int, inner_dim: int, pooler: Callable, pooler_kernel_size: int,
                 pooler_kwargs: Dict[str, Any], dropout: float = .5, padding: Union[str, int] = 'same',
                 activation_fn: Callable = nn.RReLU):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=inner_dim, kernel_size=kernel_size, padding=padding),
            activation_fn())
        self.batchnorm1 = nn.BatchNorm1d(inner_dim)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=kernel_size, padding=padding),
            activation_fn(),
            nn.Conv1d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=kernel_size, padding=padding),
            activation_fn(),
            nn.BatchNorm1d(inner_dim))
        self.pool_block = nn.Sequential(nn.Dropout(dropout), pooler(pooler_kernel_size, **pooler_kwargs),
                                        nn.BatchNorm1d(inner_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + self.batchnorm1(x)
        x = self.pool_block(x)
        return x


class DownsampleConvolution(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride_step_size: int = 3,
                 dropout: float = .25, activation_fn: Callable = nn.RReLU):
        super().__init__()
        self.downsample_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                      stride=stride_step_size), activation_fn(), nn.Dropout(dropout), nn.BatchNorm1d(output_channels))
        self.downsample_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size,
                      stride=stride_step_size), activation_fn(), nn.Dropout(dropout), nn.BatchNorm1d(output_channels))

    def forward(self, x):
        x = self.downsample_conv(x)
        return self.downsample_conv2(x)


class Classifier(nn.Module):
    def __init__(self, input_channels: int, kernel_size: int, output_channels: int, linear_input_size: int,
                 conv_activation_fn: Callable = nn.RReLU, mlp_activation_fn: Callable = nn.RReLU, dropout: float = .5,
                 mlp_dim: int = 16,
                 padding: Union[str, int] = 'same', num_classes: int = Config.n_classes):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels=input_channels, kernel_size=kernel_size, padding=padding,
                                            out_channels=output_channels),
                                  conv_activation_fn())
        self.classifier_head = nn.Sequential(nn.Dropout(dropout),
                                             nn.Linear(in_features=linear_input_size, out_features=mlp_dim),
                                             mlp_activation_fn(), nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
                                             mlp_activation_fn(),
                                             nn.Linear(in_features=mlp_dim, out_features=num_classes),
                                             mlp_activation_fn(), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier_head(x)


class FC_Res(nn.Module):
    def __init__(self, input_channels: int, num_residual_blocks: int, residual_conv_kernel_size: int,
                 residual_inner_dim: int, residual_expansion_factor: int, downsampled_channels: int,
                 downsample_kernel_size: int, classifier_kernel_size: int, linear_input_size: int,
                 classifier_conv_output_channels: int,
                 mlp_dim: int = 16,
                 stride_step_size: int = 3,
                 downsample_dropout: float = .25, classifier_dropout: float = .5, residual_dropout: float = .5,
                 conv_activation_fn: Callable = nn.RReLU,
                 mlp_activation_fn: Callable = nn.RReLU,
                 pooler: Callable = nn.MaxPool1d,
                 pooler_kernel_size: int = 2, pooler_kwargs: Dict[str, Any] = {},
                 residual_padding: Union[int, str] = 'same', classifier_padding: Union[int, str] = 'same',
                 gru_size: int = 128, batch_first: bool = True, is_bidirectional: bool = True,
                 gru_dropout: float = .25, n_gru_layers: int = 1):
        super().__init__()
        self.downsampler = DownsampleConvolution(input_channels=input_channels,
                                                 output_channels=downsampled_channels,
                                                 kernel_size=downsample_kernel_size,
                                                 stride_step_size=stride_step_size,
                                                 dropout=downsample_dropout, activation_fn=conv_activation_fn)

        self.residual_blocks = nn.ModuleList(
            [ResidualConvolution(input_channels=downsampled_channels, kernel_size=residual_conv_kernel_size,
                                 inner_dim=residual_inner_dim * residual_expansion_factor, pooler=pooler,
                                 pooler_kernel_size=pooler_kernel_size,
                                 pooler_kwargs=pooler_kwargs, activation_fn=conv_activation_fn,
                                 padding=residual_padding, dropout=residual_dropout)] + [
                ResidualConvolution(input_channels=residual_inner_dim * residual_expansion_factor ** num,
                                    kernel_size=residual_conv_kernel_size,
                                    inner_dim=residual_inner_dim * residual_expansion_factor ** (num + 1),
                                    pooler=pooler,
                                    pooler_kernel_size=pooler_kernel_size,
                                    pooler_kwargs=pooler_kwargs, activation_fn=conv_activation_fn,
                                    padding=residual_padding, dropout=residual_dropout) for num in
                range(1, num_residual_blocks)])

        #self.rnn = nn.GRU(residual_inner_dim * (residual_expansion_factor ** num_residual_blocks), hidden_size=gru_size,
        #                  batch_first=batch_first, bidirectional=is_bidirectional, dropout=gru_dropout,
        #                  num_layers=n_gru_layers)

        self.classification_head = Classifier(
            input_channels=16,
            kernel_size=classifier_kernel_size,
            linear_input_size=linear_input_size, mlp_activation_fn=mlp_activation_fn,
            conv_activation_fn=conv_activation_fn, padding=classifier_padding, dropout=classifier_dropout,
            mlp_dim=mlp_dim, output_channels=classifier_conv_output_channels)

    def forward(self, x):
        x = self.downsampler(x)
        for num, residual_block in enumerate(self.residual_blocks):
            x = residual_block(x)
        #x, _ = self.rnn(x.permute(0,2,1))
        return self.classification_head(x)


if __name__ == '__main__':
    name = 'FC-Res Expert EEGS'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_default_device(device)
    expected_shape = Config.expert_eeg_expected_shape


    model_kwargs = {'input_channels': expected_shape[0],
                    'num_residual_blocks': 3,
                    'residual_conv_kernel_size': 5,
                    'residual_inner_dim': 16,
                    'residual_expansion_factor': 1,
                    'downsampled_channels': 16,
                    'downsample_kernel_size': 5,
                    'classifier_kernel_size': 5,
                    'linear_input_size': 6912,
                    'classifier_conv_output_channels': 256,
                    'mlp_dim': 64,
                    'stride_step_size': 3,
                    'downsample_dropout': .25,
                    'classifier_dropout': .5,
                    'residual_dropout': .5,
                    'conv_activation_fn': nn.ReLU,
                    'mlp_activation_fn': nn.ReLU,
                    'pooler': nn.MaxPool1d,
                    'pooler_kernel_size': 2,
                    'pooler_kwargs': {'stride':2},
                    'residual_padding': 'same',
                    'classifier_padding': 'same'}
    normalize = True
    input_expected_shape = Config.full_data_expected_shape
    model = FC_Res(**model_kwargs)
    model.to(device)
    summary = summary(model, torch.rand(Config.batch_size, *expected_shape), verbose=0, device=device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger = get_logger(name)
    logger.info(summary)
    logger.info(f"The network has {params} trainable parameters")
