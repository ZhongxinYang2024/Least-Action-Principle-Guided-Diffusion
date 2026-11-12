"""Fully connected neural network"""
import torch
import torch.nn as nn
import functools

from . import utils, layers 
from . import layerspp

get_act = layers.get_act


@utils.register_model(name='fcnn_B1H1')
class FCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
        
        self.nf = nf = config.model.nf
        self.layers = layers = config.model.layers
        self.conditional = conditional = config.model.conditional
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        assert embedding_type in ['fourier', 'positional']
        self.scale_by_sigma = config.model.scale_by_sigma
        
        self.input_dim = input_dim = config.model.input_dim
        self.all_modules = nn.ModuleList()
        
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            assert config.training.continuous, "Fourier features are only used for continuous training."

            self.temb_GFProj = layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.model.fourier_scale)
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')
  
  
        if conditional:
            self.temb_proj1 = nn.Linear(embed_dim, 4 * nf)
            self.temb_proj2 = nn.Linear(4 * nf, 4 * nf)
        
        input_size = input_dim
        if conditional:
            input_size += 4 * nf

        prev_size = input_size
        for size in layers:  # 不包括输出层
            self.all_modules.append(nn.Linear(prev_size, size))
            prev_size = size
        
        # 添加输出层
        self.all_modules.append(nn.Linear(prev_size, input_dim))
        
        
    def forward(self, x, time_cond):
        if x.dim() > 2:
            h = x.view(x.size(0), -1)
        else:
            h = x
        
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings
            used_sigmas = time_cond # sigmas for continous
            temb = self.temb_GFProj(torch.log(used_sigmas))
        elif self.embedding_type == 'positional':
            timesteps = time_cond # index for concrete
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        else:
             raise ValueError(f'embedding type {self.embedding_type} unknown.')
                 
        
        modules = self.all_modules
        if self.conditional:
            # Timescale/scale embedding
            temb = self.temb_proj1(temb)
            temb = self.temb_proj2(self.act(temb)) 
            h = torch.cat([h, temb], dim=1)
        else:
            temb = None
        
        
        for i, layer in enumerate(self.all_modules[:-1]):
            h = self.act(layer(h))
        h = self.all_modules[-1](h)
        
        h = h.view(x.shape)
        if self.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas
        
        return h
