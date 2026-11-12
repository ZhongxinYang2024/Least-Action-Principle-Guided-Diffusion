# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with VE SDE."""
import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 64
    training.n_iters = 1000001
    training.log_freq = 500
    training.eval_freq = 1000
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 20000
    training.snapshot_freq = 50000
    ## produce samples at each snapshot.
    training.snapshot_sampling = False
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = True
    training.sde = 'vesde'

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False # no random
    sampling.snr = 0.16


    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 20
    evaluate.end_ckpt = evaluate.begin_ckpt
    evaluate.batch_size = 10
    evaluate.enable_sampling = True
    evaluate.num_samples = 0


    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'FreeFall'
    data.height = 128  # should be divisible by 2^n for n in range(num_resolutions)
    data.width = 1  # should be divisible by 2^n for n in range(num_resolutions)
    data.centered = False
    data.num_channels = 2
    data.dim_physics = 1
    data.dim_physics_distributed = 0
    
    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.name = 'ncsnpp_BCH1'
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)  # num_resolutions = len(ch_mult)  downsampling levels
    model.num_res_blocks = 1
    model.attn_resolutions = [64,]
    model.resamp_with_conv = True
    model.conditional = True
    model.conditional_physics=True
    model.fir = False  #  fir for 1d is not implemented
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conditional_physics = True
    model.conditional_physics_distributed = False
    model.Encode2DcondInto1Dvector = True
    
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
