from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import os

import matplotlib
import importlib
import os
import functools
import itertools
import torch

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
#import tensorflow_gan as tfgan
import tqdm
import io
import inspect
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils

from models import ncsnpp
from models import ncsnpp_1d
from models import ncsnpp_2d_ArbitraryShape

from models import layerspp
from models import layers
from models import normalization
from torch.utils.tensorboard import SummaryWriter
os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin/'  

#from configs import Default_ncsn2dArbShape_continous_ve as configs
#from configs import Default_ncsnpp1d_continous_ve as configs
from configs import Foil_ncsnpp2d_continous_ve as configs

config = configs.get_config()


score_model = mutils.create_model(config)


score_model = score_model.train()
x = torch.ones(8, config.data.num_channels, config.data.height, config.data.width)  # (B, C, H, W)
t = torch.ones([1] * 8)
if config.model.conditional_physics:
    y = torch.ones(8, config.data.dim_physics)
    
if config.model.conditional_physics_distributed:
    y_dis = torch.ones(8, config.data.dim_physics_distributed, config.data.height, config.data.width)

score_model = score_model.to(x.device)
if torch.cuda.is_available():	
	score_model = score_model.cuda()
	x = x.cuda(); t = t.cuda(); y = y.cuda(); y_dis = y_dis.cuda()

## fOR 
vistype = 1
match vistype:
    case 1:
        score = score_model(x, t, [y, y_dis])
        # from torchviz import make_dot
        # output = make_dot(score.mean(), 
        #                 params=dict(score_model.named_parameters()), 
        #                 show_attrs=True, 
        #                 show_saved=True
        #                 )
        # output.format = "png"
        # output.directory = "."
        # output.render("torchviz", view=True)
    case 2:
        writer = SummaryWriter('./')
        writer.add_graph(score_model, (x, y))
        writer.close()
        
        print("模型图已添加到 TensorBoard，运行以下命令查看：")
        print("tensorboard --logdir=runs/")
    case _:
        raise ValueError("Unsupported match type. Use 1 for torchviz or 2 for TensorBoard.")
