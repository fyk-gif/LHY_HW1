# -*— coding: utf-8 -*-
# @Time : 2021/6/15 10:09
# @Author : FYK

import torch
import random
import numpy as np


def init_seed(seed=0, device="cpu"):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True