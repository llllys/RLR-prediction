import numpy as np
import random
import pickle as pkl
import os
import torch
from torch.utils.data import DataLoader
from datasets.yizhuang_dataset import yizhuang_dataset

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader(args, mode='train', batch_size=None):
    data_loaders = DataLoader(
        dataset=yizhuang_dataset(args, mode),
        batch_size=args.batch_size if batch_size is None else batch_size,
        shuffle=(mode=='train'),
    )
    return data_loaders


