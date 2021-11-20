from utils.trainer import Trainer
from datasets.clothing_dataset import ClothingDataset
from datasets.augmentations import *
from models.model import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np
import os

def seed_everything(args):
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(args, split='train'):
    transforms = get_transforms_obj(split, height=args.height, width=args.width)
    dataset = ClothingDataset(args.csv_data_path, args.data_path, split, transforms)
    return dataset

def get_data_loader(dataset, split, args, sampler=None):
    if sampler is None:
        shuffle = split.lower() == 'train'
        data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=2
            )
    else:
        data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                sampler=sampler
            )
    return data_loader

def get_model(n_claases, device, args):
    model = build_model(n_claases, device, args)
    return model

def get_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    return optimizer

def get_loss_fn(args, nSamples=None, device=None, gpu=None):
    if args.weight_loss:
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(device)
        loss_func = torch.nn.CrossEntropyLoss(weight=normedWeights)
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    if gpu is not None:
        loss_fn = loss_fn.cuda(gpu)
    return loss_func

def get_scheduler(optimizer, args):
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                              factor=0.5, patience=3,
                                                              verbose=True, min_lr=1e-8)
    return lr_scheduler

def get_writer(args, gpu_num=None):
    if args.write_logs:
        if gpu_num is not None:
            log_dir = os.path.join(args.log_dir, args.experiment_name, str(gpu_num))
        else:
            log_dir = os.path.join(args.log_dir, args.experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        return writer
    return None

def get_trainer(**kwargs):
    trainer = Trainer(**kwargs)
    return trainer

