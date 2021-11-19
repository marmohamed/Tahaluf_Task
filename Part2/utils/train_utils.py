from utils.trainer import Trainer
from datasets.clothing_dataset import ClothingDataset
from datasets.augmentations import *
from models.model import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np

def seed_everything(args):
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(args, split='train'):
    transforms = get_transforms_obj(split)
    dataset = ClothingDataset(args.csv_data_path, args.data_path, split, transforms)
    return dataset

def get_data_loader(dataset, split, args):
    shuffle = split.lower() == 'train'
    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    return data_loader

def get_model(n_claases, device, args):
    model = build_model(n_claases, device, args)
    return model

def get_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    return optimizer

def get_loss_fn(args):
    loss_func = torch.nn.CrossEntropyLoss()
    return loss_func

def get_scheduler(optimizer, args):
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                              factor=0.5, patience=3,
                                                              verbose=True, min_lr=1e-8)
    return lr_scheduler

def get_writer(args):
    if args.write_logs:
        writer = SummaryWriter(log_dir=args.log_dir)
        return writer
    return None

def get_trainer(**kwargs):
    trainer = Trainer(**kwargs)
    return trainer

