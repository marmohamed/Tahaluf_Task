from ..training_utils.trainer import Trainer
from datasets.clothing_dataset import ClothingDataset
from datasets.augmentations import *
from models.model import *
from torch.utils.data import Dataset, DataLoader
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
            shuffle=True,
            num_workers=4
        )
    return data_loader

def get_model(args):
    pass

def get_optimizer(args):
    pass

def get_loss_fn(args):
    loss_func = torch.nn.CrossEntropyLoss()
    return loss_func

def get_scheduler(args):
    pass

def get_trainer(model, train_data_loader, valid_data_loader, args):
    trainer = Trainer(model, train_data_loader, valid_data_loader, args)
    return trainer

