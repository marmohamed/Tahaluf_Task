from utils.train_utils import get_dataset
from utils.args_parser import get_arguments
from utils.train_utils import *
from model_info.macs import *
from model_info.receptive_field import *
import json
import os

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def train(gpu=None, args=None):

    if args.train_multinode:
        rank = args.nr * args.gpus + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    else:
        rank=None
        
    seed_everything(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = get_dataset(args, split='train')
    valid_dataset = get_dataset(args, split='valid')  

    if args.train_multinode:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)  
    else:
        train_sampler = None
        val_sampler = None
    train_data_loader = get_data_loader(train_dataset, 'train', args, sampler=train_sampler)
    valid_data_loader = get_data_loader(valid_dataset, 'valid', args, sampler=val_sampler)

    model = get_model(train_dataset.n_classes, device, args)
    if args.train_multinode:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    if args.get_model_info:
        macs, params = get_macs(model, device, args)
        print("macs = ", str(macs), ", params = ", str(params))
        receptive_field_dict = receptive_field(model, (3, args.width, args.height))
        # print("receptive_field_dict = ", receptive_field_dict)
    optimizer = get_optimizer(model, args)
    loss_fn = get_loss_fn(args, train_dataset.label_dist, device, gpu)
    scheduler = get_scheduler(optimizer, args)
    writer = get_writer(args, get_writer)

    kwargs = {
        'model': model,
        'optimizer': optimizer,
        'train_data_loader': train_data_loader,
        'valid_data_loader': valid_data_loader,
        'loss_fn': loss_fn,
        'scheduler': scheduler,
        'device': device,
        'args': args,
        'writer': writer
    }
    model_trainer = get_trainer(**kwargs)

    model_trainer.train()

def main():

    args = get_arguments()
    os.makedirs(os.path.join(args.save_path, args.experiment_name), exist_ok=True)
    with open(os.path.join(args.save_path, args.experiment_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.train_multinode:
        args.world_size = args.gpus * args.nodes        
        os.environ['MASTER_ADDR'] = 'localhost'      
        os.environ['MASTER_PORT'] = '8888'               
        mp.spawn(train, nprocs=args.gpus, args=(args,))       
    else:
        train(None, args)

if __name__ == "__main__":
    main()

