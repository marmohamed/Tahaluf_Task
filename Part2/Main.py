from utils.train_utils import get_dataset
from utils.args_parser import get_arguments
from utils.train_utils import *
from model_info.macs import *
from model_info.receptive_field import *
import json
import os

def main():
    args = get_arguments()

    os.makedirs(os.path.join(args.save_path, args.experiment_name), exist_ok=True)
    with open(os.path.join(args.save_path, args.experiment_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    

    seed_everything(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = get_dataset(args, split='train')
    valid_dataset = get_dataset(args, split='valid')    

    train_data_loader = get_data_loader(train_dataset, 'train', args)
    valid_data_loader = get_data_loader(valid_dataset, 'valid', args)

    model = get_model(train_dataset.n_classes, device, args)
    if args.get_model_info:
        macs, params = get_macs(model, device, args)
        print("macs = ", str(macs), ", params = ", str(params))
        receptive_field_dict = receptive_field(model, (3, args.width, args.height))
        # print("receptive_field_dict = ", receptive_field_dict)
    optimizer = get_optimizer(model, args)
    loss_fn = get_loss_fn(args, train_dataset.label_dist, device)
    scheduler = get_scheduler(optimizer, args)
    writer = get_writer(args)

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


if __name__ == "__main__":
    main()

