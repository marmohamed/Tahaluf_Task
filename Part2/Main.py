from Tahaluf_Task.Part2.utils.train_utils import get_dataset
from utils.args_parser import get_arguments
from utils.train_utils import *

def main():
    args = get_arguments()

    train_dataset = get_dataset(args, split='train')
    valid_dataset = get_dataset(args, split='valid')    

    train_data_loader = get_data_loader(train_dataset, args)
    valid_data_loader = get_data_loader(valid_dataset, args)

    model = get_model(args)
    optimizer = get_optimizer(args)
    loss_fn = get_loss_fn(args)
    scheduler = get_scheduler(args)

    kwargs = {
        'model': model,
        'optimizer': optimizer,
        'train_data_loader': train_data_loader,
        'valid_data_loader': valid_data_loader,
        'loss_fn': loss_fn,
        'scheduler': scheduler,
        'args': args
    }
    model_trainer = get_trainer(**kwargs)

    model_trainer.train()




if __name__ == "__main__":
    main()
