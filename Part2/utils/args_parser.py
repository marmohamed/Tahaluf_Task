import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="The value of the seed, default=0")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs, default=10")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size, default=32")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate, default=1e-3")
    parser.add_argument("--save_path", default='./', type=str, help="Save path for the model checkpoints and the args, default=./")
    parser.add_argument("--csv_data_path", default='./csv_data_path.csv', type=str, help="Data csv file path, default=./csv_data_path.csv")
    parser.add_argument("--data_path", default='./', type=str, help="Data images path, default=./")
    parser.add_argument("--write_logs", action='store_true', help="Write tensorboard logs, default=False")
    parser.add_argument("--log_dir", default='./', type=str, help="Directory for tensorboard logs, default=./")
    parser.add_argument("--experiment_name", default='', type=str, help="Experiment name, create folder for each experiment")
    parser.add_argument("--weight_loss", action='store_true', help="use weights in the loss function to handle the imbalance, default=False")
    parser.add_argument('--model_type', default='resnet50', type=str, help="model type (resnet50 efficientnet0 cspresnext50), default=resnet50")
    parser.add_argument("--width", default=512, type=int, help="width of the image, default=512")
    parser.add_argument("--height", default=512, type=int, help="height of the image, default=512")
    parser.add_argument('--get_model_info', action='store_true', help="get the macs, params and receptive field of the model, not all models are supported, default=False")
    parser.add_argument('--mixed_precision', action='store_true', help="use mixed precision, default=False")
    
    parser.add_argument('--train_multinode', action='store_true', help="Enable distributed training")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N',
                        help="Number of nodes")
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    return args