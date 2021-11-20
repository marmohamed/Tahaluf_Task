import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--save_path", default='./', type=str)
    parser.add_argument("--csv_data_path", default='./csv_data_path.csv', type=str)
    parser.add_argument("--data_path", default='./', type=str)
    parser.add_argument("--write_logs", action='store_true')
    parser.add_argument("--log_dir", default='./', type=str)
    parser.add_argument("--experiment_name", default='', type=str)
    parser.add_argument("--weight_loss", action='store_true')
    parser.add_argument('--model_type', default='resnet50', type=str)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument('--get_model_info', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    args = parser.parse_args()
    return args