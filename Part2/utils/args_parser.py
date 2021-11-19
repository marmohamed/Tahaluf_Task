import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--save_path", default='./', type=str)
    parser.add_argument("--csv_data_path", default='./csv_data_path.csv', type=str)
    parser.add_argument("--data_path", default='./data_path/', type=str)
    args = parser.parse_args()
    return args