from train import train
import argparse


def main():
    parser = argparse.ArgumentParser(description='Training...')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch_n', type=int, default=50, help='numbers of epochs')
    parser.add_argument('--worker_n', type=int, default=2, help='numbers of workers')
    parser.add_argument('--loss_ratio', type=float, default=0.1, help='ratio - for combining 2 losses')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--clip_value', type=int, default=1, help='for clip gradient')
    args = vars(parser.parse_args())
    train()#args)


if __name__ == '__main__':
    main()