import argparse
from keras.datasets import mnist






def train(batch_size):




def generate():


def get_args():
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--mode", type=str)
    parsers.add_argument("--batch_size", type=int)

    return parsers.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size)
    elif args.mode == "generate":
