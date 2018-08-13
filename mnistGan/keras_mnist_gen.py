import argparse
from keras.datasets import mnist


import numpy as np



def train(batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train[:, :, :, ]


def generate():
    None

def get_args():
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--mode", type=str)
    parsers.add_argument("--batch_size", type=int)

    return parsers.parse_args()
train(1)
'''
if __name__ == '__main__':
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size)
    elif args.mode == "generate":
        generate()
'''