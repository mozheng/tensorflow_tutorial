import os
import random



path="D:/Data/DogsvsCats/train"
make_list_file(path)
a = load_and_shuffle_images_listfile("train.txt")
print(a)
