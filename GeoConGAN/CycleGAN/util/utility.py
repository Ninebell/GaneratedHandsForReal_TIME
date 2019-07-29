import os
import numpy as np
from PIL import Image


class DataLoader:
    def __init__(self, paths, batch_size):
        self.paths = paths
        self.batch_size = batch_size
        self.n_batch = 0
        self.load_path()


    def load_path(self):
        self.path_trainA = os.listdir(self.paths[0])
        self.path_trainB = os.listdir(self.paths[1])

        self.path_testA = os.listdir(self.paths[2])
        self.path_testB = os.listdir(self.paths[3])


    def data_load(self, is_train = True):

        if is_train:
            path_a = self.path_trainA
            path_b = self.path_trainB
            root_a = self.paths[0]
            root_b = self.paths[1]

        else:
            path_a = self.path_testA
            path_b = self.path_testB
            root_a = self.paths[2]
            root_b = self.paths[3]


        self.n_batch = int(min(len(path_a)//self.batch_size, len(path_b)//self.batch_size))

        idx = np.random.choice(self.n_batch*self.batch_size, self.n_batch*self.batch_size, replace=False)

        for n in range(self.n_batch):
            imgs_A = []
            imgs_B = []
            for i in range(self.batch_size):
                img_A = imread(os.path.join(root_a, path_a[idx[n*self.batch_size + i]]))
                img_B = imread(os.path.join(root_b, path_b[idx[n*self.batch_size + i]]))

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1
            imgs_B = np.array(imgs_B)/127.5 - 1
            yield imgs_A, imgs_B


def imread(path):
    return np.array(Image.open(path).convert('RGB'))


if __name__ == "__main__":
    paths = []
    paths.append("../../Data/horse2zebra/trainA")
    paths.append("../../Data/horse2zebra/trainB")
    paths.append("../../Data/horse2zebra/testA")
    paths.append("../../Data/horse2zebra/testB")
    data_loader = DataLoader(paths,4)
    for (imgs_A, imgs_B) in data_loader.data_load(False):
        print("H")