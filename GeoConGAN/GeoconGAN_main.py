from GeoConGAN.GeoconGAN import GeoConGAN
from GeoConGAN.SilNet.silnet import SilNet
import cv2
import os
import argparse
import numpy as np
class DataGenerator:
    def __init__(self, data_path):
        self.real_idx = 0
        self.synth_idx = 0
        self.real_idx_list = []
        self.synth_idx_list = []
        data_path = data_path.split(',')
        real_path = data_path[0]
        synth_path = data_path[1]
        self.real_path_list = DataGenerator.load_list(real_path+'/origin/')
        self.synth_path_list = DataGenerator.load_list(synth_path+'/origin/')
        self.real_sil_path_list = DataGenerator.load_list(real_path+'/mask/')
        self.synth_sil_path_list = DataGenerator.load_list(synth_path+'/mask/')
        self.real_idx_list = np.arange(len(self.real_path_list))
        self.synth_idx_list = np.arange(len(self.synth_path_list))

    def check_amount(self):
        if len(self.real_path_list) != len(self.real_sil_path_list):
            print("You must have same amount of real path list and real silhouette list")
            return False

        if len(self.synth_path_list) != len(self.synth_sil_path_list):
            print("You must have same amount of synth path list and synth silhouette list")
            return False

        return True

    def get_train_batch(self, batch_size):
        real_images = []
        synth_images = []
        real_sil_images = []
        synth_sil_images = []
        if self.real_idx + batch_size >= len(self.real_idx_list):
            self.real_idx = 0
            np.random.shuffle(self.real_idx_list)

        if self.synth_idx + batch_size >= len(self.synth_idx_list):
            self.synth_idx = 0
            np.random.shuffle(self.synth_idx_list)

        for i in range(batch_size):
            real_idx = i+self.real_idx % len(self.real_idx_list)
            synth_idx = i+self.synth_idx % len(self.synth_idx_list)
            real_idx = self.real_idx_list[real_idx]
            synth_idx = self.synth_idx_list[synth_idx]
            real_images.append(cv2.imread(self.real_path_list[real_idx], cv2.IMREAD_COLOR))
            real_sil_images.append(cv2.imread(self.real_sil_path_list[real_idx], cv2.IMREAD_GRAYSCALE))

            synth_images.append(cv2.imread(self.synth_path_list[synth_idx], cv2.IMREAD_COLOR))
            synth_sil_images.append(cv2.imread(self.synth_sil_path_list[synth_idx], cv2.IMREAD_GRAYSCALE))

        self.real_idx = (self.real_idx + batch_size) % len(self.real_idx_list)
        self.synth_idx = (self.synth_idx + batch_size) % len(self.synth_idx_list)

        real_images = np.asarray(real_images, np.float64) / 127.5 - 1
        real_sil_images = np.asarray(real_sil_images, np.float64) / 127.5 - 1
        real_sil_images = real_sil_images.reshape((-1, 256, 256, 1))

        synth_images = np.asarray(synth_images, np.float64) / 127.5 - 1
        synth_sil_images = np.asarray(synth_sil_images, np.float64) / 127.5 - 1
        synth_sil_images = synth_sil_images.reshape((-1, 256, 256, 1))

        return real_images, real_sil_images, synth_images, synth_sil_images

    @staticmethod
    def load_list(root_path):
        path_list = []
        filenames = os.listdir(root_path)
        for filename in filenames:
            full_filename = os.path.join(root_path, filename)
            if os.path.isdir(full_filename):
                child_path_list = DataGenerator.load_list(full_filename)
                for child_path in child_path_list:
                    path_list.append(child_path)
            else:
                path_list.append(full_filename)
        return path_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument for training regnet')
    parser.add_argument('--data', type=str,
                        help='data path for training GeoConGAN. You need synth, real, silhouette path')
    parser.add_argument('--s_model', type=str, help='model weight path')
    parser.add_argument('--g_model', type=str, help='model weight folder path')
    parser.add_argument('--batch', type=int, default=4, help='model weight folder path')

    FLAGS = parser.parse_args()
    if FLAGS.s_model == None:
        print('You must call the SilNet model.')
        exit(0)
    silNet = SilNet((256,256,3))
    silNet.model.load_weights(FLAGS.s_model)

    generator = DataGenerator(FLAGS.data)
    if generator.check_amount() is False:
        exit(0)

    geoconGAN = GeoConGAN(silNet.model, (256, 256, 3), FLAGS.batch, generator, "geo")

    if FLAGS.g_model is not None:
        geoconGAN.load_weight(FLAGS.g_model)

    geoconGAN.train_on_generator(10, 1500)




