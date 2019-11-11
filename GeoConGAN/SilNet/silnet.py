# SilNet is referenced from U-Net segmentation.
# It used three 2-strided convolutions and three deconvolutions.
import cv2
import os
from GeoConGAN.ImagePreprocess import HandImageGenerator
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from GeoConGAN.SilNet.unet.data import *

from PIL import Image

result_path = "d:/GeoConGAN/result"

class DataLoader:
    def __init__(self, batch_size, paths):
        self.batch_size = batch_size
        self.paths = paths
        self.n_batch = 0
        self.load_path()

    def load_path(self):
        self.path_trainA = os.listdir(self.paths[0])
        self.path_trainB = os.listdir(self.paths[1])

        self.path_testA = os.listdir(self.paths[2])
        self.path_testB = os.listdir(self.paths[3])

        print(self.path_trainA)
        print(self.path_trainB)
        print(self.path_testA)
        print(self.path_testB)

    def data_load(self, is_train=True):

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

        #idx = np.random.choice(self.n_batch*self.batch_size, self.n_batch*self.batch_size, replace=False)

        for n in range(self.n_batch):
            imgs_A = []
            imgs_B = []
            for i in range(self.batch_size):
                img_A = imread(os.path.join(root_a, path_a[n*self.batch_size + i]))
                img_B = imread(os.path.join(root_b, path_b[n*self.batch_size + i]))

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1
            imgs_B = np.array(imgs_B)/127.5 - 1

            yield imgs_A, imgs_B


def imread(path):
    image = np.array(Image.open(path))
    if image.shape[0] != 256:
        image.resize((256,256,1))

    image = image.reshape((256,256,1))
    return image


class SilNet:

    def __init__(self, shape, train_generator=None):
        self.shape = shape
        self.model = self.make_model()
        self.compile_model()
        self.train_generator = train_generator

    def make_model(self):
        def normalize_layer():
            return InstanceNormalization()

        filters = 64
        input_layer = Input(self.shape)
        c1 = Conv2D(filters=filters, kernel_size=3, padding='same', strides=1)(input_layer)
        lr = LeakyReLU()(c1)
        n = normalize_layer()(lr)
        c1 = Conv2D(filters=filters, kernel_size=3, padding='same', strides=1)(n)
        lr = LeakyReLU()(c1)
        n = normalize_layer()(lr)
        max_pool = MaxPooling2D()(n)

        c2 = Conv2D(filters=filters * 2, kernel_size=3, padding='same', strides=1)(max_pool)
        lr = LeakyReLU()(c2)
        n = normalize_layer()(lr)
        c2 = Conv2D(filters=filters * 2, kernel_size=3, padding='same', strides=1)(n)
        lr = LeakyReLU()(c2)
        n = normalize_layer()(lr)
        max_pool = MaxPooling2D()(n)

        c3 = Conv2D(filters=filters * 4, kernel_size=3, padding='same', strides=1)(max_pool)
        lr = LeakyReLU()(c3)
        n = normalize_layer()(lr)
        c3 = Conv2D(filters=filters * 4, kernel_size=3, padding='same', strides=1)(n)
        lr = LeakyReLU()(c3)
        n = normalize_layer()(lr)

        max_pool = MaxPooling2D()(n)

        c4 = Conv2D(filters=filters * 8, kernel_size=3, padding='same', strides=1)(max_pool)
        lr = LeakyReLU()(c4)
        n = normalize_layer()(lr)
        c4 = Conv2D(filters=filters * 8, kernel_size=3, padding='same', strides=1)(n)
        lr = LeakyReLU()(c4)
        up_sample = concatenate(
            [Conv2DTranspose(filters=filters * 4, kernel_size=2, strides=2, padding='same')(lr), c3], axis=3)

        c5 = Conv2D(filters=filters * 4, kernel_size=3, padding='same', strides=1)(up_sample)
        lr = LeakyReLU()(c5)
        n = normalize_layer()(lr)
        c5 = Conv2D(filters=filters * 4, kernel_size=3, padding='same', strides=1)(n)
        lr = LeakyReLU()(c5)

        up_sample = concatenate(
            [Conv2DTranspose(filters=filters * 2, kernel_size=2, strides=2, padding='same')(lr), c2], axis=3)

        c6 = Conv2D(filters=filters * 2, kernel_size=3, padding='same', strides=1)(up_sample)
        lr = LeakyReLU()(c6)
        n = normalize_layer()(lr)
        c6 = Conv2D(filters=filters * 2, kernel_size=3, padding='same', strides=1)(n)
        lr = LeakyReLU()(c6)

        up_sample = concatenate([Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same')(lr), c1],
                                axis=3)

        c7 = Conv2D(filters=filters, kernel_size=3, padding='same', strides=1)(up_sample)
        lr = LeakyReLU()(c7)
        n = normalize_layer()(lr)
        c7 = Conv2D(filters=filters, kernel_size=3, padding='same', strides=1)(n)
        lr = LeakyReLU()(c7)

        output = Conv2D(filters=1, kernel_size=1, padding='same', strides=1, activation='tanh')(lr)

        return Model(inputs=input_layer, outputs=output)

    def compile_model(self):
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train_on_batch(self, epoch):
        for epoch_idx in range(0, epoch):
            loss = self.model.fit_generator(self.train_generator.get_train_image_pair(5000), steps_per_epoch=5000, epochs=1, verbose=1)
            print(loss)
            self.test_save(epoch_idx)


    def test_save(self, epoch_idx):
        os.makedirs(result_path+"/{0}".format(epoch_idx), exist_ok=True)
        results = self.model.predict_generator(self.train_generator.get_test_image_pair(100), 100, verbose=2)
        self.model.save_weights(result_path+"/{0}/silnet_model.h5".format(epoch_idx))
        for idx, result in enumerate(results):
            result = (result+1)*127.5
            result = np.asarray(result, np.uint8)
            cv2.imwrite(result_path+"/{0}/{1}.png".format(epoch_idx,idx),result)

if __name__ == "__main__":
    generator = HandImageGenerator()
    silnet = SilNet((256,256,3), generator, 4)
    silnet.model.load_weights("D:\\GeoConGAN\\result\\57\\silnet_model.h5")
    for (origin, mask) in generator.get_test_image_pair(1000):

        result = silnet.model.predict(origin)
        print(origin.shape)
        print(origin)
        origin = (origin + 1) * 127.5
        origin = np.resize(origin, (256,256,3))
        origin = np.asarray(origin, np.uint8)
        result = (result + 1) * 127.5
        result = np.resize(result,(256,256,1))
        result = np.asarray(result,np.uint8)
        cv2.imshow("test_m",result)
        cv2.imshow("test_o",origin)

        cv2.waitKey()

