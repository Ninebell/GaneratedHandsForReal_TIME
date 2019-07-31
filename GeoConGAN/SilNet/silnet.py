# SilNet is referenced from U-Net segmentation.
# It used three 2-strided convolutions and three deconvolutions.
import cv2
import os
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Model

from GeoConGAN.SilNet.unet.data import *

from PIL import Image


result_path = "./unet/data/result"


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
    image = np.array(Image.open(path))
    if image.shape[0] == 512:
        image.resize((256,256,1))

    image = image.reshape((256,256,1))
    return image


class SilNet:

    def __init__(self, shape, train_generator, data_loader, batch_size):
        self.shape = shape
        self.model = self.make_model()
        self.compile_model()
        self.train_generator = train_generator
        self.test_generator = data_loader
        self.batch_size = batch_size

    def make_model(self):
        def normalization():
            return BatchNormalization()
        def conv(input_layer, filter):
            c = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same')(input_layer)
            a = ReLU()(c)
            n = normalization()(a)
            c = Conv2D(filters=filter, kernel_size=3, strides=2, padding='same')(n)
            return c
        def resnet(input_layer, filter):
            n = normalization()(input_layer)
            a = ReLU()(n)
            c = Conv2D(filters=filter, kernel_size=5, strides=1, padding='same')(a)
            n = normalization()(c)
            a = ReLU()(n)
            c = Conv2D(filters=filter, kernel_size=5, strides=1, padding='same')(a)
            return Add()([input_layer, c])

        def deconv2d(input_layer, filter, concat):
            upsample = UpSampling2D(size=2)(input_layer)
            merge = concatenate([upsample,concat],axis=3)
            conv2d_layer = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same')(merge)
            n = normalization()(conv2d_layer)
            return n
        filter_size = 64
        input_layer = Input(self.shape)


        conv_layer_1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same')(input_layer)
        a = ReLU()(conv_layer_1)
        n = normalization()(a)
        conv_layer_2 = Conv2D(filters=filter_size, kernel_size=3, strides=2, padding='same')(n)
        a = ReLU()(conv_layer_2)
        n = normalization()(a)

        conv_layer_3 = Conv2D(filters=filter_size*2, kernel_size=3, strides=1, padding='same')(n)
        a = ReLU()(conv_layer_3)
        n = normalization()(a)
        conv_layer_4 = Conv2D(filters=filter_size*2, kernel_size=3, strides=2, padding='same')(n)
        a = ReLU()(conv_layer_4)
        n = normalization()(a)

        res_net = resnet(n, filter_size*2)
        for i in range(0, 5):
            res_net = resnet(res_net, filter_size*2)

        deconv_1 = deconv2d(res_net, filter_size*2, conv_layer_3)
        deconv_2 = deconv2d(deconv_1, filter_size, conv_layer_1)

        output_layer = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(deconv_2)

        return Model(inputs=input_layer, outputs=output_layer)

    def compile_model(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    def train_on_batch(self, epoch):
        for epoch_idx in range(0, epoch):
##            for idx, (input_image, label) in enumerate(self.data_loader.data_load()):
##                loss = self.model.train_on_batch(input_image, label)
            loss = self.model.fit_generator(self.train_generator, steps_per_epoch=300, epochs=5)


            print(loss)
            self.test_save(epoch_idx)

    def test_save(self, epoch_idx):
        os.makedirs(result_path+"/{0}".format(epoch_idx), exist_ok=True)

        for i, (image, label) in enumerate(self.test_generator.data_load(False)):

            results = self.model.predict(image)

            results = (results + 1) * 127.5

            for b in range(self.batch_size):
                result = np.asarray(results[b], dtype=np.uint8)
                save_path = result_path+"/{0}/result_{1}.png".format(epoch_idx, i*self.batch_size+b)

                result = cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)
                result = cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
                cv2.imwrite(save_path, result)
                '''
                result_image = Image.fromarray(result)
                result_image.save(result_path+"/{0}/result_{1}.png".format(epoch_idx, i*self.batch_size+b))
                '''

if __name__ == "__main__":
    paths = [
        "./unet/data/train/input",
        "./unet/data/train/label",
        "./unet/data/test/input",
        "./unet/data/test/label"
    ]

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest',
                         validation_split=0.2)
    data_loader = DataLoader(batch_size=4, paths=paths)

    myGene = trainGenerator(4,'./unet/data/train','input','label',data_gen_args,save_to_dir = None)
    silnet = SilNet((256,256,1), myGene, data_loader, 4)
    silnet.model.summary()
    silnet.train_on_batch(500)

