from keras.engine.topology import Network
import random
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from PIL import Image
from skimage import color
import keras
import datetime
import os
from keras.layers import *
from keras.models import Model
from GeoConGAN.CycleGAN.util.utility import *

horse2zebra = "../Data/horse2zebra"
result_path = "D:\\horse2zebra_result"


class CycleGAN:
    def __init__(self, img_shape, batch_size, paths):
        print(img_shape)
        self.shape = img_shape
        self.disc_patch = (16,16,1)
        self.batch_size = batch_size
        self.lambda_cycle = 10.0
        self.lambda_id = 0.1 * self.lambda_cycle
        self.data_loader = DataLoader(paths, batch_size)

        self.__build__()
        self.combined.summary()
        self.genBA.summary()
#        keras.utils.plot_model(self.combined,'check model.png',show_shapes=True)

    def __build__(self):

        optimizer = keras.optimizers.Adam(0.0002, 0.5)
        inputA = Input(self.shape, name='InputA')
        inputB = Input(self.shape, name='InputB')

        self.genAB = CycleGAN.generator(self.shape, 'genAB')
        self.genBA = CycleGAN.generator(self.shape, 'genBA')

        self.discA = CycleGAN.discriminator(self.shape, 'discA')
        self.discB = CycleGAN.discriminator(self.shape, 'discB')

        self.discA.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        self.discB.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        fake_B = self.genAB(inputA)
        fake_A = self.genBA(inputB)

        reconstr_A = self.genBA(fake_B)
        reconstr_B = self.genAB(fake_A)

        img_A_id = self.genBA(inputA)
        img_B_id = self.genAB(inputB)

        self.discA.trainable = False
        self.discB.trainable = False

        valid_A = self.discA(fake_A)
        valid_B = self.discB(fake_B)

        self.combined = Model(inputs=[inputA, inputB],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse','mse',
                                    'mae','mae',
                                    'mae','mae'],
                              loss_weights=[1,1,
                                            self.lambda_cycle,self.lambda_cycle,
                                            self.lambda_id,self.lambda_id],
                              optimizer=optimizer)

    @staticmethod
    def discriminator(shape, name='discriminator'):
        d_filter = 32
        def normalization_layer():
            return InstanceNormalization()

        def conv2d(filter, kernel, stride, input, normal=True):
            conv2d_layer = Conv2D(filters=filter, kernel_size=kernel, padding='same', strides=stride)(input)
            relu = LeakyReLU(alpha=0.2)(conv2d_layer)

            if normal:
                return normalization_layer()(relu)
            else:
                return relu

        input_layer = Input(shape=shape)
        c64 = conv2d(filter=d_filter, kernel=4, stride=2, input=input_layer, normal=False)
        c128 = conv2d(filter=d_filter*2, kernel=4, stride=2, input=c64)
        c256 = conv2d(filter=d_filter*4, kernel=4, stride=2, input=c128)
        c512 = conv2d(filter=d_filter*8, kernel=4, stride=2, input=c256)
        output_layer = Conv2D(name="output", filters=1, kernel_size=4, strides=1, padding='same')(c512)

        return Model(inputs=input_layer, outputs=output_layer, name=name)

    @staticmethod
    def generator(shape, name='generator'):
        g_filter = 32
        def normalization_layer():
            return InstanceNormalization()

        def conv2d(filter, kernel, input, stride, activation="relu"):
            conv2d_layer = Conv2D(filters=filter, kernel_size=kernel, padding='same', strides=stride)(input)
            acti = ReLU()(conv2d_layer)
            norm = normalization_layer()(acti)
            return norm

        def residual_block(input_layer, filter_size):
            norm_layer = normalization_layer()(input_layer)
            relu = ReLU()(norm_layer)
            conv2d_layer = Conv2D(filters=filter_size, kernel_size=3, padding='same')(relu)

            norm_layer = normalization_layer()(conv2d_layer)
            relu = ReLU()(norm_layer)
            conv2d_layer = Conv2D(filters=filter_size, kernel_size=3, padding='same')(relu)

            return Add()([input_layer, conv2d_layer])

        def upsample(filter, input):
            upsample_layer = UpSampling2D(size=2)(input)
            conv2d_layer = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same', activation="relu")(upsample_layer)
            norm = normalization_layer()(conv2d_layer)
            return norm

        '''
        c7s1_k: 7x7 conv-> instance norm with relu  k filters and stride 1.
        dk: 3x3 conv -> instance norm with relu -> k filter and stride 1.
        Rk: residual two 3x3 conv samve filter k.
        uk 3x3 fractional stride conv -> instance norm with relu  k filter ans stride 1/2
        
        256 x 256: c7s1-64, d128, d256, r256 x 9, u128, u64, c7s1-3
        '''
        input_layer = Input(shape)

        c7s1_64 = conv2d(filter=g_filter, stride=1, kernel=7, input=input_layer)
        d128 = conv2d(filter=g_filter * 2, stride=2, kernel=3, input=c7s1_64)
        d256 = Conv2D(filters=g_filter * 4, strides=2, kernel_size=3, padding="same")(d128)

        r256 = residual_block(d256, g_filter * 4)
        for i in range(0, 8):
            r256 = residual_block(r256, g_filter * 4)

        up128 = upsample(g_filter*2, r256)
        up64 = upsample(g_filter, up128)

        c7s1_3 = Conv2D(filters=3, kernel_size=7, padding="same", strides=1, activation="tanh")(up64)
        output_layer = c7s1_3

        return Model(inputs=input_layer, outputs=output_layer, name=name)

    def train(self, epochs, sample_interval=50):
        start_time = datetime.datetime.now()
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            d_loss = [0, 0]
#           random.shuffle(self.data_loader)
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.data_load()):
                fake_B = self.genAB.predict(imgs_A)
                fake_A = self.genBA.predict(imgs_B)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B])

                dA_loss_real = self.discA.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.discA.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.discB.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.discB.train_on_batch(fake_B, fake)

                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                elapsed_time = datetime.datetime.now() - start_time

                if batch_i % ((self.data_loader.n_batch)//10) == 0:
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                           % (epoch, epochs,
                               batch_i, (self.data_loader.n_batch),
                               d_loss[0], 100*d_loss[1],
                               g_loss[0],
                               np.mean(g_loss[1:3]),
                               np.mean(g_loss[3:5]),
                               np.mean(g_loss[5:6]),
                               elapsed_time))
                if batch_i == ((self.data_loader.n_batch)-1):
                    self.test_save(epoch)


    def test_save(self, batch_i):
        os.makedirs(result_path+"/{0}".format(batch_i), exist_ok=True)
        self.discA.save_weights(result_path+"/{0}/discA.h5".format(batch_i))
        self.discB.save_weights(result_path+"/{0}/discB.h5".format(batch_i))
        self.genAB.save_weights(result_path+"/{0}/genAB.h5".format(batch_i))
        self.genBA.save_weights(result_path+"/{0}/genBA.h5".format(batch_i))
        for i, (imgs_A, imgs_B) in enumerate(self.data_loader.data_load(False)):
            fake_b = self.genAB.predict(imgs_A)
            fake_a = self.genBA.predict(imgs_B)
            fake_a = (fake_a + 1) * 127.5
            fake_b = (fake_b + 1) * 127.5

            for b in range(self.batch_size):
                fake_img_b = np.asarray(fake_b[b], dtype=np.uint8)
                fake_img_a = np.asarray(fake_a[b], dtype=np.uint8)

                fake_img_b = Image.fromarray(fake_img_b)
                fake_img_a = Image.fromarray(fake_img_a)

                fake_img_a.save(result_path+"/{0}/A_fake{1}.png".format(batch_i, i*self.batch_size+b))
                fake_img_b.save(result_path+"/{0}/B_fake{1}.png".format(batch_i, i*self.batch_size+b))


if __name__ == "__main__":
    paths = []
    paths.append("../Data/horse2zebra/trainA")
    paths.append("../Data/horse2zebra/trainB")
    paths.append("../Data/horse2zebra/testA")
    paths.append("../Data/horse2zebra/testB")
    test = CycleGAN((256, 256, 3), 2, paths)
    test.train(epochs=500)

