from GeoConGAN.SilNet.silnet import SilNet
import os
import numpy as np
import cv2
from GeoConGAN.CycleGAN.model import CycleGAN
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from GeoConGAN.ImagePreprocess import HandImageGenerator, np2cv

class GeoConGAN:
    def __init__(self, silNet, input_shape, batch_size, generator, tag):
        self.batch_size = batch_size
        self.generator = generator
        self.tag = tag
        self.__build__(silNet, input_shape)

    def __build__(self, silNet, input_shape):
        lambda_cycle = 10.0
        real_origin_input = Input(input_shape, name="real_input")

        synth_origin_input = Input(input_shape, name="synth_input")

        self.real_disc = CycleGAN.discriminator(input_shape, name="real_disc")
        self.synth_disc = CycleGAN.discriminator(input_shape, name="synth_disc")
        optimizer = Adam(0.0002, 0.5)

        self.real_disc.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        self.synth_disc.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        self.real_disc.trainable = False
        self.synth_disc.trainable = False

        self.synth2real = CycleGAN.generator(input_shape, name="synth2real")
        self.real2synth = CycleGAN.generator(input_shape, name="real2synth")

        fake_synth = self.real2synth(real_origin_input)
        fake_real = self.synth2real(synth_origin_input)


        resconstr_real = self.synth2real(fake_synth)
        resconstr_synth = self.real2synth(fake_real)

        identity_real = self.synth2real(real_origin_input)
        identity_synth = self.real2synth(synth_origin_input)

        valid_real = self.real_disc(fake_real)
        valid_synth = self.synth_disc(fake_synth)

        silNet.trainable = False

        comp_synth_mask = silNet(fake_real)
        comp_real_mask = silNet(fake_synth)

        self.combined_model = Model(inputs=[real_origin_input,
                                       synth_origin_input],
                               outputs=[valid_real, valid_synth,
                                        resconstr_real, resconstr_synth,
                                        identity_real, identity_synth,
                                        comp_real_mask, comp_synth_mask
                                    ])
                                        # comp_real_mask, comp_synth_mask])

        self.combined_model.compile(optimizer=optimizer,
                               loss=['mse', 'mse',
                                     'mae', 'mae',
                                     'mae', 'mae',
                                     'binary_crossentropy', 'binary_crossentropy'],
                               loss_weights=[1, 1,
                                             lambda_cycle, lambda_cycle,
                                             lambda_cycle * 0.1, lambda_cycle * 0.1,
                                             1,1])
        self.real_disc.summary()
        self.combined_model.summary()

    def train_on_generator(self, epoches, steps_on_epoches):
        valid = np.ones((self.batch_size,)+(16,16,1))
        fake = np.zeros((self.batch_size,)+(16,16,1))
        for epoch in range(0, epoches):
            for step in range(0, steps_on_epoches):
                (real_image, real_mask, synth_image, synth_mask) = self.generator.get_train_batch(self.batch_size)

                fake_real = self.synth2real.predict(synth_image)
                fake_synth = self.real2synth.predict(real_image)

                loss_real_real = self.real_disc.train_on_batch(real_image, valid)
                loss_real_fake = self.real_disc.train_on_batch(fake_real, fake)
                loss_real = np.add(loss_real_real, loss_real_fake) * 0.5

                loss_synth_real = self.synth_disc.train_on_batch(synth_image, valid)
                loss_synth_fake = self.synth_disc.train_on_batch(fake_synth, fake)
                loss_synth = np.add(loss_synth_real, loss_synth_fake) * 0.5

                disc_loss = np.add(loss_real, loss_synth) * 0.5

                combined_loss = self.combined_model.train_on_batch(x=[real_image, synth_image],
                                                                   y=[valid, valid,
                                                                     real_image, synth_image,
                                                                     real_image, synth_image,
                                                                     real_mask, synth_mask
                                                                     ])

                print("epoch({0}/{1}): steps({2}/{3}):".format(epoch+1, epoches, step+1, steps_on_epoches),
                      disc_loss[0], disc_loss[1]*100,
                      combined_loss[0],
                      np.mean(combined_loss[1:3]),
                      np.mean(combined_loss[3:5]),
                      np.mean(combined_loss[5:6])
                      )

            self.test_save(epoch)

    def test_save(self, epoch):
        root_path = ".\\result{0}\\{1}".format(self.tag, epoch)
        os.makedirs(root_path, exist_ok=True)
        os.makedirs(root_path+"\\mask", exist_ok=True)
        os.makedirs(root_path+"\\origin", exist_ok=True)

        for i in range(0, 100//self.batch_size):
            (real_image, real_mask, synth_image, synth_mask) = self.generator.get_test_batch(self.batch_size)
            results_synth = self.synth2real.predict_on_batch(synth_image)
            results_real = self.real2synth.predict_on_batch(real_image)

            real = np2cv(real_image[0], (256,256,3))
            synth = np2cv(synth_image[0], (256,256,3))

            cv2.imshow("real", real)
            cv2.imshow("synth", synth)


            for j, result in enumerate(results_synth):
                result = (result + 1) * 127.5
                result = np.asarray(result, np.uint8)
                result = np.resize(result, (256,256,3))
                cv2.imshow("remake", result)
                cv2.waitKey(10)
                cv2.imwrite(root_path+"\\mask\\{0}.png".format(i*self.batch_size + j), result)

            for j, result in enumerate(results_real):
                result = (result + 1) * 127.5
                result = np.asarray(result, np.uint8)
                result = np.resize(result, (256,256,3))
                cv2.imshow("remake", result)
                cv2.waitKey(10)
                cv2.imwrite(root_path+"\\origin\\{0}.png".format(i*self.batch_size + j), result)

        self.real2synth.save_weights(root_path+"\\real2synth.h5")
        self.synth2real.save_weights(root_path+"\\synth2real.h5")
        self.real_disc.save_weights(root_path+"\\real_disc.h5")
        self.synth_disc.save_weights(root_path+"\\synth_disc.h5")

    def load_weight(self, path):
        self.real2synth.load_weights(path+"\\real2synth.h5")
        self.synth2real.load_weights(path+"\\synth2real.h5")
        self.real_disc.load_weights(path+"\\real_disc.h5")
        self.synth_disc.load_weights(path+"\\synth_disc.h5")

if __name__ == "__main__":
    generator = HandImageGenerator()
    silNet = SilNet((256,256,3), generator, 4)

    silNet.model.load_weights("D:\\GeoConGAN\\silnet\\silnet_model.h5")

    geoconGAN = GeoConGAN(silNet.model, (256, 256, 3), 2, generator,"geo")
    geoconGAN.load_weight("D:\\GeoConGAN\\result_geo\\26")

    geoconGAN.train_on_generator(100, 2000)

