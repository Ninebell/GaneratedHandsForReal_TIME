'''
'''
from keras.applications import resnet50
from keras.layers import *
from keras.models import Model
from RegNet.projLayer import ProjLayer, RenderingLayer, ReshapeChannelToLast
import os
import matplotlib.pyplot as plt
import time


class RegNet:
    def __init__(self, input_shape):
        input_layer = Input(input_shape)
        resnet = resnet50.ResNet50(input_tensor=input_layer, weights='imagenet', include_top=False)
        conv = RegNet.make_conv(resnet.output)
        # conv = Conv2D(filters=1024 ,kernel_size=3, strides=1, padding='same', activation='relu')(conv)
        # heatmap_4f = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', name='heatmap_4f')(conv)
        # heatmap_4f_up = Deconv2D(filters=22,
        #                          kernel_size=4,
        #                          strides=2,
        #                          padding='same',
        #                          name='heatmap_4f_up')(heatmap_4f)
        # heatmap_before_proj = Deconv2D(filters=22,
        #                                kernel_size=4,
        #                                strides=2,
        #                                padding='same',
        #                                name='heatmap_beofre_proj')(heatmap_4f_up)
        flat = Flatten()(conv)
        fc_joints3d_1_before_proj = Dense(200, name='fc_joints3d_1_before_proj')(flat)
        joints3d_prediction_before_proj = Dense(63, name='joints3d_prediction_before_proj')(fc_joints3d_1_before_proj)
        reshape_joints3D_before_proj = Reshape((21,1,3), name='reshape_joints3D_before_proj')(joints3d_prediction_before_proj)
        temp = Reshape((21,3))(reshape_joints3D_before_proj)
        projLayer = ProjLayer()(temp)
        print(projLayer.shape)
        heatmaps_pred3D = RenderingLayer([32,32], coeff=1, name='heatmaps_pred3D')(projLayer)
        heatmaps_pred3D_reshape = ReshapeChannelToLast()(heatmaps_pred3D)

        conv_rendered_2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(heatmaps_pred3D_reshape)
        conv_rendered_3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(conv_rendered_2)
        concat_pred_rendered = concatenate([conv, conv_rendered_3])
        conv_rendered_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(concat_pred_rendered)
        heatmap_prefinal_small = Conv2D(filters=64, kernel_size=3,strides=1,padding='same')(conv_rendered_4)
        heatmap_prefinal = Deconv2D(filters=21, kernel_size=4, strides=2, padding='same', name='heatmap_prefinal')(heatmap_prefinal_small)
        heatmap_final = Deconv2D(filters=21, kernel_size=4, strides=2, padding='same', name='heatmap_final')(heatmap_prefinal)

        flat = Flatten()(conv_rendered_4)
        fc_joints3D_1_final = Dense(200, name='fc_joints3D_1_final')(flat)
        joints3D_final = Dense(63, name='joints3D_prediction_final')(fc_joints3D_1_final)
        joints3D_final_vec = Reshape((21,1,3), name='joint3d_final')(joints3D_final)

        self.model = Model(inputs=input_layer, output=[reshape_joints3D_before_proj, joints3D_final_vec, heatmap_final])
        # self.model = Model(inputs=input_layer, output=projLayer)
        self.model.summary()


    @staticmethod
    def make_conv(input_layer):
        conv4e = RegNet.conv_block(3, 1, 512, input_layer)
        conv4f = RegNet.conv_block(3, 1, 256, conv4e)
        return conv4f

    @staticmethod
    def conv_block(k, s, f, input_layer):
        conv = Conv2D(kernel_size=k, strides=s, filters=f, padding='same', dtype='float32')(input_layer)
        batch = BatchNormalization()(conv)
        # To - Do
        # I need to apply Scale
        return batch

    def train_on_batch(self, train_generator, test_generator):
        steps = train_generator.__len__()

        idx = 0
        test_idx = 1
        for image, crop_param, joint_3d, joint_3d_rate, joint_2d in train_generator.getitem():
            start_time = time.time()
            joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 1, 3))
            result = self.model.train_on_batch(x=[image],
                                               y=[joint_3d_rate, joint_3d_rate, joint_2d])

            spend_time = time.time() - start_time
            rest = steps-idx
            print('{0}/{1}'.format(idx, steps),
                  result,
                  "rest: {0:.2f}".format(spend_time * rest))

            idx = (idx + 1) % steps

            if idx % (steps//100) == 1:
                self.test_on_batch(test_generator, test_idx)
                test_idx += 1

    def test_on_batch(self, test_generator, epoch):
        root = "D:\\RegNet\\result\\{0}".format(epoch)
        # root = "C:\\RegNet\\result\\{0}".format(epoch)
        os.makedirs(root, exist_ok=True)
        idx = 0
        for image, crop_param, joint_3d, joint_3d_rate, joint_2d in test_generator.getitem():

            result = self.model.predict_on_batch(x=[image])
            joint_3d_rate = np.reshape(joint_3d_rate, (-1,21,3))
            final_3d_rate = result[1]
            heatmap = result[2]
            for i in range(0, test_generator.batch_size):

                plt.imsave(root + "\\image_{0}_{1}.png".format(idx, i+1), image[i])

                o3r_file = open(root+"\\joint_3d_{0}_{1}.txt".format(idx,i+1), 'w')
                for k in range(0, 21):
                    value1 = joint_3d_rate[i][k]
                    value2 = final_3d_rate[i][k]
                    val_str = "{0}, {1}, {2} : {3}, {4}, {5} diff: {6}\n".format(value1[0], value1[1], value1[2],
                                                                                 value2[0,0], value2[0,1], value2[0,2],
                                                                                 abs(value1[0]-value2[0,0])+
                                                                                 abs(value1[1]-value2[0,1])+
                                                                                 abs(value1[2]-value2[0,2]) )
                    o3r_file.writelines(val_str)
                o3r_file.close()

                result_image = heatmap[i,:,:,0]
                for j in range(1,21):
                    result_image += heatmap[i,:,:,j]
                plt.imsave(root + "\\joint_{0}_{1}.png".format(idx, i+1), result_image)
                joint = result[2][0][0]
            for t in result[2][0]:
                joint += t

            joint *= 128


            idx = (idx + 1)

if __name__ == "__main__":
    RegNet((256, 256, 3))
