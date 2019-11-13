'''
                         -------------------------------------------------------------------> conv
    RegNet <=   ResNet50 -> intermediate 3D positions -> ProjLayer -> rendered 2D heatmaps -> conv  -> 2D heatmaps
                   |                                                                                -> 3D positions
                   |                                                                                       |
                   ▼                                                                                       ▼
                ground truth    (L2 loss)                                                               (L2 loss)
'''
import cv2
import time
import os
from PIL import Image
from keras.models import Model
from keras.layers import *
from keras.utils import Sequence
from RegNet.projLayer import RenderingLayer, ReshapeChannelToLast, ProjLayer
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import resnet50

def change3D_2D(points, crop_param):
    intrinsics = [[617.173, 0, 315.453],
                  [0, 617.173, 242.256],
                  [0, 0, 1]]
    intrinsics = np.asarray(intrinsics, np.float64)
    k = np.dot(intrinsics, points)
    points_2d = ((k[0]/k[2]-crop_param[0])*crop_param[2], (k[1]/k[2]-crop_param[1])*crop_param[2])
    return points_2d

class DataGenerator(Sequence):
    def __init__(self, dir_path, batch_size=1, shuffle=True, heatmap_shape=[32,32]):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dir_path = dir_path
        self.heatmap_shape = heatmap_shape
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dir_path)) // self.batch_size)

    def __getitem__(self, item):
        indexes = self.indexes[item*self.batch_size:(item+1)*self.batch_size]
        dir_path = [self.dir_path[i] for i in indexes]
        return self.__data_generation(dir_path)

    def getitem(self):
        for i in range(0,len(self.dir_path)//self.batch_size):
            i = i*self.batch_size
            dir_path = [self.dir_path[j] for j in range(i,self.batch_size + i)]
            yield self.__data_generation(dir_path)
        self.on_epoch_end()

    def __data_generation(self, dir_path):
        image = [np.asarray(Image.open(path+"_color_composed.png")) for path in dir_path]
        image = np.asarray(image, np.uint8)
        image = np.asarray(image, np.float)
        image = image / 255.0
        crop_param = []
        joint_3d = []
        joint_2d_heatmap = []
        joint_3d_rate = []
        for path in dir_path:
            value = open(path+"_crop_params.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            crop_param.append(value)

            value = open(path+"_joint_pos_global.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            joint_3d.append(value)

            value = open(path+"_joint_pos.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            joint_3d_rate.append(value)

            value = open(path+"_joint2D.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            value = np.asarray(value)
            value = np.reshape(value, (21, 2))
            for val in value:
                heat_map = gaussian_heat_map(val/8, self.heatmap_shape[0])
                joint_2d_heatmap.append(heat_map)

        crop_param = np.asarray(crop_param)
        crop_param = np.reshape(crop_param, (-1, 1, 3))

        joint_3d = np.asarray(joint_3d)
        joint_3d = np.reshape(joint_3d, (-1,63))

        joint_3d_rate = np.asarray(joint_3d_rate)
        joint_2d_heatmap = np.asarray(joint_2d_heatmap)
        joint_2d_heatmap = np.reshape(joint_2d_heatmap, (-1, 21, self.heatmap_shape[0], self.heatmap_shape[1]))
        joint_2d_heatmap = np.moveaxis(joint_2d_heatmap, 1, 3)

        return image, crop_param, joint_3d, joint_3d_rate, joint_2d_heatmap

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dir_path))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class RegNet:
    def __init__(self, input_shape, heatmap_shape):
        self.min_loss = [10000.0, 10000., 100000., 100000., 100000., 100000., 100000.]
        self.heatmap_shape=input_shape
        input_layer = Input(input_shape)
        resnet = resnet50.ResNet50(input_tensor=input_layer, weights='imagenet', include_top=False)
        conv = RegNet.make_conv(resnet.output)
        flat = Flatten()(conv)
        fc_joints3d_1_before_proj = Dense(200, name='fc_joints3d_1_before_proj')(flat)
        joints3d_prediction_before_proj = Dense(63, name='joints3d_prediction_before_proj')(fc_joints3d_1_before_proj)
        reshape_joints3D_before_proj = Reshape((21,1,3), name='reshape_joints3D_before_proj')(joints3d_prediction_before_proj)
        temp = Reshape((21,3))(reshape_joints3D_before_proj)
        projLayer = ProjLayer(heatmap_shape)(temp)
        heatmaps_pred3D = RenderingLayer(heatmap_shape, coeff=1, name='heatmaps_pred3D')(projLayer)
        heatmaps_pred3D_reshape = ReshapeChannelToLast(heatmap_shape)(heatmaps_pred3D)

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
        conv4e_relu = ReLU()(conv4e)
        conv4f = RegNet.conv_block(3, 1, 256, conv4e_relu)
        conv4f_relu = ReLU()(conv4f)
        return conv4f_relu

    @staticmethod
    def conv_block(k, s, f, input_layer):
        conv = Conv2D(kernel_size=k, strides=s, filters=f, padding='same', dtype='float32')(input_layer)
        batch = BatchNormalization()(conv)
        # To - Do
        # I need to apply Scale
        return batch

    def train_on_batch(self, epoch, train_generator, test_generator):
        steps = train_generator.__len__()

        idx = 0
        for i in range(0, epoch):
            for image, crop_param, joint_3d, joint_3d_rate, joint_2d in train_generator.getitem():
                start_time = time.time()
                joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 1, 3))
                result = self.model.train_on_batch(x=[image],
                                                   y=[joint_3d_rate, joint_3d_rate, joint_2d])

                spend_time = time.time() - start_time
                rest = steps-idx
                print('epoch:{0} iteration{1}/{2}'.format(i, idx, steps),
                      result,
                      "rest: {0:.2f}".format(spend_time * rest))

                idx = (idx + 1) % steps

            self.test_on_batch(test_generator, i+1)

    def test_on_batch(self, test_generator, epoch):
        root = "D:\\RegNet\\result\\{0}".format(epoch)
        os.makedirs(root, exist_ok=True)
        idx = 0
        sum_result = [0.0,0.,0.,0.,0.,0.,0.]
        for image, crop_param, joint_3d, joint_3d_rate, joint_2d in test_generator.getitem():
            joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 1, 3))
            result = self.model.test_on_batch(x=[image],
                                              y=[joint_3d_rate, joint_3d_rate, joint_2d])
            result = np.asarray(result)
            sum_result = np.asarray(sum_result)
            sum_result += result
            idx += 1
        sum_result /= idx
        if self.min_loss[0] > sum_result[0]:
            self.min_loss = sum_result
            print(epoch, self.min_loss)
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
                                                                                     value2[0, 0], value2[0, 1], value2[0, 2],
                                                                                     abs(value1[0]-value2[0,0]) +
                                                                                     abs(value1[1]-value2[0,1]) +
                                                                                     abs(value1[2]-value2[0,2]))
                        o3r_file.writelines(val_str)
                    o3r_file.close()

                    result_image = heatmap[i,:,:,0]
                    for j in range(1,21):
                        result_image += heatmap[i,:,:,j]
                    plt.imsave(root + "\\joint_{0}_{1}.png".format(idx, i+1), result_image)
                self.model.save(root+"\\regnet.h5".format(epoch))

                idx = (idx + 1)

def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

def gaussian_heat_map(x, N):
    X = np.linspace(0, N, N)
    Y = np.linspace(0, N, N)
    X, Y = np.meshgrid(X, Y)
    mu = np.array([x[0], x[1]])
    Sigma = np.array([[ 3.0 , 0.], [0.,  3.]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = multivariate_gaussian(pos, mu, Sigma)
    return Z

def make_dir_path(root_path):
    pathes = []
    no_object = root_path + "\\data\\noObject"
    for i in range(1,141):
        end = 1025
        if i == 69:
            end = 217
        for j in range(1,end):
            pathes.append(no_object +"\\{0:04d}\\{1:04d}".format(i,j))
    with_object = root_path + "\\data\\withObject"
    for i in range(1, 184):
        end = 1025
        if i == 92:
            end = 477
        for j in range(1, end):

            pathes.append(with_object + "\\{0:04d}\\{1:04d}".format(i, j))

    return pathes


if __name__ == "__main__":
    print("Hello")