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
from keras.layers import *
from keras.models import Model
from keras import backend as k_b
from keras.utils import Sequence, plot_model
from RegNet.projLayer import *
import matplotlib.pyplot as plt
import numpy as np


def change3D_2D(points, crop_param):
    intrinsics = [[617.173, 0, 315.453],
                  [0, 617.173, 242.256],
                  [0, 0, 1]]
    intrinsics = np.asarray(intrinsics, np.float64)
    k = np.dot(intrinsics, points)
    points_2d = ((k[0]/k[2]-crop_param[0])*crop_param[2], (k[1]/k[2]-crop_param[1])*crop_param[2])
    return points_2d

class DataGenerator(Sequence):
    def __init__(self, dir_path, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dir_path = dir_path
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dir_path)) // self.batch_size)

    def __getitem__(self, item):
        indexes = self.indexes[item*self.batch_size:(item+1)*self.batch_size]
        dir_path = [self.dir_path[i] for i in indexes]
        return self.__data_generation(dir_path)

    def getitem(self):
        for i in range(0,len(self.dir_path)/self.batch_size):
            i = i*self.batch_size
            dir_path = [self.dir_path[j] for j in range(i,self.batch_size + i)]
            yield self.__data_generation(dir_path)
        self.on_epoch_end()

    def __data_generation(self, dir_path):
        image = [np.asarray(Image.open(path+"_color_composed.png")) for path in dir_path]
        image = np.asarray(image, np.float)
        image = image / 255.0
        x = []
        y = []
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
                heat_map = gaussian_heat_map(val/8, 32)
                joint_2d_heatmap.append(heat_map)

        crop_param = np.asarray(crop_param)
        crop_param = np.reshape(crop_param, (-1, 1, 3))

        joint_3d = np.asarray(joint_3d)
        joint_3d = np.reshape(joint_3d, (-1,63))

        joint_3d_rate = np.asarray(joint_3d_rate)
        joint_2d_heatmap = np.asarray(joint_2d_heatmap)
        joint_2d_heatmap = np.reshape(joint_2d_heatmap, (-1, 21, 32, 32))
        joint_2d_heatmap = np.moveaxis(joint_2d_heatmap, 1, 3)

        return image, crop_param, joint_3d, joint_3d_rate, joint_2d_heatmap

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dir_path))
        if self.shuffle:
            np.random.shuffle(self.indexes)

class Length2Rate(Layer):
    def __init__(self, **kwargs):
        super(Length2Rate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Length2Rate, self).build(input_shape)

    def call(self, x):
        w = x[:,0]
        m0 = x[:,9]
        distance = k_b.square(w - m0)
        distance = k_b.sqrt(distance[:,0] + distance[:,1] + distance[:,2])
        distance = k_b.reshape(distance, (-1, 1))
        distance = k_b.repeat(distance, 21)
        m0 = k_b.repeat(m0, 21)
        result = (x-m0)/distance
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])
#
# class ProjLayer(Layer):
#     def __init__(self, output_shape, **kwargs):
#         self.output_size = output_shape
#
#         self.ones = k_b.ones((21, self.calc_cell_units(), 2))
#
#         self.intrinsics = [[617.173, 0, 0.],
#                            [0., 617.173, 0],
#                            [315.453, 242.256, 1]]
#
#         self.intrinsics_tensor = k_b.ones((3,3), dtype='float32')
#         k_b.set_value(self.intrinsics_tensor, self.intrinsics)
#         self.intrinsics_tensor = k_b.reshape(self.intrinsics_tensor, (1, 3, 3))
#
#         pair = []
#         for i in range(0, self.calc_cell_units()):
#             pair.append((i%self.output_size[0], i//self.output_size[1]))
#         pair = np.asarray(pair)
#         self.back_board = k_b.ones((self.calc_cell_units(), 2))
#         k_b.set_value(self.back_board, pair)
#         super(ProjLayer, self).__init__(**kwargs)
#
#     def calc_cell_units(self):
#         return self.output_size[0]*self.output_size[1]
#
#     def build(self, input_shape):
#         super(ProjLayer, self).build(input_shape)
#
#     def call(self, x):
#         joint_3d = x[0]             #   -1, 21, 3
#         crop_prom = x[1]            #   -1, 1, 3
#         #
#
#         global_joint_2d = k_b.dot(joint_3d, self.intrinsics_tensor)     # -1, 21, 3 X 3, 3 = -1, 21, 1, 3
#         global_joint_2d = k_b.reshape(global_joint_2d, [-1, 21, 3])     # -1, 21, 3
#
#         #
#         scale = global_joint_2d[:,:,2]                                  # -1, 21
#         scale = k_b.reshape(scale, [-1,21,1])                           # -1, 21, 1
#
#         global_joint_2d = global_joint_2d[:, :, :2] / scale             # -1, 21, 2
#
#         joint_2d = (global_joint_2d - crop_prom[:, :, :2])   # -1, 21, 2
#
#         b = k_b.repeat(crop_prom[:,:,2],2)
#         b = k_b.reshape(b, [-1,1,2])
#         joint_2d = joint_2d * b
#
#         joint_2d = k_b.reshape(joint_2d, [-1, 21, 1, 2])                # -1, 21, 1, 2
#         joint_2d_ones = joint_2d * self.ones
#
#         diff = (joint_2d_ones - self.back_board)                        # -1, 21, 65535, 2 - -1, 21, 65535, 2
#         coeff = 10.0
#         fac = (k_b.square(diff[:, :, :, 0]) + k_b.square(diff[:, :, :, 1])) / (coeff)
#         son_value = k_b.exp(-fac/2.0)
#         mom_value = (2.0*np.pi) * (coeff)
#
#         result = son_value/mom_value
#         result = k_b.reshape(result, [-1,21,256,256])
#         return result
#
#     def compute_output_shape(self, input_shape):
#         input_a, input_b = input_shape
#         return (input_a[0], 21, self.output_size[0], self.output_size[1])
#


class RegNet:
    residual_conv_index=1
    residual_skip_index=1
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.__build__()

    def __build__(self):
        image_input_layer = Input(self.input_shape)
        res4c = RegNet.__build__resnet__(image_input_layer)
        intermediate_3D_rate = RegNet.make_intermediate_3D_position(res4c)
        projLayer = ProjLayer()(intermediate_3D_rate)
        projLayer = RenderingLayer(output_shape=[256,256], coeff=10)(projLayer)
        # intermediate_3D_rate = Length2Rate(name='intermediate_3D')(intermediate_3D_position)
        # projLayer = ProjLayer((256,256), trainable=False)([intermediate_3D_position, crop_param_input_layer])

        # projLayer_conv = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(projLayer)
        # projLayer_conv = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(projLayer_conv)
        # projLayer_conv = Conv2D(filters=1024, kernel_size=3, strides=2, padding='same')(projLayer_conv)
        conv = Conv2D(kernel_size=3, strides=1, padding='same', filters=32, activation='relu')(projLayer)
        max_pool = MaxPool2D()(conv)
        conv = Conv2D(kernel_size=3, strides=1, padding='same', filters=64, activation='relu')(max_pool)
        max_pool = MaxPool2D()(conv)
        conv = Conv2D(kernel_size=3, strides=1, padding='same', filters=128, activation='relu')(max_pool)
        max_pool = MaxPool2D()(conv)
        # conv = Conv2D(kernel_size=3, strides=1, padding='same', activation='relu')(max_pool)
        # max_pool = MaxPool2D()(conv)

        concat = concatenate([max_pool, res4c], axis=1)
        conv = RegNet.make_conv(max_pool)
        joint_3d_result, heat_map = RegNet.make_main_loss(conv)
        return Model(inputs=[image_input_layer],
                     outputs=[intermediate_3D_rate, joint_3d_result, heat_map])

    def train_on_batch(self, train_generator, test_generator):
        steps = train_generator.__len__()

        idx = 0
        test_idx = 1
        for image, crop_param, joint_3d, joint_3d_rate, joint_2d in train_generator.getitem():
            start_time = time.time()

            joint_3d_rate = np.reshape(joint_3d_rate, (-1, 21, 3))
            joint_2d = np.reshape(joint_2d, (-1, 21, 256, 256))
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

            joint = result[2][0][0]
            for t in result[2][0]:
                joint += t

            joint *= 128

            image = np.moveaxis(image[0], 0, 2)
            print('joint', np.sum(joint), )
            cv2.imwrite(root+"\\joint_{0}.png".format(idx), joint)
            plt.imsave(root+"\\image_{0}.png".format(idx), image)

            idx = (idx + 1)

    @staticmethod
    def __build__resnet__(input_layer):
        feature = 64
        length = 128
        conv1 = RegNet.conv_block(7, 1, feature, input_layer)
        max_pool = MaxPooling2D()(conv1)
        print(max_pool.shape)
        # res2a = RegNet.residual_block_convolution(1, feature, feature*4, max_pool)
        res2a = RegNet.residual_block_convolution(1, feature, feature*4, feature, length)(max_pool)
        res2b = RegNet.residual_block_identity_skip(feature, feature*4, length)(res2a)
        res2c = RegNet.residual_block_identity_skip(feature, feature*4, length)(res2b)

        # res3a = RegNet.residual_block_convolution(2, feature*2, feature*8, res2c)

        print(res2c.shape)
        res3a = RegNet.residual_block_convolution(2, feature*2, feature*8, feature*4, length)(res2c)
        res3b = RegNet.residual_block_identity_skip(feature*2, feature*8, length/2)(res3a)
        res3c = RegNet.residual_block_identity_skip(feature*2, feature*8, length/2)(res3b)

        # res4a = RegNet.residual_block_convolution(2, feature*4, feature*16, res3c)
        res4a = RegNet.residual_block_convolution(2, feature*4, feature*16, feature*8, length/2)(res3c)
        res4b = RegNet.residual_block_identity_skip(feature*4, feature*16, length/4)(res4a)
        res4c = RegNet.residual_block_identity_skip(feature*4, feature*16, length/4)(res4b)
        res4d = RegNet.residual_block_identity_skip(feature*4, feature*16, length/4)(res4c)
        return res4d

    @staticmethod
    def make_main_loss(input_layer):
        conv = Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(UpSampling2D()(input_layer))
        conv = Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(UpSampling2D()(conv))
        conv = Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu')(UpSampling2D()(conv))
        heat_map = Conv2D(kernel_size=3, strides=1, filters=21, padding='same', activation='sigmoid', name='heat_map')(conv)

        inner200 = RegNet.inner_product(input_layer, 200)
        inner3joints = RegNet.inner_product(inner200,3*21,flatten=False)
        inner3joints = Reshape((21, 3), name='inner3joints')(inner3joints)
        return inner3joints, heat_map

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

    @staticmethod
    def residual_block_identity_skip(f1, f2, length):
        input_layer = Input((f2, length, length))
        conv = RegNet.conv_block(1, 1, f1, input_layer)
        relu = ReLU()(conv)
        conv = RegNet.conv_block(3, 1, f1, relu)
        relu = ReLU()(conv)
        conv = RegNet.conv_block(1, 1, f2, relu)
        # To-Do
        # Actually, In paper I have to use Eltwise SUM. But I don't know about it.
        add = Add()([input_layer, conv])
        relu = ReLU()(add)
        model = Model(inputs=input_layer, outputs=relu, name='res_block_identity_{0}'.format(RegNet.residual_skip_index))
        RegNet.residual_skip_index += 1
        return model

    @staticmethod
    def residual_block_convolution(s, f1, f2, f3, length):
        input_layer = Input((f3, length, length))
        conv1 = RegNet.conv_block(1, s, f1, input_layer)
        relu = ReLU()(conv1)
        conv1 = RegNet.conv_block(3, 1, f1, relu)
        relu = ReLU()(conv1)
        conv1 = RegNet.conv_block(1, 1, f2, relu)
        # To-Do
        # Actually, In paper, I have to use Eltwise SUM. But I don't know about it.
        conv2 = RegNet.conv_block(1, s, f2, input_layer)
        add = Add()([conv2, conv1])
        relu = ReLU()(add)
        print('input', input_layer.shape)
        print('output', relu.shape)
        model = Model(inputs=input_layer, outputs=relu, name='res_block_conv_{0}'.format(RegNet.residual_conv_index))
        RegNet.residual_conv_index += 1
        return model

#    |      Caffe        |          Keras            |
#    |    InnerProduct   |       Fully-Connected     |
#    |    EuclideanLoss  |            L2             |
    @staticmethod
    def inner_product(input_layer, output_num, activation=None, flatten=True):
        if flatten:
            input_layer = Flatten()(input_layer)
        dense = Dense(output_num, activation=activation)(input_layer)
        return dense

    @staticmethod
    def make_intermediate_3D_position(input_layer):
        inner = RegNet.inner_product(input_layer, 3*21)
        reshape_block = Reshape(target_shape=(21, 3))(inner)
        return reshape_block


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

def make_dir_path():
    pathes = []
    # root_path = "C:\\Users\\Jonghoe\\Downloads\\GANeratedDataset_v3\\GANeratedHands_Release"
    root_path = "D:\\GANeratedDataset_v3\\GANeratedHands_Release"
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
