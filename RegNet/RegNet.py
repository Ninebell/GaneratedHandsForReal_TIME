'''
                         -------------------------------------------------------------------> conv
    RegNet <=   ResNet50 -> intermediate 3D positions -> ProjLayer -> rendered 2D heatmaps -> conv  -> 2D heatmaps
                   |                                                                                -> 3D positions
                   |                                                                                       |
                   ▼                                                                                       ▼
                ground truth    (L2 loss)                                                               (L2 loss)
'''
import csv
from skimage import io
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from keras import backend as k_b
from keras.utils import Sequence
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

        for i in range(0,len(self.dir_path)):
            dir_path = [self.dir_path[j] for j in range(i,self.batch_size + i)]
            yield self.__data_generation(dir_path)
        self.on_epoch_end()

    def __data_generation(self, dir_path):
        image = [io.imread(path+"_color_composed.png") for path in dir_path]
        image = np.asarray(image, np.float)
        image = image / 255.0
        x = []
        y = []
        crop_param = []
        joint_3d = []
        joint_2d_heatmap = []
        for path in dir_path:
            value = open(path+"_crop_params.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            crop_param.append(value)

            value = open(path+"_joint_pos_global.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            joint_3d.append(value)

            value = open(path+"_joint2D.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            value = np.asarray(value)
            value = np.reshape(value, (21, 2))
            for val in value:
                heat_map = gaussian_heat_map(val)
                joint_2d_heatmap.append(heat_map)

        crop_param = np.asarray(crop_param)
        crop_param = np.reshape(crop_param, (-1, 1, 3))

        joint_3d = np.asarray(joint_3d)
        joint_3d = np.reshape(joint_3d, (-1,63))

        joint_2d_heatmap = np.asarray(joint_2d_heatmap)
        joint_2d_heatmap = np.reshape(joint_2d_heatmap, (-1, 21, 256, 256))

        return image, crop_param, joint_3d, joint_2d_heatmap

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dir_path))
        if self.shuffle:
            np.random.shuffle(self.indexes)

class ProjLayer(Layer):
    def __init__(self, input_size, **kwargs):
        self.input_size = input_size

        self.ones = k_b.ones((21, self.calc_cell_units(), 2))

        self.intrinsics = [[617.173, 0, 0.],
                           [0., 617.173, 0],
                           [315.453, 242.256, 1]]

        self.intrinsics_tensor = k_b.ones((3,3))
        k_b.set_value(self.intrinsics_tensor, self.intrinsics)
        self.intrinsics_tensor = k_b.reshape(self.intrinsics_tensor, (1, 3, 3))

        pair = []
        for i in range(0, self.calc_cell_units()):
            pair.append((i%self.input_size[0], i//self.input_size[1]))
        pair = np.asarray(pair)
        self.back_board = k_b.ones((self.calc_cell_units(), 2))
        k_b.set_value(self.back_board, pair)
        super(ProjLayer, self).__init__(**kwargs)

    def calc_cell_units(self):
        return self.input_size[0]*self.input_size[1]

    def build(self, input_shape):
        super(ProjLayer, self).build(input_shape)

    def call(self, x):
        joint_3d = x[0]             #   -1, 21, 3
        # print(joint_3d.shape)
        crop_prom = x[1]            #   -1, 1, 3
        #
        # print(crop_prom.shape)
        global_joint_2d = k_b.dot(joint_3d, self.intrinsics_tensor)     # -1, 21, 3 X 3, 3 = -1, 21, 1, 3
        global_joint_2d = k_b.reshape(global_joint_2d, [-1, 21, 3])     # -1, 21, 3
        #
        scale = global_joint_2d[:,:,2]                                  # -1, 21
        scale = k_b.reshape(scale, [-1,21,1])                           # -1, 21, 1
        global_joint_2d = global_joint_2d[:, :, :2] / scale             # -1, 21, 2
        joint_2d = (global_joint_2d - crop_prom[:, :, :2])   # -1, 21, 2
        joint_2d = joint_2d * crop_prom[:,:,:2]
        joint_2d = k_b.reshape(joint_2d, [-1, 21, 1, 2])                # -1, 21, 1, 2
        joint_2d_ones = joint_2d * self.ones
        diff = (joint_2d_ones - self.back_board)                        # -1, 21, 65535, 2 - -1, 21, 65535, 2
        print(diff.shape)
        fac = k_b.square(diff[:, :, :, 0]) + k_b.square(diff[:, :, :, 1])
        son_value = k_b.exp(-fac/2)
        mom_value = (2*np.pi)

        result = son_value/mom_value
        result = k_b.reshape(result, [-1,21,256,256])
        return result

    def compute_output_shape(self, input_shape):
        input_a, input_b = input_shape
        return (input_a[0], 21, 256, 256)



class RegNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.__build__()

    def __build__(self):
        image_input_layer = Input(self.input_shape)
        crop_param_input_layer = Input(shape=(1,3))
        res4c = RegNet.__build__resnet__(image_input_layer)
        intermediate_3D_position = RegNet.make_intermediate_3D_position(res4c)
        print(intermediate_3D_position.shape)
        projLayer = ProjLayer((256,256), trainable=False)([intermediate_3D_position, crop_param_input_layer])
        conv = RegNet.make_conv(projLayer)
        joint_3d_result, heat_map = RegNet.make_main_loss(conv)
        return Model(inputs=[image_input_layer, crop_param_input_layer],
                     outputs=[intermediate_3D_position, joint_3d_result, heat_map])

    @staticmethod
    def __build__resnet__(input_layer):
        feature = 64
        conv1 = RegNet.conv_block(7, 1, feature, input_layer)
        max_pool = MaxPooling2D()(conv1)
        res2a = RegNet.residual_block_convolution(1, feature, feature*4, max_pool)
        res2b = RegNet.residual_block_identity_skip(feature, feature*4, res2a)
        res2c = RegNet.residual_block_identity_skip(feature, feature*4, res2b)

        res3a = RegNet.residual_block_convolution(2, feature*2, feature*8, res2c)
        res3b = RegNet.residual_block_identity_skip(feature*2, feature*8, res3a)
        res3c = RegNet.residual_block_identity_skip(feature*2, feature*8, res3b)

        res4a = RegNet.residual_block_convolution(2, feature*4, feature*16, res3c)
        res4b = RegNet.residual_block_identity_skip(feature*4, feature*16, res4a)
        res4c = RegNet.residual_block_identity_skip(feature*4, feature*16, res4b)
        res4d = RegNet.residual_block_identity_skip(feature*4, feature*16, res4c)
        return res4d

    @staticmethod
    def make_main_loss(input_layer):
        conv = Conv2D(kernel_size=3, strides=2, filters=256, padding='same')(input_layer)
        conv = Conv2D(kernel_size=3, strides=2, filters=512, padding='same')(conv)
        conv = Conv2D(kernel_size=3, strides=2, filters=1024, padding='same')(conv)
        inner200 = RegNet.inner_product(conv, 200)
        inner3joints = RegNet.inner_product(inner200,3*21,False)
        inner3joints = Reshape((21, 3))(inner3joints)
        heat_map = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',name='heatmaps')(input_layer)
        return inner3joints, heat_map

    @staticmethod
    def make_conv(input_layer):
        conv4e = RegNet.conv_block(5, 1, 512, input_layer)
        conv4f = RegNet.conv_block(5, 1, 256, conv4e)
        return conv4f

    @staticmethod
    def deconv_block(k, s, fm, input_layer, fixed=False):
        return Deconv2D(filters=fm, kernel_size=k, strides=s, padding='same')(input_layer)

    @staticmethod
    def conv_block(k, s, f, input_layer):
        conv = Conv2D(kernel_size=k, strides=s, filters=f, padding='same')(input_layer)
        batch = BatchNormalization()(conv)
        # To - Do
        # I need to apply Scale
        return batch

    @staticmethod
    def residual_block_identity_skip(f1, f2, input_layer):
        conv = RegNet.conv_block(1, 1, f1, input_layer)
        relu = ReLU()(conv)
        conv = RegNet.conv_block(3, 1, f1, relu)
        relu = ReLU()(conv)
        conv = RegNet.conv_block(1, 1, f2, relu)
        # To-Do
        # Actually, In paper I have to use Eltwise SUM. But I don't know about it.
        add = Add()([input_layer, conv])
        relu = ReLU()(add)
        return relu

    @staticmethod
    def residual_block_convolution(s, f1, f2, input_layer):
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
        return relu

#    |      Caffe        |          Keras            |
#    |    InnerProduct   |       Fully-Connected     |
#    |    EuclideanLoss  |            L2             |
    @staticmethod
    def inner_product(input_layer, output_num, flatten=True):
        if flatten:
            input_layer = Flatten()(input_layer)
        dense = Dense(output_num)(input_layer)
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

def gaussian_heat_map(x):
    N = 256
    X = np.linspace(0, 255, N)
    Y = np.linspace(0, 255, N)
    X, Y = np.meshgrid(X, Y)
    mu = np.array([x[0], x[1]])
    Sigma = np.array([[ 1.0 , 0.], [0.,  1.]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = multivariate_gaussian(pos, mu, Sigma)
    return Z

def make_dir_path():
    pathes = []
    no_object = "D:\\GANeratedDataset_v3\\GANeratedHands_Release\\data\\noObject"
    for i in range(1,141):
        end = 1025
        if i == 69:
            end = 217
        for j in range(1,end):
            pathes.append(no_object +"\\{0:04d}\\{1:04d}".format(i,j))
    with_object = "D:\\GANeratedDataset_v3\\GANeratedHands_Release\\data\\withObject"
    for i in range(1, 184):
        end = 1025
        if i == 92:
            end = 477
        for j in range(1, end):

            pathes.append(with_object + "\\{0:04d}\\{1:04d}".format(i, j))

    return pathes

if __name__ == "__main__2":
    dir_path = make_dir_path()
    gen = DataGenerator(dir_path, batch_size=2, shuffle=False)
    input1 = Input(shape=(21, 3))
    input2 = Input(shape=(1, 3))
    # res4c = RegNet.__build__resnet__(input1)
#    intermediate = RegNet.make_intermediate_3D_position(res4c)
    gaus = ProjLayer([256, 256])([input1, input2])
    # conv = RegNet.make_conv(gaus)
    # joint3d, heat_map = RegNet.make_main_loss(conv)
    model = Model(inputs=[input1, input2], outputs=[gaus])
    model.summary()

    model.compile(loss=['mean_squared_error'], metrics=['accuracy'], optimizer=Adam(lr=1e-4))

    i = 1
    for image, crop_param, joint_3d, joint_2d in gen.getitem():
        joint_3d = np.reshape(joint_3d, (-1, 21, 3))
        joint_2d = np.reshape(joint_2d, (-1, 21, 256, 256))
        crop_param = np.reshape(crop_param, (-1, 1, 3))

        # p = model.predict(x=[joint_3d, crop_param])
        # io.imshow(p[0,0])
        # io.show()

        p = model.train_on_batch(x=[joint_3d, crop_param], y=[joint_2d])
        print(i, p)
        i = i + 1

    for i in range(len(joint_2d[0])):

        #test = (gaussian_heat_map(joint_2d[0][i]))
        io.imshow(p[0, i])
        io.show()

        plt.imshow(joint_2d[0][i], cmap='hot', interpolation='nearest')
        plt.show()


if __name__ == "__main__":
    pathes = make_dir_path()
    np.random.shuffle(pathes)
    path_len = len(pathes)
    train_data = pathes[:path_len//10*8]
    test_data = pathes[path_len//10*8:]
    train_gen = DataGenerator(dir_path=train_data, batch_size=2)
    test_gen = DataGenerator(dir_path=test_data)

    regNet = RegNet(input_shape=(256, 256, 3))
    optimizer = Adam(lr=1e-4)
    regNet.model.compile(optimizer=optimizer,
                         loss=['mean_squared_error',
                               'mean_squared_error',
                               'mean_squared_error'],
                         )
    regNet.model.summary()
    idx = 0
    for image, crop_param, joint_3d, joint_2d in train_gen.getitem():
        joint_3d = np.reshape(joint_3d, (-1, 21, 3))
        joint_2d = np.reshape(joint_2d, (-1, 21, 256, 256))
        crop_param = np.reshape(crop_param, (-1, 1, 3))
        result = regNet.model.train_on_batch(x=[image, crop_param], y=[joint_3d, joint_3d, joint_2d])
        print(result)

    regNet.model.summary()
