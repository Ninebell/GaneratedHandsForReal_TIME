'''
                         -------------------------------------------------------------------> conv
    RegNet <=   ResNet50 -> intermediate 3D positions -> ProjLayer -> rendered 2D heatmaps -> conv  -> 2D heatmaps
                   |                                                                                -> 3D positions
                   |                                                                                       |
                   ▼                                                                                       ▼
                ground truth    (L2 loss)                                                               (L2 loss)
'''
from keras.layers import *
from keras.models import Model
from keras import backend as k_b
import numpy as np


def change3D_2D(points, crop_param):
    intrinsics = [[617.173, 0, 315.453],
                  [0, 617.173, 242.256],
                  [0, 0, 1]]
    intrinsics = np.asarray(intrinsics, np.float64)
    k = np.dot(intrinsics, points)
    points_2d = ((k[0]/k[2]-crop_param[0])*crop_param[2], (k[1]/k[2]-crop_param[1])*crop_param[2])
    return points_2d


class ProjLayer(Layer):
    def __init__(self, input_size, batch_size, **kwargs):
        self.input_size = input_size
        self.batch_size = batch_size
        super(ProjLayer, self).__init__(**kwargs)

    def calc_cell_units(self):
        return self.input_size[0]*self.input_size[1]

    def build(self, input_shape):
        self.ones = k_b.ones((21, self.calc_cell_units(), 2))

        self.intrinsics = [[617.173, 0, 315.453],
                           [0, 617.173, 242.256],
                           [0, 0, 1]]

        self.crop_value = []

        self.intrinsics_tensor = k_b.ones((3,3))
        k_b.set_value(self.intrinsics_tensor, self.intrinsics)
        self.intrinsics_tensor = k_b.reshape(self.intrinsics_tensor, (1, 3, 3))

        pair = []
        for i in range(0, self.calc_cell_units()):
            pair.append((i%self.input_size[0], i//self.input_size[1]))
        pair = np.asarray(pair)
        self.back_board = k_b.ones((self.calc_cell_units(), 2))
        k_b.set_value(self.back_board, pair)
        # self.back_board = k_b.repeat(self.back_board, 21)
        # self.back_board = k_b.reshape(self.back_board, (21,self.calc_cell_units(),2))

#        self.back_board = k_b.repeat(self.back_board, 3)
        print(self.back_board.shape)
        super(ProjLayer, self).build(input_shape)

    def call(self, x):
        # x = k_b.reshape(x, (-1, -1, 6))
        # joint_3d = x[:, :, :3]
        # crop_prom = x[:, :, 3:]
        joint_3d = x[0]
        print('joint3d', joint_3d.shape)
        # joint_3d = k_b.transpose(joint_3d)
#        joint_3d = k_b.reshape(joint_3d, (-1, 3, 21))
        crop_prom = x[1]
        print("crop_prom: ",crop_prom.shape)
#        crop_prom = k_b.reshape(crop_prom, (-1, 1, 3))
#        joint_3d = k_b.transpose(joint_3d)

        # print('joint3d_transpose: ', joint_3d.shape)
        self.intrinsics_tensor = k_b.transpose(self.intrinsics_tensor)
        global_joint_2d = k_b.dot(joint_3d, self.intrinsics_tensor)
        print('shape: ', global_joint_2d.shape)

        global_joint_2d = k_b.reshape(global_joint_2d, (-1, 21, 3))

        print('shape: ', global_joint_2d.shape)
        scale = k_b.repeat(global_joint_2d[:, :, 2], 2)
        scale = k_b.reshape(scale, (-1, 21, 2))
        print('scale: ', scale.shape)
        global_joint_2d = global_joint_2d[:, :, :2] / scale
        print("global_joint_2d: ", global_joint_2d.shape)
        joint_2d = (global_joint_2d - crop_prom[:, :, :2])*crop_prom[:, :, 2]
        joint_2d = k_b.reshape(joint_2d,(-1,21,1,2))
        print("joint_2d: ", joint_2d.shape)
        print("joint_2d ones: ", (joint_2d * self.ones).shape)
        print("back_board: ", self.back_board.shape)
        joint_2d_ones = joint_2d * self.ones
        diff = (joint_2d_ones - self.back_board)
        print("diff: ", diff.shape)
        fac = K.square(diff[:, :, 0]) + K.square(diff[:, :, 1])
        son_value = k_b.exp(-fac/2)
        mom_value = (2*np.pi)
        print(fac.shape)
        result = k_b.reshape(son_value/mom_value, (-1, self.input_size[0], self.input_size[1], 21))
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.input_size[0], self.input_size[1], 1)

class RegNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.__build__()

    def __build__(self):
        image_input_layer = Input(self.input_shape)
        print(image_input_layer.shape)
        crop_param_input_layer = Input((1,3))
        res4c = self.__build__resnet__(image_input_layer)
        self.intermediate_3D_position = RegNet.make_intermediate_3D_position(res4c)
        print(self.intermediate_3D_position.shape, crop_param_input_layer.shape)
#        merge = k_b.concatenate([self.intermediate_3D_position, crop_param_input_layer])
#        self.intermediate_3D_position = RegNet.make_intermediate_3D_position(image_input_layer)
        projLayer = ProjLayer((256,256), batch_size=2)([self.intermediate_3D_position, crop_param_input_layer])
        conv = RegNet.make_conv(projLayer)
        output_layer1, output_layer2 = RegNet.make_main_loss(conv)
        return Model(inputs=[image_input_layer,crop_param_input_layer], outputs=[output_layer1, output_layer2])

    def __build__resnet__(self, input_layer):
        feature = 64
        conv1 = RegNet.conv_block(7,1,feature,input_layer)
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

        conv = Conv2D(kernel_size=3, strides=1, filters=1, padding='same')(input_layer)
        # deconv = RegNet.deconv_block(4, 2, 256*256, conv)
        # deconv = RegNet.deconv_block(4, 2, 256*256, deconv, True)
        return conv, inner3joints

    @staticmethod
    def make_conv(input_layer):
        print(input_layer.shape)
        conv4e = RegNet.conv_block(3,1,512,input_layer)
        conv4f = RegNet.conv_block(3,1,256, conv4e)
        return conv4f

    @staticmethod
    def deconv_block(k, s, fm, input_layer, fixed=False):
        return Deconv2D(filters=fm,kernel_size=k,strides=s,padding='same')(input_layer)

    @staticmethod
    def conv_block(k, s, f, input_layer):
        print('what',input_layer.shape)
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
        inner = k_b.reshape(inner,(-1, 21, 3))
        return inner

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


if __name__ == "__main__":
   nope=[100,100]
    z = gaussian_heat_map(nope)

    input = Input(shape=(1,6))
#    flat = Flatten()(input)

   gaus = ProjLayer((256,256))(input)
    minuses = []
#    conv = Conv2D(filters=1, kernel_size=2, strides=1, padding='valid',trainable=False)(input)

    model = Model(inputs=[input], outputs=[gaus])

    model.compile(optimizer='adam',loss="mae")

    y = np.arange(1,256*256+1,1)
    y = np.reshape(y, (256, 256))
    ys = []
    ys.append(y)
    ys.append(y)
    ys.append(y)
    ys.append(y)
    ys = np.asarray(ys)

    model.summary()
    y = model.predict(x)

    print("***************************************************************")
    print(y[0])
    print("***************************************************************")
    points = [-54.176, 7.2007, 375.21]
    crop_param = [217.71,121.36,0.84725]
    print(gaussian_heat_map((7.6291,112.74)))
