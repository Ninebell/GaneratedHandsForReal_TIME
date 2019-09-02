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
import keras.losses
class RegNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.__build__()

    def __build__(self):
        input_layer = Input(self.input_shape)
        res4c = self.__build__resnet__(input_layer)
        self.intermediate_3D_position = RegNet.make_intermediate_3D_position(res4c)

        output_layer = None
        return Model(x=[input_layer], y=[output_layer])

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
    def conv_block(k, s, f, input_layer):
        conv = Conv2D(kernel_size=k, strides=s, filters=f)(input_layer)
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
    def inner_product(input_layer, output_num):
        flat = Flatten()(input_layer)
        dense = Dense(output_num)(flat)
        return dense

    @staticmethod
    def make_intermediate_3D_position(input_layer):
        inner = RegNet.inner_product(input_layer, 3*21)
        return inner

    @staticmethod
    def make_projLayer(input_layer):
        d_layer = keras.layers.Reshape((3, 21), input_layer)
        return d_layer


