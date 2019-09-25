'''
'''
from keras.applications import resnet50
from keras.layers import *
from keras.models import Model
from RegNet.projLayer import ProjLayer, RenderingLayer, ReshapeChannelToLast

class RegNet:
    def __init__(self, input_shape):
        input_layer = Input(input_shape)
        print(input_layer.shape)
        resnet = resnet50.ResNet50(input_tensor=input_layer, weights='imagenet', include_top=False)
        conv = RegNet.make_conv(resnet.output)
        print(conv.shape)
        heatmap_4f = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', name='heatmap_4f')(conv)
        print(heatmap_4f.shape)
        heatmap_4f_up = Deconv2D(filters=22,
                                 kernel_size=4,
                                 strides=2,
                                 padding='same',
                                 name='heatmap_4f_up')(heatmap_4f)
        print(heatmap_4f_up.shape)
        heatmap_before_proj = Deconv2D(filters=22,
                                       kernel_size=4,
                                       strides=2,
                                       padding='same',
                                       name='heatmap_beofre_proj')(heatmap_4f_up)
        print(heatmap_before_proj.shape)

        flat = Flatten()(conv)
        print(flat.shape)
        fc_joints3d_1_before_proj = Dense(200, name='fc_joints3d_1_before_proj')(flat)
        print(fc_joints3d_1_before_proj.shape)
        joints3d_prediction_before_proj = Dense(63, name='joints3d_prediction_before_proj')(fc_joints3d_1_before_proj)
        print(joints3d_prediction_before_proj.shape)
        reshape_joints3D_before_proj = Reshape((21,1,3), name='reshape_joints3D_before_proj')(joints3d_prediction_before_proj)
        print(reshape_joints3D_before_proj.shape)
        projLayer = ProjLayer()(reshape_joints3D_before_proj)
        print(projLayer.shape)
        heatmaps_pred3D = RenderingLayer([32,32], coeff=1, name='heatmaps_pred3D')(projLayer)
        print(heatmaps_pred3D.shape)
        heatmaps_pred3D = ReshapeChannelToLast()(heatmaps_pred3D)
        print(heatmaps_pred3D.shape)

        conv_rendered_2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(heatmaps_pred3D)
        print(conv_rendered_2.shape)
        conv_rendered_3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(conv_rendered_2)

        concat_pred_rendered = concatenate([conv_rendered_3, resnet.output])
        conv_rendered_4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(concat_pred_rendered)
        heatmap_prefinal_small = Conv2D(filters=64, kernel_size=3,strides=1,padding='same')(conv_rendered_4)
        heatmap_prefinal = Deconv2D(filters=22, kernel_size=4, strides=2, padding='same', name='heatmap_prefinal')(heatmap_prefinal_small)
        heatmap_final = Deconv2D(filters=22, kernel_size=4, strides=2, padding='same', name='heatmap_final')(heatmap_prefinal)

        flat = Flatten()(conv_rendered_4)
        fc_joints3D_1_final = Dense(200, name='fc_joints3D_1_final')(flat)
        joints3D_final = Dense(63, name='joints3D_prediction_final')(fc_joints3D_1_final)
        joints3D_final_vec = Reshape((21,1,3))(joints3D_final)

        self.model = Model(inputs=input_layer, output=[heatmap_before_proj, heatmap_final, joints3D_final_vec])
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

if __name__ == "__main__":
    K.set_image_data_format('channels_last')
    RegNet((256, 256, 3))
