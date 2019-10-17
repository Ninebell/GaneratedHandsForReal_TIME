from keras.layers import *
import keras.backend as k_b
import numpy as np

class ProjLayer(Layer):
    def __init__(self, **kwargs):
        self.range = 1.5
        self.heatmap_size = 32
        super(ProjLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        super(ProjLayer, self).build(input_shape)

    def call(self,x):
        return (x[:,:,:2] + self.range) / (2*self.range) * (self.heatmap_size-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)


# 2d point [-1,1] to rendering gaussian 2D heat_map
class RenderingLayer(Layer):
    def __init__(self, output_shape, coeff, **kwargs):
        self.output_size = output_shape
        self.coeff = coeff

        self.base = k_b.ones((1, 32*32), dtype=np.float)
        self.ones = k_b.ones((21, 1, 2))
        self.board_ones = k_b.ones((21, self.calc_cell_units(), 2))

        pair = []
        for i in range(0, self.calc_cell_units()):
            pair.append((i%self.output_size[0], i//self.output_size[1]))
        pair = np.asarray(pair)

        self.back_board = k_b.ones((self.calc_cell_units(),2))
        print(pair.shape)
        k_b.set_value(self.back_board, pair)
        super(RenderingLayer, self).__init__(**kwargs)

    def calc_cell_units(self):
        return self.output_size[0]*self.output_size[1]

    def build(self, input_shape):
        super(RenderingLayer, self).build(input_shape)

    def call(self, x):

        joint_2d = x
        joint_2d = k_b.reshape(joint_2d, [-1, 21, 2])                # -1, 21, 1, 2

        joint_2d = k_b.reshape(joint_2d, [-1, 21, 1, 2])                # -1, 21, 1, 2
        joint_2d_ones = joint_2d * self.board_ones

        diff = (joint_2d_ones - self.back_board)                        # -1, 21, 65535, 2 - -1, 21, 65535, 2
        fac = (k_b.square(diff[:, :, :, 0]) + k_b.square(diff[:, :, :, 1])) / (self.coeff)
        son_value = k_b.exp(-fac/2.0)
        mom_value = (2.0*np.pi) * (self.coeff)

        result = son_value / mom_value

        result = k_b.reshape(result, [-1, 21, self.output_size[0] * self.output_size[1]])
        return result

    def compute_output_shape(self, input_shape):
        input_a = input_shape
        return (input_a[0], 21, self.output_size[0], self.output_size[1])


class ReshapeChannelToLast(Layer):
    def __init__(self, **kwargs):
        self.base = k_b.ones((1, 32*32), dtype=np.float)
        super(ReshapeChannelToLast, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeChannelToLast, self).build(input_shape)

    def call(self,x):
        x = k_b.reshape(x, (-1, 21, 32*32))
        base = k_b.reshape(x[:,0,:] * self.base, (-1,32,32,1))
        for i in range(1, 21):
            test = (x[:,i,:] * self.base)
            test = k_b.reshape(test, (-1,32,32,1))
            base = k_b.concatenate([base,test])
        print(base.shape)
        return base

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3], input_shape[1])

