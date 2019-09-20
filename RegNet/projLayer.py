from keras.layers import *
import keras.backend as k_b


class ProjLayer(Layer):
    def __init__(self, **kwargs):
        super(ProjLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, 2),
                                      initializer='uniform',
                                      trainable=True)
        super(ProjLayer, self).build(input_shape)

    def call(self,x):
        return k_b.tanh(K.dot(x, self.kernel))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)


# 2d point [-1,1] to rendering gaussian 2D heat_map
class RenderingLayer(Layer):
    def __init__(self, output_shape, coeff, **kwargs):
        self.output_size = output_shape
        self.coeff = coeff

        self.ones = k_b.ones((21, 1, 2))
        self.board_ones = k_b.ones((21, self.calc_cell_units(), 2))

        # self.ones = k_b.ones((2,1,21))
        # self.board_ones = k_b.ones((2, self.calc_cell_units(), 21))

        pair = []
        for i in range(0, self.calc_cell_units()):
            pair.append((i%self.output_size[0], i//self.output_size[1]))
        pair = np.asarray(pair)
        self.back_board = k_b.ones((self.calc_cell_units(), 2))
        print(pair.shape)
        k_b.set_value(self.back_board, pair)
        super(RenderingLayer, self).__init__(**kwargs)

    def calc_cell_units(self):
        return self.output_size[0]*self.output_size[1]

    def build(self, input_shape):
        super(RenderingLayer, self).build(input_shape)

    def call(self, x):

        joint_2d = x
        joint_2d = k_b.reshape(joint_2d, [-1, 21, 1, 2])                # -1, 21, 1, 2
        # joint_2d = k_b.reshape(joint_2d, [-1, 2, 1, 21])                # -1, 21, 1, 2
        joint_2d = joint_2d + self.ones
        joint_2d = joint_2d * 127.5
        joint_2d_ones = joint_2d * self.board_ones

        diff = (joint_2d_ones - self.back_board)                        # -1, 21, 65535, 2 - -1, 21, 65535, 2
        fac = (k_b.square(diff[:, :, :, 0]) + k_b.square(diff[:, :, :, 1])) / (self.coeff)
        son_value = k_b.exp(-fac/2.0)
        mom_value = (2.0*np.pi) * (self.coeff)

        result = son_value/mom_value
        result = k_b.reshape(result, [-1, 21, 256, 256])
        return result

    def compute_output_shape(self, input_shape):
        input_a = input_shape
        return (input_a[0], 21, self.output_size[0], self.output_size[1])
#
#
# class ProjLayer(Layer):
#     def __init__(self, output_size, coeff_variance, **kwargs):
#         self.output_size = output_size
#         self.variance = coeff_variance
#         super(ProjLayer, self).__init__(**kwargs)
#
#     def calc_cell_units(self):
#         return self.output_size[0]*self.output_size[1]
#
#     def build(self, input_shape):
#         super(ProjLayer, self).build(input_shape)
#     def call(self, x):
#         # joint_3d = x[0]             #   -1, 21, 3
#         # crop_prom = x[1]            #   -1, 1, 3
#         # #
#         #
#         # global_joint_2d = k_b.dot(joint_3d, self.intrinsics_tensor)     # -1, 21, 3 X 3, 3 = -1, 21, 1, 3
#         # global_joint_2d = k_b.reshape(global_joint_2d, [-1, 21, 3])     # -1, 21, 3
#         #
#         # #
#         # scale = global_joint_2d[:,:,2]                                  # -1, 21
#         # scale = k_b.reshape(scale, [-1,21,1])                           # -1, 21, 1
#         #
#         # global_joint_2d = global_joint_2d[:, :, :2] / scale             # -1, 21, 2
#         #
#         # joint_2d = (global_joint_2d - crop_prom[:, :, :2])   # -1, 21, 2
#
#         # b = k_b.repeat(crop_prom[:,:,2], 2)
#         # b = k_b.reshape(b, [-1,1,2])
#         # joint_2d = joint_2d * b
#         #
#
#         joint_2d = (x+1) * self.output_size/2
#         # joint_2d[:,:,0] = (joint_2d[:,:,0]+1)
#         # joint_2d[:,:,0] *= self.output_shape[0]/2
#         # joint_2d[:,:,1] = (joint_2d[:,:,1]+1)*self.output_shape[1]/2
#         joint_2d = k_b.reshape(joint_2d, [-1, 21, 1, 2])                # -1, 21, 1, 2
#         joint_2d_ones = joint_2d * self.ones
#
#         diff = (joint_2d_ones - self.back_board)                        # -1, 21, 65535, 2 - -1, 21, 65535, 2
#         fac = (k_b.square(diff[:, :, :, 0]) + k_b.square(diff[:, :, :, 1])) / (self.variance)
#         son_value = k_b.exp(-fac/2.0)
#         mom_value = (2.0*np.pi) * (self.variance)
#
#         result = son_value/mom_value
#         result = k_b.reshape(result, [-1,self.output_shape[0],self.output_shape[1],1])
#         return result
#
#     def compute_output_shape(self, input_shape):
#         input_a, input_b = input_shape
#         return (input_a[0], self.output_shape[0], self.output_shape[1], self.output_shape[2])

