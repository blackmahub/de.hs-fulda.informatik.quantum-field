import ConvKernels as ck

import tensorflow as tF
import numpy as np

class TensorflowOps:
    sess = tF.InteractiveSession()
    # kernel_t_fft, kernel_x_fft, kernel_y_fft, kernel_z_fft, imag_zeros

    def __init__(self):
        self.initialize_kernels()
    
    def initialize_kernels(self):
        kernels = ck.ConvKernels()
        self.imag_zeros = tF.Variable(kernels.imag_zeros)
        self.kernel_t_fft = tF.Variable(tF.fft(tF.complex(kernels.CONST_ConvKernelt, self.imag_zeros)))
        self.kernel_x_fft = tF.Variable(tF.fft(tF.complex(kernels.CONST_ConvKernelx, self.imag_zeros)))
        self.kernel_y_fft = tF.Variable(tF.fft(tF.complex(kernels.CONST_ConvKernely, self.imag_zeros)))
        self.kernel_z_fft = tF.Variable(tF.fft(tF.complex(kernels.CONST_ConvKernelz, self.imag_zeros)))
        tF.initialize_all_variables()

    # calculating using tensorflow
    def define_tf_roll(self, arr, CONST_m):
        tleft = tF.manip.roll(arr, shift=-1, axis=1)
        tright = tF.manip.roll(arr, shift=1, axis=1)

        xleft = tF.manip.roll(arr, shift=-1, axis=2)
        xright = tF.manip.roll(arr, shift=1, axis=2)

        yleft = tF.manip.roll(arr, shift=-1, axis=3)
        yright = tF.manip.roll(arr, shift=1, axis=3)

        zleft = tF.manip.roll(arr, shift=-1, axis=4)
        zright = tF.manip.roll(arr, shift=1, axis=4)

        common = arr * (-8 + (CONST_m ** 2 / 2) * arr)

        total = tleft + tright + xleft + xright + yleft + yright + zleft + zright + common

        print(total)

        return tF.reduce_sum(total)


    def define_tf_graph(self, CONST_m):
        arr = tF.placeholder(dtype=tF.float64)
        S = tF.map_fn(lambda a: self.define_tf_roll(a, CONST_m), arr[:, ])

        return S, arr


    def calculate_action_tf(self, S, placeholder_dict, action_name):
        tf_action = self.sess.run([S], feed_dict=placeholder_dict)[0]
        print("S_TF_%s: %s" % (action_name, str(tf_action)))
        return tf_action


    def define_tf_convolution_for_1_lattice(self, arr, CONST_m):
        arr_fft = tF.fft(tF.complex(arr, self.imag_zeros))

        conv_axis_0 = tF.real(tF.ifft(arr_fft * self.kernel_t_fft))
        conv_axis_1 = tF.real(tF.ifft(arr_fft * self.kernel_x_fft))
        conv_axis_2 = tF.real(tF.ifft(arr_fft * self.kernel_y_fft))
        conv_axis_3 = tF.real(tF.ifft(arr_fft * self.kernel_z_fft))

        common = arr * ((CONST_m ** 2 / 2) * arr)

        return tF.reduce_sum(conv_axis_0 + conv_axis_1 + conv_axis_2 + conv_axis_3 + common) / CONST_Volume


    def define_tf_convolution(self, CONST_m):
        placeholder_arr = tF.placeholder(tF.float32)

        conv_action = tF.map_fn(lambda a: self.define_tf_convolution_for_1_lattice(a, CONST_m), placeholder_arr[:, ])

        return conv_action, placeholder_arr

