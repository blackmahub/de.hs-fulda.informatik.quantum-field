import ConvKernels as ck

import tensorflow as tF
import numpy as np

import time as t

class TensorflowOps:
    sess = tF.InteractiveSession()

    def __init__(self):
        pass


    # calculating using tensorflow
    def define_tf_roll(self, CONST_m):
        arr = tF.placeholder(dtype=tF.float64)
        arr_size = tF.placeholder(dtype=tF.float64)

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

        S = tF.reduce_sum(total) / arr_size

        return S, arr, arr_size


    def calculate_action_tf(self, S, placeholder_dict, action_name):
        tf_action = self.sess.run([S], feed_dict=placeholder_dict)[0]
        print("S_TF_%s: %s" % (action_name, str(tf_action)))
        return tf_action

    # todo: precompute kernels on gpu and put that into gloabl_variables_initializer
    def define_conv_action_alt(self, CONST_m, arrShape):
        kernel_shape = arrShape

        kernel_t = np.zeros(kernel_shape, dtype=np.float32)
        kernel_t[:, 0:3, 0, 0, 0] = [1, -2, 1]
        kernel_x = np.zeros(kernel_shape, dtype=np.float32)
        kernel_x[:, 0, 0:3, 0, 0] = [1, -2, 1]
        kernel_y = np.zeros(kernel_shape, dtype=np.float32)
        kernel_y[:, 0, 0, 0:3, 0] = [1, -2, 1]
        kernel_z = np.zeros(kernel_shape, dtype=np.float32)
        kernel_z[:, 0, 0, 0, 0:3] = [1, -2, 1]

        imag_zeros = np.zeros(kernel_shape, dtype=np.float32)

        arrVar = tF.Variable(initial_value=tF.random_uniform(arrShape, -1,1, dtype=tF.float32)) ;

        arr_fft = tF.fft(tF.complex(arrVar, imag_zeros), name="arr_fft")

        conv_axis_1 = tF.multiply(tF.fft(tF.complex(kernel_t, imag_zeros)), arr_fft, name="conv_axis_1")
        conv_axis_2 = tF.multiply(tF.fft(tF.complex(kernel_x, imag_zeros)), arr_fft, name="conv_axis_2")
        conv_axis_3 = tF.multiply(tF.fft(tF.complex(kernel_y, imag_zeros)), arr_fft, name="conv_axis_3")
        conv_axis_4 = tF.multiply(tF.fft(tF.complex(kernel_z, imag_zeros)), arr_fft, name="conv_axis_4")

        convRes = tF.real(tF.ifft(conv_axis_1+conv_axis_2+conv_axis_3+conv_axis_4)) ;

        common = tF.multiply(arrVar * arrVar,(CONST_m ** 2 / 2) , name="common")
        conv_action = tF.reduce_sum(convRes + common) / np.prod(arrShape) ;
        #tF.global_variables_initializer();
        return conv_action, arrVar ;


    def define_tf_convolve(self, CONST_m, arr):

        kernel_shape = [1, arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]]

        arr_tf = tF.constant(arr, dtype=tF.float32, name="const_arr")
        mass_tf = tF.constant(CONST_m, dtype=tF.float32, name="const_mass")
        imag_zeros = tF.constant(0, dtype=tF.float32, name="const_imag_zeros")
        total_volume = tF.constant(arr.size, dtype=tF.float32, name="const_total_volume")

        indices = [
            [0, 0, 0, 0, 0],  # for all axis
            [0, 0, 0, 0, 1],  # for axis 4
            [0, 0, 0, 0, 2],  # for axis 4
            [0, 0, 0, 1, 0],  # for axis 3
            [0, 0, 0, 2, 0],  # for axis 3
            [0, 0, 1, 0, 0],  # for axis 2
            [0, 0, 2, 0, 0],  # for axis 2
            [0, 1, 0, 0, 0],  # for axis 1
            [0, 2, 0, 0, 0]   # for axis 1
        ]
        values = [1, -2, 1, -2, 1, -2, 1, -2, 1]
        kernel = tF.cast(tF.sparse_tensor_to_dense(tF.SparseTensor(indices, values, kernel_shape)), tF.float32, name="kernel")

        kernel_fft = tF.Variable(initial_value=tF.fft(tF.complex(kernel, imag_zeros)), name="var_kernel_fft")

        self.sess.run(tF.global_variables_initializer())

        arr_fft = tF.fft(tF.complex(arr_tf, imag_zeros), name="op_arr_fft")
        conv_axis = tF.real(tF.ifft(tF.multiply(arr_fft, kernel_fft)), name="op_convolve_axis")
        common = tF.multiply(tF.multiply(arr_tf, arr_tf), tF.divide(tF.multiply(mass_tf, mass_tf), 2), name="op_common")
        conv_action = tF.divide(tF.reduce_sum(tF.add(conv_axis, common)), total_volume, name="graph_conv_action")

        return conv_action

