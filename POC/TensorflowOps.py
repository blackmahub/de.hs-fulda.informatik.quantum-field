import ConvKernels as ck

import tensorflow as tf
import numpy as np

import time as t

class TensorflowOps:
    sess = tf.InteractiveSession()

    def __init__(self):
        pass

    def __del__(self):
        self.sess.close()

    # calculating using tensorflow
    def define_tf_roll_graph(self, CONST_m):
        arr = tf.placeholder(dtype=tf.float32)
        arr_size = tf.placeholder(dtype=tf.float32)

        tleft = tf.manip.roll(arr, shift=-1, axis=1)
        tright = tf.manip.roll(arr, shift=1, axis=1)

        xleft = tf.manip.roll(arr, shift=-1, axis=2)
        xright = tf.manip.roll(arr, shift=1, axis=2)

        yleft = tf.manip.roll(arr, shift=-1, axis=3)
        yright = tf.manip.roll(arr, shift=1, axis=3)

        zleft = tf.manip.roll(arr, shift=-1, axis=4)
        zright = tf.manip.roll(arr, shift=1, axis=4)

        common = arr * (-8 + (CONST_m ** 2 / 2) * arr)

        total = tleft + tright + xleft + xright + yleft + yright + zleft + zright + common

        S = tf.reduce_sum(total) / arr_size

        return S, arr, arr_size


    def calculate_action_tf_roll(self, S, placeholder_dict, print_all):
        tf_action_roll = self.sess.run(S, feed_dict=placeholder_dict)
        if print_all:
            print("S_TF_Roll: "+str(tf_action_roll))
        return tf_action_roll


    def define_conv_kernels(self, CONST_m, arrShape, arrSize):

        kernel_size = [1, arrShape[1], arrShape[2], arrShape[3], arrShape[4]]

        global mass_tf
        mass_tf = tf.constant(CONST_m, dtype=tf.float32, name="const_mass")

        global imag_zeros
        imag_zeros = tf.constant(0, dtype=tf.float32, name="const_imag_zeros")

        global total_volume
        total_volume = tf.constant(arrSize, dtype=tf.float32, name="const_total_volume")

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
        kernel = tf.cast(tf.sparse_tensor_to_dense(tf.SparseTensor(indices, values, kernel_size)), tf.float32, name="kernel")

        global kernel_fft
        kernel_fft = tf.Variable(initial_value=tf.fft(tf.complex(kernel, imag_zeros)), name="var_kernel_fft")


    def define_conv_action_graph(self, CONST_m, arr):

        arrVar = tf.Variable(initial_value=arr, dtype=tf.float32, name="arr")

        arr_fft = tf.fft(tf.complex(arrVar, imag_zeros), name="arr_fft")

        conv_axis = tf.real(tf.ifft(tf.multiply(arr_fft, kernel_fft)), name="op_convolve_axis")
        common = tf.multiply(tf.multiply(arrVar, arrVar), tf.divide(tf.multiply(mass_tf, mass_tf), 2), name="op_common")
        conv_action = tf.divide(tf.reduce_sum(tf.add(conv_axis, common)), total_volume, name="graph_conv_action")
        
        return conv_action


    def calculate_action_tf_convolve(self, graph, print_all):
        self.sess.run(tf.global_variables_initializer())
        tf_action_conv = self.sess.run(graph)
        if print_all:
            print("S_TF_Conv: "+str(tf_action_conv))
        return tf_action_conv