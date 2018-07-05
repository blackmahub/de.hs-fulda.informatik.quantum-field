
import tensorflow as tf
import numpy as np

import time as t

class TensorflowOps:

    def __init__(self, CONST_m, fieldMin, fieldMax, shape):
        self.sess = tf.Session()
        self.arrVar = tf.Variable(initial_value=tf.random_uniform(shape, fieldMin, fieldMax, dtype=tf.float32))
        self.arrSize = tf.constant(np.prod(shape), dtype=tf.float32)
        self.mass_tf = tf.constant(CONST_m, dtype=tf.float32, name="const_mass")

    def __del__(self):
        self.sess.close()

    def initialize_tensorflow_variables(self):
        self.sess.run(tf.global_variables_initializer())

    # calculating using tensorflow
    def define_tf_roll_graph(self):
        # arr = tf.placeholder(dtype=tf.float32)
        # arr_size = tf.placeholder(dtype=tf.float32)

        tleft = tf.manip.roll(self.arrVar, shift=-1, axis=1)
        tright = tf.manip.roll(self.arrVar, shift=1, axis=1)

        xleft = tf.manip.roll(self.arrVar, shift=-1, axis=2)
        xright = tf.manip.roll(self.arrVar, shift=1, axis=2)

        yleft = tf.manip.roll(self.arrVar, shift=-1, axis=3)
        yright = tf.manip.roll(self.arrVar, shift=1, axis=3)

        zleft = tf.manip.roll(self.arrVar, shift=-1, axis=4)
        zright = tf.manip.roll(self.arrVar, shift=1, axis=4)

        common = self.arrVar * (-8 + (self.mass_tf ** 2 / 2) * self.arrVar)

        total = tleft + tright + xleft + xright + yleft + yright + zleft + zright + common

        S = tf.reduce_sum(total) / self.arrSize

        return S


    def calculate_action_tf_roll(self, S, print_all):
        tf_action_roll = self.sess.run(S)
        if print_all:
            print("S_TF_Roll: "+str(tf_action_roll))
        return tf_action_roll


    def define_conv_kernels(self):

        kernel_size = [1, self.arrVar.shape[1].value, self.arrVar.shape[2].value, self.arrVar.shape[3].value, self.arrVar.shape[4].value]

        global imag_zeros
        imag_zeros = tf.constant(0, dtype=tf.float32, name="const_imag_zeros")

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


    def define_conv_action_graph(self):
        # if arr is not None:
        #     # workz but very slow; but we need this because we need to work on the same array for all the operations
        #     arrVar = tf.Variable(initial_value=arr, dtype=tf.float32, name="arr")
        # else:
            # create array variable on GPU, no transfer --> much faster
            # arrVar = tf.Variable(initial_value=tf.random_uniform(shape, field_min, field_max, dtype=tf.float32))
        # arrVar = tf.Variable(initial_value=tf.random_uniform(shape, -1, 1, dtype=tf.float32))

        arr_fft = tf.fft(tf.complex(self.arrVar, imag_zeros), name="arr_fft")

        conv_axis = tf.real(tf.ifft(tf.multiply(arr_fft, kernel_fft)), name="op_convolve_axis")
        common = tf.multiply(tf.multiply(self.arrVar, self.arrVar), tf.divide(tf.multiply(self.mass_tf, self.mass_tf), 2), name="op_common")
        conv_action = tf.divide(tf.reduce_sum(tf.add(conv_axis, common)), self.arrSize, name="graph_conv_action")

        return conv_action


    def calculate_action_tf_convolve(self, graph, print_all):
        # took kernel initialization out of time measurement. That was actually a huge perf boost
        # as kernels should not be considered here, they can be loaded from disk one or similar
        #self.sess.run(tf.global_variables_initializer())
        tf_action_conv = self.sess.run(graph)
        if print_all:
            print("S_TF_Conv: "+str(tf_action_conv))
        return tf_action_conv
