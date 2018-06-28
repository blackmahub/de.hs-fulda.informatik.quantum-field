import numpy as np
import tensorflow as tf

class ActionCal:

    def __define_conv_kernels(self, CONST_m, arrShape, arrSize):

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
        kernel = tf.cast(tf.sparse_tensor_to_dense(tf.SparseTensor(indices, values, arrShape)), tf.float32, name="kernel")

        global kernel_fft
        kernel_fft = tf.Variable(initial_value=tf.fft(tf.complex(kernel, imag_zeros)), name="var_kernel_fft")


    def __define_conv_action_graph(self, CONST_m, field_min, field_max, arrShape):
        arrVar = tf.Variable(initial_value=tf.random_uniform(arrShape, field_min, field_max, dtype=tf.float32))

        arr_fft = tf.fft(tf.complex(arrVar, imag_zeros), name="arr_fft")

        conv_axis = tf.real(tf.ifft(tf.multiply(arr_fft, kernel_fft)), name="op_convolve_axis")
        common = tf.multiply(tf.multiply(arrVar, arrVar), tf.divide(tf.multiply(mass_tf, mass_tf), 2), name="op_common")
        conv_action = tf.divide(tf.reduce_sum(tf.add(conv_axis, common)), total_volume, name="graph_conv_action")
        
        return conv_action, arrVar


    def calculate_action(self, CONST_m, field_min, field_max, shape):
        interesting = []
        with tf.Session() as sess:
            self.__define_conv_kernels(CONST_m, shape, np.prod(shape))
            grdo = tf.train.GradientDescentOptimizer(learning_rate = 100)
            S,arrVar = self.__define_conv_action_graph(CONST_m, field_min, field_max, shape)
            sess.run(tf.global_variables_initializer())
            S_val = sess.run(S)
            print(S_val)
            interesting.append(arrVar.eval(sess))
            updateOp = grdo.minimize(S, var_list=[arrVar])
            for step in range(0,1000000):
                retList = sess.run([updateOp, S])
                if retList[1] == 0:
                    interesting.append(arrVar.eval(sess))
                print (retList[1]) 
            interesting.append(arrVar.eval(sess))
        return interesting

        