import numpy as np
import tensorflow as tf

import POC.TensorflowOps as tOps

class ActionCal:

    def calculate_action(self, CONST_m, field_min, field_max, shape):
        tensorOp = tOps.TensorflowOps(CONST_m, field_min, field_max, shape)
        interesting = []
        with tensorOp.sess as sess:
            tensorOp.define_conv_kernels()
            grdo = tf.train.GradientDescentOptimizer(learning_rate = 100)
            S = tensorOp.define_conv_action_graph()
            sess.run(tf.global_variables_initializer())
            S_val = sess.run(S)
            print(S_val)
            interesting.append(tensorOp.arrVar.eval(sess))
            updateOp = grdo.minimize(S, var_list=[tensorOp.arrVar])
            for step in range(0,8000):
                # print(step)
                retList = sess.run([updateOp, S])
                print (retList[1]) 
            interesting.append(tensorOp.arrVar.eval(sess))
        return interesting

        