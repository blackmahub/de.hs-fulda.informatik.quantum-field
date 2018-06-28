import LoopOps as lo
import PyOps as po
import TensorflowOps as to
import RandomGenerator as rg
import PlotGraphs as pg

import tensorflow as tF ;

import time

import numpy as np
import scipy as sp
import argparse


# main function
def main():

    loop = lo.LoopOperations()
    pyops = po.PyOps()
    plot = pg.PlotGraphs()
    tens = to.TensorflowOps()

    CONST_m = 1

    parser = argparse.ArgumentParser(description="Calculate action of 4D Space Time Histories")
    parser.add_argument('-n', '--noh', help='Number of random space time histories to generate', type=int, default=5)
    parser.add_argument('-dim', '--dimensions', help='Sizes for each of the 4 dimensions', type=int, nargs=4, default=[20,20,20,20], metavar=('dim1', 'dim2', 'dim3', 'dim4'))
    parser.add_argument('-f', '--field', help='Field MIN and MAX', type=int, nargs=2, default=[-10,10], metavar=('min', 'max'))
    parser.add_argument('-dl', '-disable-loop', help='Disable loop calculation', action='store_true')
    parser.add_argument('-dpr', '-disable-python-roll', help='Disable python roll calculation', action='store_true')
    parser.add_argument('-dpc', '-disable-python-convolve', help='Disable scipy convolve calculation', action='store_true')
    parser.add_argument('-dtr', '-disable-tensorflow-roll', help='Disable tensorflow roll calculation', action='store_true')
    parser.add_argument('-dtc', '-disable-tensorflow-convolve', help='Disable tensorflow convolve calculation', action='store_true')
    parser.add_argument('-p', '-printall', help='Print each action while calculating', action='store_true')
    parser.add_argument('-sc', '-showchart', help='Plot chart in the end', action='store_true')

    args = parser.parse_args()

    noh = args.noh
    dim1 = args.dimensions[0]
    dim2 = args.dimensions[1]
    dim3 = args.dimensions[2]
    dim4 = args.dimensions[3]

    field_min = args.field[0]
    field_max = args.field[1]

    disable_loop = args.dl
    disable_pyroll = args.dpr
    disable_pyconv = args.dpc
    disable_tFroll = args.dtr
    disable_tFconv = args.dtc

    print_all = args.p

    show_chart = args.sc

    randGen = rg.RandomHistoryGenerator(noh, dim1, dim2, dim3, dim4, field_min, field_max)

    labels = []
    time_values = []

    if not disable_loop:
        labels.append('Loop')
        t0 = time.time()
        loop.calculate_action_loop(randGen.arr, CONST_m, print_all)
        tf = time.time() - t0
        time_values.append(tf)
    if not disable_pyroll:
        labels.append('NumPy Roll')
        t0 = time.time()
        pyops.calculate_action_roll(randGen.arr, CONST_m, print_all)
        tf = time.time() - t0
        time_values.append(tf)
    if not disable_tFroll:
        labels.append('Tensorflow Roll')
        S, arr, arr_size = tens.define_tf_roll(CONST_m)
        placeholder_dict = {
            arr: randGen.arr,
            arr_size: randGen.arr.size
        }
        t0 = time.time()
        tens.calculate_action_tf(S, placeholder_dict, "Roll")
        tf = time.time() - t0
        time_values.append(tf)
    if not disable_pyconv:
        labels.append('NumPy Conv')
        t0 = time.time()
        pyops.calculate_action_convolve(randGen.arr, CONST_m, print_all)
        tf = time.time() - t0
        time_values.append(tf)
    if not disable_tFconv:
        labels.append('Tensorflow Conv')
        conv_action = tens.define_tf_convolve(CONST_m, randGen.arr)
        placeholder_dict = {}
        t0 = time.time()
        tens.calculate_action_tf(conv_action, placeholder_dict, "Convolve")
        tf = time.time() - t0
        time_values.append(tf)

    if show_chart:
        plot.plot_graph(labels, time_values)

    with tF.Session() as sess:
      grdo = tF.train.GradientDescentOptimizer(learning_rate = 0.01) ;
      S,arrVar = tens.define_conv_action_alt(CONST_m, [noh, dim1,dim2, dim3, dim4]) ;
      sess.run(tF.global_variables_initializer()) ;
      S_val = sess.run(S)

      print(S_val) ;
      print (S, arrVar) ;
      updateOp = grdo.minimize(S, var_list=[arrVar]) ;
      for step in range(0,100000):
        retList = sess.run([updateOp, S]) ;
        print (retList[1]) ;




    # define_lattice()
    # STF,plTF = define_tf_graph()
    # tf_conv, placeholder_conv, placeholder_kernel_t, placeholder_kernel_x, \
    # placeholder_kernel_y, placeholder_kernel_z, placeholder_imag_zeros = define_tf_convolution()
    # volumes = []
    # loop_actions = []
    # loop_times = []
    # roll_actions = []
    # roll_times = []
    # tensor_roll_actions = []
    # tensor_roll_times = []
    # conv_actions = []
    # conv_times = []
    # tensor_conv_actions = []
    # tensor_conv_times = []
    # for count in range(1):

    #     print("Dimension: " + str(arr.ndim))
    #     print("Size: " + str(arr.size))
    #     volumes.append(arr.size)
    #     print("Shape: " + str(arr.shape))

    #     t0 = time.time()
    #     calculate_action_loop()
    #     # loop_actions.append(calculate_action_loop())
    #     tf = time.time() - t0
    #     loop_times.append(tf)
    #     # print("Loop_Time: "+ str(tf))

    #     t0 = time.time()
    #     calculate_action_roll()
    #     # roll_actions.append(calculate_action_roll())
    #     tf = time.time() - t0
    #     roll_times.append(tf)
    #     # print("Roll_Time: "+ str(tf))

    #     t0 = time.time()
    #     # tensor_roll_actions.append(calculate_action_tf(sess, STF, {plTF: arr}, "Roll"))
    #     calculate_action_tf(sess, STF, {plTF: arr}, "Roll")
    #     tf = time.time() - t0
    #     tensor_roll_times.append(tf)
    #     # print("Tensor_Roll_Time: "+ str(tf))

    #     t0 = time.time()
    #     calculate_action_convolve()
    #     # conv_actions.append(calculate_action_convolve())
    #     tf = time.time() - t0
    #     conv_times.append(tf)
    #     # print("Conv_Time: "+ str(tf))

    #     t0 = time.time()
    #     kernel_shape = arr[0].shape
    #     kernel_t = np.zeros(kernel_shape, dtype=np.float32)
    #     kernel_t[0:3, 0, 0, 0] = [1, -2, 1]
    #     kernel_x = np.zeros(kernel_shape, dtype=np.float32)
    #     kernel_x[0, 0:3, 0, 0] = [1, -2, 1]
    #     kernel_y = np.zeros(kernel_shape, dtype=np.float32)
    #     kernel_y[0, 0, 0:3, 0] = [1, -2, 1]
    #     kernel_z = np.zeros(kernel_shape, dtype=np.float32)
    #     kernel_z[0, 0, 0, 0:3] = [1, -2, 1]
    #     imag_zeros = np.zeros(kernel_shape, dtype=np.float32)
    #     placeholder_dict = {
    #         placeholder_conv: arr,
    #         placeholder_kernel_t: kernel_t,
    #         placeholder_kernel_x: kernel_x,
    #         placeholder_kernel_y: kernel_y,
    #         placeholder_kernel_z: kernel_z,
    #         placeholder_imag_zeros: imag_zeros
    #     }
    #     # tensor_conv_actions.append(calculate_action_tf(sess, tf_conv, placeholder_dict, "Convolution"))
    #     calculate_action_tf(sess, tf_conv, placeholder_dict, "Convolution")
    #     tf = time.time() - t0
    #     tensor_conv_times.append(tf)
    #     # print("Tensor_Convolution_Time: " + str(tf))
    #     print()

    #     # increase_dimension_sizes(1.5)

    # plot_graph(volumes, loop_times, roll_times, tensor_roll_times, conv_times, tensor_conv_times)
    # sess.close()


if __name__ == "__main__":
    main()
