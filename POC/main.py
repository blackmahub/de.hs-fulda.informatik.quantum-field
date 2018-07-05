import LoopOps as lo
import PyOps as po
import TensorflowOps as to
import PlotGraphs as pg

import time

import numpy as np
import argparse


# main function
def main():

    loop = lo.LoopOperations()
    pyops = po.PyOps()
    plot = pg.PlotGraphs()

    CONST_m = 1

    parser = argparse.ArgumentParser(description="Calculate action of 4D Space Time Histories")
    parser.add_argument('-n', '--noh', help='Number of random space time histories to generate', type=int, default=50)
    parser.add_argument('-dim', '--dimensions', help='Sizes for each of the 4 dimensions', type=int, nargs=4, default=[20,20,20,20], metavar=('dim1', 'dim2', 'dim3', 'dim4'))
    parser.add_argument('-f', '--field', help='Field MIN and MAX', type=float, nargs=2, default=[-10,10], metavar=('min', 'max'))
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

    labels = []
    time_values = []

    action_values = []

    tens = to.TensorflowOps(CONST_m, field_min, field_max, [noh, dim1, dim2, dim3, dim4])
    tens.initialize_tensorflow_variables()
    arr = tens.arrVar.eval(tens.sess)

    if not disable_loop:
        labels.append('Loop')
        t0 = time.time()
        action_values.append(loop.calculate_action_loop(arr, CONST_m, print_all))
        _tf = time.time() - t0
        time_values.append(_tf)
    if not disable_pyroll:
        labels.append('NumPy Roll')
        t0 = time.time()
        action_values.append(pyops.calculate_action_roll(arr, CONST_m, print_all))
        _tf = time.time() - t0
        time_values.append(_tf)
    if not disable_tFroll:
        labels.append('Tensorflow Roll')
        S = tens.define_tf_roll_graph()
        t0 = time.time()
        action_values.append(tens.calculate_action_tf_roll(S, print_all))
        _tf = time.time() - t0
        time_values.append(_tf)
    if not disable_pyconv:
        labels.append('NumPy Conv')
        t0 = time.time()
        action_values.append(pyops.calculate_action_convolve(arr, CONST_m, print_all))
        _tf = time.time() - t0
        time_values.append(_tf)
    if not disable_tFconv:
        labels.append('Tensorflow Conv')
        tens.define_conv_kernels()
        conv_action = tens.define_conv_action_graph()
        # put the global var init BEFORE time measurement, and just once!
        tens.initialize_tensorflow_variables()
        t0 = time.time()
        action_values.append(tens.calculate_action_tf_convolve(conv_action, print_all))
        _tf = time.time() - t0
        time_values.append(_tf)

    if print_all:
        print("SD from mean: "+str(np.std(action_values)))

    if show_chart:
        plot.plot_graph(labels, time_values)

if __name__ == "__main__":
    main()
