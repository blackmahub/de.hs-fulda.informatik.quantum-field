# import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tF
sess = tF.InteractiveSession()

# initial dimension sizes
CONST_N = 10
CONST_Dimension1 = 20
CONST_Dimension2 = 20
CONST_Dimension3 = 20
CONST_Dimension4 = 20
CONST_Volume = CONST_Dimension1 * CONST_Dimension2 * CONST_Dimension3 * CONST_Dimension4

# max and min for each point
CONST_FieldMin = -10
CONST_FieldMax = 10

# constant mass
CONST_m = 1

# convolution kernel
CONST_ConvKernelt = np.zeros([3,1,1,1])
CONST_ConvKernelt [:,0,0,0] = [1,-2,1]
CONST_ConvKernelx = np.zeros([1,3,1,1])
CONST_ConvKernelx [0,:,0,0] = [1,-2,1]
CONST_ConvKernely = np.zeros([1,1,3,1])
CONST_ConvKernely [0,0,:,0] = [1,-2,1]
CONST_ConvKernelz = np.zeros([1,1,1,3])
CONST_ConvKernelz [0,0,0,:] = [1,-2,1]

# initial empty lattice. This will be dynamically generated later
arr = np.empty([CONST_N, CONST_Dimension1, CONST_Dimension2, CONST_Dimension3, CONST_Dimension4])


# random lattice generation
def define_lattice():
    global arr
    for n in range(CONST_N): 
        arr[n,:,:,:] = np.random.uniform(CONST_FieldMin, CONST_FieldMax, size = (CONST_Dimension1, CONST_Dimension2, CONST_Dimension3, CONST_Dimension4))    


# increasing volume of lattice for comparison
def increase_dimension_sizes(factor):
    global CONST_Dimension1 
    CONST_Dimension1 = int(CONST_Dimension1 * factor)
    global CONST_Dimension2 
    CONST_Dimension2 = int(CONST_Dimension2 * factor)
    global CONST_Dimension3
    CONST_Dimension3 = int(CONST_Dimension3 * factor)
    global CONST_Dimension4
    CONST_Dimension4 = int(CONST_Dimension4 * factor)
    define_lattice()  


# calculating action by nested for loops, normalizing by volume
def calculate_action_loop():
    Sum = []
    for n in range(CONST_N):
        S = 0
        for t in range(CONST_Dimension1):
            for x in range(CONST_Dimension2):
                for y in range(CONST_Dimension3):
                    for z in range(CONST_Dimension4):
                        tFactor = (arr[n, t+1, x, y, z] if t < CONST_Dimension1 - 1 else arr[n, 0,x,y,z]) + (arr[n, t-1, x, y, z] if t > 0 else arr[n, CONST_Dimension1 - 1, x, y, z]) #- (2 * arr[t,x,y,z])
                        xFactor = (arr[n, t, x+1, y, z] if x < CONST_Dimension2 - 1 else arr[n, t,0,y,z]) + (arr[n, t, x-1, y, z] if x > 0 else arr[n, CONST_Dimension2 - 1, y, z]) #- (2 * arr[t,x,y,z])
                        yFactor = (arr[n, t, x, y+1, z] if y < CONST_Dimension3 - 1 else arr[n, t,x,0,z]) + (arr[n, t, x, y-1, z] if y > 0 else arr[n, t, x, CONST_Dimension3 - 1, z]) #- (2 * arr[t,x,y,z])
                        zFactor = (arr[n, t, x, y, z+1] if z < CONST_Dimension4 - 1 else arr[n, t,x,y,0]) + (arr[n, t, x, y, z-1] if z > 0 else arr[n, t, x, y, CONST_Dimension4 - 1]) #- (2 * arr[t,x,y,z])
                        # tFactor = (arr[t+1, x, y, z] if t < CONST_Dimension1 - 1 else 0) + (arr[t-1, x, y, z] if t > 0 else 0) - (2 * arr[t,x,y,z])
                        # xFactor = (arr[t, x+1, y, z] if x < CONST_Dimension2 - 1 else 0) + (arr[t, x-1, y, z] if x > 0 else 0) - (2 * arr[t,x,y,z])
                        # yFactor = (arr[t, x, y+1, z] if y < CONST_Dimension3 - 1 else 0) + (arr[t, x, y-1, z] if y > 0 else 0) - (2 * arr[t,x,y,z])
                        # zFactor = (arr[t, x, y, z+1] if z < CONST_Dimension4 - 1 else 0) + (arr[t, x, y, z-1] if z > 0 else 0) - (2 * arr[t,x,y,z])
                        S += tFactor + xFactor + yFactor + zFactor - (8 * arr[n, t,x,y,z]) + (CONST_m**2/2)*(arr[n, t,x,y,z]**2)
        S /= CONST_Volume
        Sum.append(S)
    # print("S_Loop: "+ str(S))
    return Sum


# calculating action by roll, normalizing by volume
def calculate_action_roll():
    Sum = []
    for n in range(CONST_N):
        S = 0
        tLeft = np.roll(arr[n,:,:,:], -1, axis = 1)
        tRight = np.roll(arr[n,:,:,:], 1, axis = 1)

        xLeft = np.roll(arr[n,:,:,:], -1, axis = 0)
        xRight = np.roll(arr[n,:,:,:], 1, axis = 0)

        yLeft = np.roll(arr[n,:,:,:], -1, axis = 2)
        yRight = np.roll(arr[n,:,:,:], 1, axis = 2)

        zLeft = np.roll(arr[n,:,:,:], -1, axis = 3)
        zRight = np.roll(arr[n,:,:,:], 1, axis = 3)

        common =  arr[n,:,:,:] * (-8 + (CONST_m**2/2) * arr[n,:,:,:])
        
        total = tLeft + tRight + xLeft + xRight + yLeft + yRight + zLeft + zRight + common    
        S= total.sum() / CONST_Volume

        Sum.append(S)
    # print("S_Roll: " + str(S))
    return Sum


# calculating using tensorflow
def define_tf_roll(arr):
    tleft = tF.manip.roll(arr, shift=-1, axis=1)
    tright = tF.manip.roll(arr, shift=1, axis=1)

    xleft = tF.manip.roll(arr, shift=-1, axis=0)
    xright = tF.manip.roll(arr, shift=1, axis=0)

    yleft = tF.manip.roll(arr, shift=-1, axis=2)
    yright = tF.manip.roll(arr, shift=1, axis=2)

    zleft = tF.manip.roll(arr, shift=-1, axis=3)
    zright = tF.manip.roll(arr, shift=1, axis=3)

    common = arr * (-8 + (CONST_m ** 2 / 2) * arr)

    total = tleft + tright + xleft + xright + yleft + yright + zleft + zright + common

    return tF.reduce_sum(total) / CONST_Volume


def define_tf_graph():
    arr = tF.placeholder(dtype=tF.float64)
    S = tF.map_fn(lambda a: define_tf_roll(a), arr[:, ])

    return S, arr


def calculate_action_tf(sess, S, placeholder_dict, action_name):
    tf_action = sess.run([S], feed_dict=placeholder_dict)[0]
    print("S_TF_%s: %s" % (action_name, str(tf_action)))
    return tf_action


# calculating using convolution
def calculate_action_convolve():
    Sum = []
    for n in range(CONST_N):
        S = 0
        convarrt = sp.ndimage.filters.convolve(arr[n,:,:,:], CONST_ConvKernelt, mode = 'wrap')
        convarrx = sp.ndimage.filters.convolve(arr[n,:,:,:], CONST_ConvKernelx, mode = 'wrap')
        convarry = sp.ndimage.filters.convolve(arr[n,:,:,:], CONST_ConvKernely, mode = 'wrap')
        convarrz = sp.ndimage.filters.convolve(arr[n,:,:,:], CONST_ConvKernelz, mode = 'wrap')
        common =  arr[n,:,:,:] * ((CONST_m**2/2) * arr[n,:,:,:])
        S = (convarrt + convarrx + convarry + convarrz + common).sum() / CONST_Volume
        Sum.append(S)
        # print("S_Convolve: " + str(S))
    return Sum

def define_tf_convolution_for_1_lattice(arr, imag_zeros, kernel_t_fft, kernel_x_fft, kernel_y_fft, kernel_z_fft):
    arr_fft = tF.fft(tF.complex(arr, imag_zeros))

    conv_axis_0 = tF.real(tF.ifft(arr_fft * kernel_t_fft))
    conv_axis_1 = tF.real(tF.ifft(arr_fft * kernel_x_fft))
    conv_axis_2 = tF.real(tF.ifft(arr_fft * kernel_y_fft))
    conv_axis_3 = tF.real(tF.ifft(arr_fft * kernel_z_fft))

    common = arr * ((CONST_m ** 2 / 2) * arr)

    return tF.reduce_sum(conv_axis_0 + conv_axis_1 + conv_axis_2 + conv_axis_3 + common) / CONST_Volume


def define_tf_convolution():
    placeholder_arr = tF.placeholder(tF.float32)
    placeholder_kernel_t = tF.placeholder(tF.float32)
    placeholder_kernel_x = tF.placeholder(tF.float32)
    placeholder_kernel_y = tF.placeholder(tF.float32)
    placeholder_kernel_z = tF.placeholder(tF.float32)
    placeholder_imag_zeros = tF.placeholder(tF.float32)

    kernel_t_fft = tF.fft(tF.complex(placeholder_kernel_t, placeholder_imag_zeros))
    kernel_x_fft = tF.fft(tF.complex(placeholder_kernel_x, placeholder_imag_zeros))
    kernel_y_fft = tF.fft(tF.complex(placeholder_kernel_y, placeholder_imag_zeros))
    kernel_z_fft = tF.fft(tF.complex(placeholder_kernel_z, placeholder_imag_zeros))

    conv_action = tF.map_fn(lambda a: define_tf_convolution_for_1_lattice(a, placeholder_imag_zeros, \
                                                            kernel_t_fft, kernel_x_fft, kernel_y_fft, kernel_z_fft), placeholder_arr[:, ])

    return conv_action, placeholder_arr, placeholder_kernel_t, placeholder_kernel_x, \
           placeholder_kernel_y, placeholder_kernel_z, placeholder_imag_zeros


# plot comparison graph
def plot_graph(volumes, loop_times, roll_times, tensor_roll_times, conv_times, tensor_conv_times):
    n_groups = len(volumes)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.9

    rects1 = ax.bar(index, loop_times, bar_width,
                    alpha=opacity, color='b',
                    label='Nested For Loops')

    rects2 = ax.bar(index + bar_width, roll_times, bar_width,
                    alpha=opacity, color='r',
                    label='NumPy Roll')

    rects3 = ax.bar(index + 2 * bar_width, tensor_roll_times, bar_width,
                    alpha=opacity, color='g',
                    label='Tensorflow Roll')

    rects4 = ax.bar(index + 3 * bar_width, conv_times, bar_width,
                    alpha=opacity, color='c',
                    label='SciPy Convolve')

    rects5 = ax.bar(index + 4 * bar_width, tensor_conv_times, bar_width,
                    alpha=opacity, color='y',
                    label='Tensorflow Convolve')

    ax.set_xlabel('Volume')
    ax.set_ylabel('Time (in s)')
    ax.set_title('Times by Volumes')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(volumes)
    ax.legend()
    fig.tight_layout()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)
    autolabel(ax, rects5)

    plt.show()


# Attach a text label above each bar displaying its height
def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,
                str(round(height, 3)),
                ha='center', va='bottom')


# main function
def main():
    define_lattice()
    STF,plTF = define_tf_graph()
    tf_conv, placeholder_conv, placeholder_kernel_t, placeholder_kernel_x, \
    placeholder_kernel_y, placeholder_kernel_z, placeholder_imag_zeros = define_tf_convolution()
    volumes = []
    loop_actions = []
    loop_times = []
    roll_actions = []
    roll_times = []
    tensor_roll_actions = []
    tensor_roll_times = []
    conv_actions = []
    conv_times = []
    tensor_conv_actions = []
    tensor_conv_times = []
    for count in range(3):

        print("Dimension: " + str(arr.ndim))
        print("Size: " + str(arr.size))
        volumes.append(arr.size)
        print("Shape: " + str(arr.shape))

        t0 = time.time()
        calculate_action_loop()
        # loop_actions.append(calculate_action_loop())
        tf = time.time() - t0
        loop_times.append(tf)
        # print("Loop_Time: "+ str(tf))

        t0 = time.time()
        calculate_action_roll()
        # roll_actions.append(calculate_action_roll())
        tf = time.time() - t0
        roll_times.append(tf)
        # print("Roll_Time: "+ str(tf))

        t0 = time.time()
        # tensor_roll_actions.append(calculate_action_tf(sess, STF, {plTF: arr}, "Roll"))
        calculate_action_tf(sess, STF, {plTF: arr}, "Roll")
        tf = time.time() - t0
        tensor_roll_times.append(tf)
        # print("Tensor_Roll_Time: "+ str(tf))

        t0 = time.time()
        calculate_action_convolve()
        # conv_actions.append(calculate_action_convolve())
        tf = time.time() - t0
        conv_times.append(tf)
        # print("Conv_Time: "+ str(tf))

        t0 = time.time()
        kernel_shape = arr[0].shape
        kernel_t = np.zeros(kernel_shape, dtype=np.float32)
        kernel_t[0:3, 0, 0, 0] = [1, -2, 1]
        kernel_x = np.zeros(kernel_shape, dtype=np.float32)
        kernel_x[0, 0:3, 0, 0] = [1, -2, 1]
        kernel_y = np.zeros(kernel_shape, dtype=np.float32)
        kernel_y[0, 0, 0:3, 0] = [1, -2, 1]
        kernel_z = np.zeros(kernel_shape, dtype=np.float32)
        kernel_z[0, 0, 0, 0:3] = [1, -2, 1]
        imag_zeros = np.zeros(kernel_shape, dtype=np.float32)
        placeholder_dict = {
            placeholder_conv: arr,
            placeholder_kernel_t: kernel_t,
            placeholder_kernel_x: kernel_x,
            placeholder_kernel_y: kernel_y,
            placeholder_kernel_z: kernel_z,
            placeholder_imag_zeros: imag_zeros
        }
        # tensor_conv_actions.append(calculate_action_tf(sess, tf_conv, placeholder_dict, "Convolution"))
        calculate_action_tf(sess, tf_conv, placeholder_dict, "Convolution")
        tf = time.time() - t0
        tensor_conv_times.append(tf)
        # print("Tensor_Convolution_Time: " + str(tf))
        print()

        # increase_dimension_sizes(1.5)

    plot_graph(volumes, loop_times, roll_times, tensor_roll_times, conv_times, tensor_conv_times)
    sess.close()


if __name__ == "__main__":
    main()
