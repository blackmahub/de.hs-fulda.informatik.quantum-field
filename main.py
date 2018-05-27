# import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
# import scipy as sp
import tensorflow as tF
sess = tF.InteractiveSession()

# initial dimension sizes
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
CONST_ConvKernel = [1, -2, 1]

# initial empty lattice. This will be dynamically generated later
arr = np.empty([CONST_Dimension1, CONST_Dimension2, CONST_Dimension3, CONST_Dimension4])

# random lattice generation
def define_lattice():
    global arr 
    arr = np.random.uniform(CONST_FieldMin, CONST_FieldMax, size = (CONST_Dimension1, CONST_Dimension2, CONST_Dimension3, CONST_Dimension4))    


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
    S = 0
    for t in range(CONST_Dimension1):
        for x in range(CONST_Dimension2):
            for y in range(CONST_Dimension3):
                for z in range(CONST_Dimension4):
                    tFactor = (arr[t+1, x, y, z] if t < CONST_Dimension1 - 1 else arr[0,x,y,z]) + (arr[t-1, x, y, z] if t > 0 else arr[CONST_Dimension1 - 1, x, y, z]) #- (2 * arr[t,x,y,z])
                    xFactor = (arr[t, x+1, y, z] if x < CONST_Dimension2 - 1 else arr[t,0,y,z]) + (arr[t, x-1, y, z] if x > 0 else arr[t, CONST_Dimension2 - 1, y, z]) #- (2 * arr[t,x,y,z])
                    yFactor = (arr[t, x, y+1, z] if y < CONST_Dimension3 - 1 else arr[t,x,0,z]) + (arr[t, x, y-1, z] if y > 0 else arr[t, x, CONST_Dimension3 - 1, z]) #- (2 * arr[t,x,y,z])
                    zFactor = (arr[t, x, y, z+1] if z < CONST_Dimension4 - 1 else arr[t,x,y,0]) + (arr[t, x, y, z-1] if z > 0 else arr[t, x, y, CONST_Dimension4 - 1]) #- (2 * arr[t,x,y,z])
                    # tFactor = (arr[t+1, x, y, z] if t < CONST_Dimension1 - 1 else 0) + (arr[t-1, x, y, z] if t > 0 else 0) - (2 * arr[t,x,y,z])
                    # xFactor = (arr[t, x+1, y, z] if x < CONST_Dimension2 - 1 else 0) + (arr[t, x-1, y, z] if x > 0 else 0) - (2 * arr[t,x,y,z])
                    # yFactor = (arr[t, x, y+1, z] if y < CONST_Dimension3 - 1 else 0) + (arr[t, x, y-1, z] if y > 0 else 0) - (2 * arr[t,x,y,z])
                    # zFactor = (arr[t, x, y, z+1] if z < CONST_Dimension4 - 1 else 0) + (arr[t, x, y, z-1] if z > 0 else 0) - (2 * arr[t,x,y,z])
                    S += tFactor + xFactor + yFactor + zFactor - (8 * arr[t,x,y,z]) + (CONST_m**2/2)*(arr[t,x,y,z]**2)
    
    S /= CONST_Volume
    print("S_Loop: "+ str(S))
    return S


# calculating action by roll, normalizing by volume
def calculate_action_roll():

    tLeft = np.roll(arr, -1, axis = 1)
    tRight = np.roll(arr, 1, axis = 1)

    xLeft = np.roll(arr, -1, axis = 0)
    xRight = np.roll(arr, 1, axis = 0)

    yLeft = np.roll(arr, -1, axis = 2)
    yRight = np.roll(arr, 1, axis = 2)

    zLeft = np.roll(arr, -1, axis = 3)
    zRight = np.roll(arr, 1, axis = 3)

    common =  arr * (-8 + (CONST_m**2/2) * arr)
    
    total = tLeft + tRight + xLeft + xRight + yLeft + yRight + zLeft + zRight + common    
    S= total.sum() / CONST_Volume

    print("S_Roll: " + str(S))
    return S

# calculating using tensorflow
def calculate_action_tf():

    tleft = tF.manip.roll(arr,shift= -1,axis= 1)
    tright = tF.manip.roll(arr,shift= 1,axis= 1)

    xleft = tF.manip.roll(arr,shift= -1,axis= 0)
    xright = tF.manip.roll(arr,shift= 1,axis= 0)

    yleft = tF.manip.roll(arr,shift=-1,axis= 2)
    yright = tF.manip.roll(arr,shift= 1,axis= 2)

    zleft = tF.manip.roll(arr,shift= -1,axis= 3)
    zright = tF.manip.roll(arr,shift= 1,axis= 3)

    common =  arr * (-8 + (CONST_m**2/2) * arr)

    total = tleft + tright + xleft + xright + yleft + yright + zleft + zright + common
    S= tF.reduce_sum(total) / CONST_Volume
    S= tF.Print(S,[S], message="T_Roll: ")
    S.eval()
    return S

# plot comparison graph
def plot_graph(volumes, loop_times, roll_times, tensor_roll_times):
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
                    label='Numpy Roll')

    rects3 = ax.bar(index + 2 * bar_width, tensor_roll_times, bar_width,
                    alpha=opacity, color='g',
                    label='Tensorflow Roll')

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
    volumes = []
    loop_actions = []
    loop_times = []
    roll_actions = []
    roll_times = []
    tensor_roll_actions = []
    tensor_roll_times = []
    for count in range(3):

        print("Dimension: " + str(arr.ndim))
        print("Size: " + str(arr.size))
        volumes.append(arr.size)
        print("Shape: " + str(arr.shape))

        t0 = time.time()
        loop_actions.append(calculate_action_loop())
        tf = time.time() - t0
        loop_times.append(tf)
        print("Loop_Time: "+ str(tf))

        t0 = time.time()
        roll_actions.append(calculate_action_roll())
        tf = time.time() - t0
        roll_times.append(tf)
        print("Roll_Time: "+ str(tf))

        t0 = time.time()        
        tensor_roll_actions.append(calculate_action_tf())
        tf = time.time() - t0
        tensor_roll_times.append(tf)    
        print("Tensor_Time: "+ str(tf))
        print()

        increase_dimension_sizes(1.5)

    plot_graph(volumes, loop_times, roll_times, tensor_roll_times)
        
            
if __name__ == "__main__":
    main()
