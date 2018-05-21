# import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
# import scipy as sp

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


# plot comparison graph
def plot_graph(volumes, loop_times, roll_times):

    ax1 = plt.subplot(311)
    ax1.set_title("Loop Times vs Volumes")
    ax1.set_xlabel("Volumes")
    ax1.set_ylabel("Times in Second")
    plt.plot(volumes, loop_times)
    plt.setp(ax1.get_xticklabels())

    ax2 = plt.subplot(312, sharex=ax1)
    ax2.set_title("Roll Times vs Volumes")
    ax2.set_xlabel("Volumes")
    ax2.set_ylabel("Times in Second")
    plt.plot(volumes, roll_times)

    plt.xlim(0, np.amax(volumes))
    plt.show()

# main function
def main():
    volumes = []
    loop_actions = []
    loop_times = []
    roll_actions = []
    roll_times = []
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
        print()

        increase_dimension_sizes(1.5)

    plot_graph(volumes, loop_times, roll_times)
        
            
if __name__ == "__main__":
    main()
