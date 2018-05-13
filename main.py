import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

CONST_Dimension1 = 2
CONST_Dimension2 = 5
CONST_Dimension3 = 5
CONST_Dimension4 = 5

CONST_FieldMin = -0.001
CONST_FieldMax = 0.001

temp_LagrangianDensity = 5
#arr = np.random.uniform(CONST_FieldMin, CONST_FieldMax, size = (CONST_Dimension1, CONST_Dimension2, CONST_Dimension3, CONST_Dimension4))
map
# a = np.multiply(a, temp_LagrangianDensity)
# print(a)
# sum = a.sum()
# print(sum)
# print(arr)
m = 1
# S = 0
# for t in range(CONST_Dimension1):
#     for x in range(CONST_Dimension2):
#         for y in range(CONST_Dimension3):
#             for z in range(CONST_Dimension4):
#                 tFactor = (arr[t+1, x, y, z] if t < CONST_Dimension1 - 1 else 0) + (arr[t-1, x, y, z] if t > 0 else 0) - (2 * arr[t,x,y,z])
#                 xFactor = (arr[t, x+1, y, z] if x < CONST_Dimension2 - 1 else 0) + (arr[t, x-1, y, z] if x > 0 else 0) - (2 * arr[t,x,y,z])
#                 yFactor = (arr[t, x, y+1, z] if y < CONST_Dimension3 - 1 else 0) + (arr[t, x, y-1, z] if y > 0 else 0) - (2 * arr[t,x,y,z])
#                 zFactor = (arr[t, x, y, z+1] if z < CONST_Dimension4 - 1 else 0) + (arr[t, x, y, z-1] if z > 0 else 0) - (2 * arr[t,x,y,z])
#                 S += tFactor + xFactor + yFactor + zFactor + (m**2/2)*(arr[t,x,y,z]**2)
#                 print("arr["+str(t)+","+str(x)+","+str(y)+","+str(z)+"]: " + str(arr[t,x,y,z]))
#                 print("S: "+ str(S))

def calculate():
    arr = np.random.uniform(CONST_FieldMin, CONST_FieldMax, size = (CONST_Dimension1, CONST_Dimension2, CONST_Dimension3, CONST_Dimension4))
    print("Dimension: " + str(arr.ndim))
    print("Size: " + str(arr.size))
    print("Shape: " + str(arr.shape))
    #print("SUM: " + str(arr.sum()))
    
    #t0 = time.time()
    S2 = 0
    for t in range(CONST_Dimension1):
        for x in range(CONST_Dimension2):
            for y in range(CONST_Dimension3):
                for z in range(CONST_Dimension4):
                    tFactor = (arr[t+1, x, y, z] if t < CONST_Dimension1 - 1 else 0) + (arr[t-1, x, y, z] if t > 0 else 0) #- (2 * arr[t,x,y,z])
                    xFactor = (arr[t, x+1, y, z] if x < CONST_Dimension2 - 1 else 0) + (arr[t, x-1, y, z] if x > 0 else 0) #- (2 * arr[t,x,y,z])
                    yFactor = (arr[t, x, y+1, z] if y < CONST_Dimension3 - 1 else 0) + (arr[t, x, y-1, z] if y > 0 else 0) #- (2 * arr[t,x,y,z])
                    zFactor = (arr[t, x, y, z+1] if z < CONST_Dimension4 - 1 else 0) + (arr[t, x, y, z-1] if z > 0 else 0) #- (2 * arr[t,x,y,z])
                    S2 += tFactor + xFactor + yFactor + zFactor - (8 * arr[t,x,y,z]) + (m**2/2)*(arr[t,x,y,z]**2)
                    #print("arr["+str(t)+","+str(x)+","+str(y)+","+str(z)+"]: " + str(arr[t,x,y,z]))

    S2 /= (CONST_Dimension1 * CONST_Dimension2 * CONST_Dimension3 * CONST_Dimension4)
    print("S2: "+ str(S2))
    #print("Time: "+ str(time.time() - t0))

    # t0 = time.time()
    # S2 = 0
    # for (t, x, y, z) in itertools.product(range(CONST_Dimension1), range(CONST_Dimension2), range(CONST_Dimension3), range(CONST_Dimension4)):
    #     tFactor = (arr[t+1, x, y, z] if t < CONST_Dimension1 - 1 else 0) + (arr[t-1, x, y, z] if t > 0 else 0) #- (2 * arr[t,x,y,z])
    #     xFactor = (arr[t, x+1, y, z] if x < CONST_Dimension2 - 1 else 0) + (arr[t, x-1, y, z] if x > 0 else 0) #- (2 * arr[t,x,y,z])
    #     yFactor = (arr[t, x, y+1, z] if y < CONST_Dimension3 - 1 else 0) + (arr[t, x, y-1, z] if y > 0 else 0) #- (2 * arr[t,x,y,z])
    #     zFactor = (arr[t, x, y, z+1] if z < CONST_Dimension4 - 1 else 0) + (arr[t, x, y, z-1] if z > 0 else 0) #- (2 * arr[t,x,y,z])
    #     S2 += tFactor + xFactor + yFactor + zFactor - (8 * arr[t,x,y,z]) + (m**2/2)*(arr[t,x,y,z]**2)

    # print("itertools S2: "+ str(S2))
    # print("itertools Time: "+ str(time.time() - t0))

    return S2
    

actions = []
for count in range(10):
    actions.append(calculate())

#print(actions)

fig = plt.figure(figsize=(50,50))
ax = plt.axes()

ax.plot(range(len(actions)), actions)
plt.scatter(range(len(actions)), actions)
plt.show()
