import numpy as np

class LoopOperations:
    
    # calculating action by nested for loops, normalizing by volume
    def calculate_action_loop(self, arr, CONST_m, show_msg):
        noh, dim1, dim2, dim3, dim4 = arr.shape
        Sum = []
        for n in range(noh):
            S = 0
            for t in range(dim1):
                for x in range(dim2):
                    for y in range(dim3):
                        for z in range(dim4):
                            tFactor = (arr[n, t+1, x, y, z] if t < dim1 - 1 else arr[n, 0,x,y,z]) + (arr[n, t-1, x, y, z] if t > 0 else arr[n, dim1 - 1, x, y, z]) #- (2 * arr[t,x,y,z])
                            xFactor = (arr[n, t, x+1, y, z] if x < dim2 - 1 else arr[n, t,0,y,z]) + (arr[n, t, x-1, y, z] if x > 0 else arr[n, t, dim2 - 1, y, z]) #- (2 * arr[t,x,y,z])
                            yFactor = (arr[n, t, x, y+1, z] if y < dim3 - 1 else arr[n, t,x,0,z]) + (arr[n, t, x, y-1, z] if y > 0 else arr[n, t, x, dim3 - 1, z]) #- (2 * arr[t,x,y,z])
                            zFactor = (arr[n, t, x, y, z+1] if z < dim4 - 1 else arr[n, t,x,y,0]) + (arr[n, t, x, y, z-1] if z > 0 else arr[n, t, x, y, dim4 - 1]) #- (2 * arr[t,x,y,z])
                            S += tFactor + xFactor + yFactor + zFactor - (8 * arr[n, t,x,y,z]) + (CONST_m**2/2)*(arr[n, t,x,y,z]**2)
            S /= (dim1 * dim2 * dim3 * dim4)
            Sum.append(S)
        avg = sum(Sum)/float(len(Sum))
        if show_msg:
            print("S_Loop: "+ str(avg))
        return avg