import ConvKernels as ck

import numpy as np
import scipy.ndimage as ndimg

class PyOps:

    kernels = ck.ConvKernels()

    # calculating action by roll, normalizing by volume
    def calculate_action_roll(self, arr, CONST_m, show_msg):
        noh, dim1, dim2, dim3, dim4 = arr.shape
        Sum = []
        for n in range(noh):
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
            S= total.sum() / (dim1 * dim2 * dim3 * dim4)

            Sum.append(S)
        if show_msg:
            print("S_Roll: " + str(Sum))
        return Sum

    # calculating using convolution
    def calculate_action_convolve(self, arr, CONST_m, show_msg):
        noh, dim1, dim2, dim3, dim4 = arr.shape        
        Sum = []
        for n in range(noh):
            S = 0
            convarrt = ndimg.filters.convolve(arr[n,:,:,:], self.kernels.CONST_ConvKernelt, mode = 'wrap')
            convarrx = ndimg.filters.convolve(arr[n,:,:,:], self.kernels.CONST_ConvKernelx, mode = 'wrap')
            convarry = ndimg.filters.convolve(arr[n,:,:,:], self.kernels.CONST_ConvKernely, mode = 'wrap')
            convarrz = ndimg.filters.convolve(arr[n,:,:,:], self.kernels.CONST_ConvKernelz, mode = 'wrap')
            common =  arr[n,:,:,:] * ((CONST_m**2/2) * arr[n,:,:,:])
            S = (convarrt + convarrx + convarry + convarrz + common).sum() / (dim1 * dim2 * dim3 * dim4)
            Sum.append(S)
        if show_msg:        
            print("S_Convolve: " + str(Sum))
        return Sum