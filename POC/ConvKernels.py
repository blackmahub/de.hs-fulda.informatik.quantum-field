import numpy as np

class ConvKernels:

    CONST_ConvKernelt = np.zeros([3,1,1,1])
    CONST_ConvKernelt [:,0,0,0] = [1,-2,1]

    CONST_ConvKernelx = np.zeros([1,3,1,1])
    CONST_ConvKernelx [0,:,0,0] = [1,-2,1]

    CONST_ConvKernely = np.zeros([1,1,3,1])
    CONST_ConvKernely [0,0,:,0] = [1,-2,1]
    
    CONST_ConvKernelz = np.zeros([1,1,1,3])
    CONST_ConvKernelz [0,0,0,:] = [1,-2,1]

    imag_zeros = np.zeros(CONST_ConvKernelt.shape, dtype=np.float32)