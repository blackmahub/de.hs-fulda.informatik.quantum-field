import numpy as np

# max and min for each point
CONST_FieldMin = -10
CONST_FieldMax = 10

class RandomHistoryGenerator:
    # initial empty lattice. This will be dynamically generated later
    arr = np.empty([1, 1, 1, 1, 1], dtype=np.float32)

    def __init__(self, noh, dim1, dim2, dim3, dim4, fieldmin, fieldmax):
        self.arr = np.empty([noh, dim1, dim2, dim3, dim4], dtype=np.float32)
        for n in range(noh): 
            self.arr[n,:,:,:] = np.random.uniform(fieldmin, fieldmax, size = (dim1, dim2, dim3, dim4))
