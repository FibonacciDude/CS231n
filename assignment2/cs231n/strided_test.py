import numpy as np
import numpy.lib.stride_tricks as lib

"""
These are some methods in order to take a sort of "convolution field of view" using stride tricks.
"""

def get_height(x_height, f_height, stride):
    assert x_height > f_height
    assert (x_height - f_height) % stride == 0

    return (x_height - f_height) // stride + 1

def split_arr(x, f_height, stride):
    dsize = x.itemsize
    x_shape = x.shape #oo, you can do the same thing with a structured or "recarray" (record array)
    out_height = get_height(x.shape[0], f_height, stride)
    shape_new = (out_height, f_height, x.shape[1])
    strides_new = (np.prod(x_shape[1:]) * dsize, x_shape[1] * dsize, dsize)
    return lib.as_strided(x, shape = shape_new, strides = strides_new)

if __name__ == "__main__":
    x = np.arange(20).reshape(4, 5)
    stride = 1 
    f_height = 3
    x_split = split_arr(x, f_height, stride)
    print(x)
    print(x_split)

