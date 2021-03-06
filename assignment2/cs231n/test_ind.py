import numpy as np
import numpy.lib.stride_tricks as tricks

"""
Get index of a certain value using fun indexing
Assumes that the input is an arange with some offset "o".
"""
"""
Pseudocode:

arr = ...
num = ...
#find num
a[num // np.prod(a.shape[1:]), (remainder from the previous operation using / ) // np.prod(a.shape[2:]) ..., // last, % last]

Python code:
def find_num(arr, num, offset = 0):
    num = np.array([num + offset]).astype(arr.dtype)
    if len(arr.shape) <= 2:
        return (num // arr.shape[1], num % arr.shape[1])
    ind = (find_num(arr, num % np.prod(arr.shape[1:]), num),) 
    return (num // np.prod(arr.shape[1:],) + ind)
"""
    
def find_num(arr, num, offset = 0):
    num = np.array([num + offset]).astype(arr.dtype)[0]
    if len(arr.shape) <= 2:
        return (num // arr.shape[1], num % arr.shape[1])
    ind = find_num(arr[0], num % np.prod(arr.shape[1:]))
    return ((num // np.prod(arr.shape[1:]),) + ind)

def get_vals(tpl):
    #peek inside
    vals = []
    for t in tpl:
        if isinstance(t, tuple):
            vals_in = get_vals(t)
            for v in vals_in:
                vals.append(v)
        else:
            vals.append(t)
            
    return vals

print(get_vals((((2, 2), 2, (((2,),),)) )))















































def get_ind(arr, num, offset = 0):
    #offset the number back!
    dtype = arr.dtype
    n = np.array([num + offset])
    num = n.astype(dtype)[0]

    if len(arr.shape) <= 2:
            return (num // arr.shape[1], num % arr.shape[1])

    ind = get_ind(arr[0], num % np.prod(arr.shape[1:]))
    return ((num // np.prod(arr.shape[1:]),) + ind)

if __name__ == "__main__":

    offset = 2
    shape = (4, 5, 5, 2)
    a = np.arange(200).reshape(*shape).astype("uint8")
    a -= 2 #offset it by 2
    ind = get_ind(a, 120, offset = 2)
    ind_2 = find_num(a, 120, offset = 2)
    
    print(ind_2)
    
    print(a[ind])
    print(a[ind_2])

