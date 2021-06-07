import numpy as np

H, W = 5, 5
img = np.random.rand(H, W)
pad = 1
stride = 1
padding = ((pad,), (pad,))
img = np.pad(img, padding)
HH, WW = 3, 3 
f = np.arange(HH * WW).reshape(HH, WW)

H_out = (H - HH + 2 * pad) // stride + 1
W_out = (W - WW + 2 * pad) // stride + 1

print(H_out, W_out)

print(f)

print('img shape {}'.format((H, W)))
print('padded img shape {}'.format(img.shape))
print('filter shape {}'.format(f.shape))

for i in range(H_out):
    for j in range(W_out):
        #location is i * stride, j * stride
        #take out the border
        x_ = np.array([i*stride,i*stride + HH]) - pad
        x_ = np.clip(x_, a_min=0, a_max=H) #we don't need the a_max

        y_ = np.array([j*stride,j*stride + WW]) - pad
        y_ = np.clip(y_, a_min=0, a_max=W) #we don't need the a_max
        

        clipped_img = img[x_[0]:x_[1], y_[0]:y_[1]]


        x_l = x_[1] - x_[0]
        y_l = y_[1] - y_[0]

        clipped_filter = f
        if x_[0] == 0:
            clipped_filter = clipped_filter[-x_l:, :]
        else:
            clipped_filter = clipped_filter[:x_l, :]

        if y_[0] == 0:
            clipped_filter = clipped_filter[:, -y_l:]
        else:
            clipped_filter = clipped_filter[:, :y_l]

        print(i, j)
        print('coords {}, {}'.format(x_, y_))
        print(f)
        print(clipped_filter)
        print(clipped_filter.shape)
        print(clipped_img.shape)

        if clipped_filter.shape != clipped_img.shape:
            raise ValueError("Incorrect")

        
