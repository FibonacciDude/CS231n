import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) // stride + 1
  out_width = (W + 2 * padding - field_width) // stride + 1

  i0 = np.repeat(np.arange(field_height), field_width)
  print(i0)
  i0 = np.tile(i0, C)
  print(i0)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  print(i1)
  j0 = np.tile(np.arange(field_width), field_height * C)
  print(j0)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  print(j1)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
  print(k)

  return (k, i, j)


if __name__ == "__main__":
    
    x = np.random.randint(5, size=(2, 4, 4, 4))
    k, i, j = get_im2col_indices(x.shape, 2, 2)
    print(k)
    print(i)
    print(j)
