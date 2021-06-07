import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_shaped = x.reshape(x.shape[0], -1)
  out = x_shaped@w + b[np.newaxis, :]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  db = np.sum(dout, axis = 0)
  dx = dout@w.T
  dw = x.reshape(dx.shape).T@dout
  dx = dx.reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = (x > 0) * x
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
 
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout * (x > 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var: Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    s_mean = x.mean(axis = 0, keepdims = True)
    s_var = x.var(axis = 0, keepdims = True)
    
    x_centered = x - s_mean
    x_std = np.sqrt(s_var + eps)
    out = x_centered / x_std
    xhat = out
    out = gamma * out + beta
    
    cache = (xhat, 1 / x_std, gamma)
    
    running_mean = momentum * running_mean + s_mean * (1 - momentum)
    running_var = momentum * running_var + s_var * (1 - momentum)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    out = (x - running_mean) / np.sqrt(running_var + eps) * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
    
  """
  x, xhat, out, s_mean, s_var, std_eps, x_centered, gamma, beta, eps = cache
  n, d = x.shape
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################\

  dgamma = np.sum(xhat* dout, axis = 0)
  dbeta = np.sum(dout, axis = 0)
    
  #x_centered = (x - s_mean)
  var_eps = std_eps ** 2
  #std_eps = np.sqrt(var_eps)


  dnorm = dout * gamma
  #dmain = 1 / std_eps * dnorm + np.sum( dnorm *  x_centered * - 1 / var_eps, axis = 0) * 1 / (std_eps) / n * x_centered
  dmain = gamma * (dout / std_eps - np.sum(dout * x_centered / var_eps, axis = 0) * x_centered / (n * std_eps))
  dx = dmain - np.sum(dmain, axis = 0) / n

  #who cares if we have the alternate version??
  
  """
  #dmain = gamma / std_eps * (dout - np.sum(dout / (x_centered + eps), axis = 0) * x_centered)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
    
  xhat, istd, gamma = cache
  n = xhat.shape[0]

  dbeta = np.sum(dout, axis = 0)
  dgamma = np.sum(dout*xhat, axis = 0)

  #now look at THIS bad boy!
  dx = (dout*n - dgamma*xhat -dbeta) * (gamma*istd/n) #istd = 1 / std
  #haha! Mine is ~46 characters!
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    #since we drop with probability p, the ones left will be (1 - p)
    mask = (np.random.rand(*x.shape) > p) / (1 - p)
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
    
  return dx

def _convolve(x, f, stride, hp, wp):
    """
    Take in an input x, with 3 dimensions (C, H, W) and a filter f with 
    3 dimensions (C, HH, WW).
    
    We also take in an empty activation map to fill.
      Input:
          - x: Input data of shape (C, H, W)
          - f: Filter weights of shape (C, HH, WW)        
          - stride: Stride
          -hp: H'
          -wp: W' 
          (both defined below)
      Returns:
            The convolution of x with filter f. Shape (H', W')
            H' = 1 + (H - HH) / stride
            W' = 1 + (W - WW) / stride
            
    No checking is done to see if the filter fits or is valid.
    """
    _, H, W = x.shape
    _, HH, WW = f.shape
    act_map = np.empty((hp, wp))

    for i in range(0, H - HH + 1, stride):
        for j in range(0, W - WW + 1, stride):
            visual_field = x[:, i:i+HH, j:j+WW]
            act_map[i // stride, j // stride] = np.dot(visual_field.flatten(), f.flatten())
     
    return act_map

def conv_forward_naive(x, w, b, conv_param, dims = {}):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """

  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  out = None
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
    
    
  padding = conv_param['pad']
  stride = conv_param['stride']
  cache = (x, w, b, conv_param)

  #get the shapes of the input dimension to initialize
  H_p = (H + 2 * padding - HH) // stride + 1
  W_p = (W + 2 * padding - WW) // stride + 1
    
  #checkfor correctness of input
  assert((H + 2 * padding - HH) % stride == 0)
  assert((WW + 2 * padding - WW) % stride == 0)  

  #first pad the input data on its height and width
  pad = ((0,), (0,), (padding,), (padding,))
  out = np.pad(x, pad)


  act_map = np.empty((N, F, H_p, W_p))
 
  #move around the datapoints N and the filters F and convolve like a madman
  for n in range(N):
        for f in range(F):
            act_map[n, f] = _convolve(out[n, ...], w[f, ...], stride, H_p, W_p) + b[f]
  
  out = act_map
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
   
  x, w, b, conv_param = cache


  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  _, _, H_out, W_out = dout.shape
    
  padding = conv_param['pad']
  stride = conv_param['stride']

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  #dx = np.zeros(x.shape)    
  db = np.sum(dout, axis = (0, 2, 3))

  #add padding
  pad = ((0,), (0,), (padding,), (padding,))
  x = np.pad(x, pad)

  
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  for i in range(H_out):
      for j in range(W_out):
          for f in range(F):
              for n in range(N):

                  #facts: dim(visual_field) = (N, Depth, HH, WW) (since you take the field of view)
                  #dim(act_map | dout) = (N, F, H', W')
                  #also recall dim(x) = (N, C, H, W) and dim(w) = (F, C, HH, WW)


                  weight = dout[n, f, i, j]
                  #dx[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[f] * weight                   
                
                  visual_field = x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] 
                  dw[f] += visual_field * weight
                  
                
                    
                  #fix: sum over filters in parallel                    
                  #do: overlap region
                    
                  #overlap region with respect to the non-padded input field
                  x_ = np.array([i*stride,i*stride+HH]) - padding
                  x_ = np.clip(x_, a_min=0, a_max=H) #we don't need the a_max

                  y_ = np.array([j*stride,j*stride+WW]) - padding
                  y_ = np.clip(y_, a_min=0, a_max=W) #we don't need the a_max
                
                  x_l = x_[1] - x_[0]
                  y_l = y_[1] - y_[0]
                
                
                  clipped_filter = w[f]
                    
                  if x_[0] == 0:
                      clipped_filter = clipped_filter[:, -x_l:, :]
                  else:
                      clipped_filter = clipped_filter[:, :x_l, :]
  
                  if y_[0] == 0:
                      clipped_filter = clipped_filter[:, :, -y_l:]
                  else:
                      clipped_filter = clipped_filter[:, :, :y_l]

                  
                  dx[n, :, x_[0]:x_[1], y_[0]:y_[1]] += clipped_filter * weight
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']

  N, C, H, W = x.shape

  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  H_out = (H - HH) // stride + 1
  W_out = (W - WW) // stride + 1
  
  out = np.empty((N, C, H_out, W_out))
    
  for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                out[n, :, i, j] = np.max(x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW], axis = (1, 2))
                
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']

  N, C, H, W = x.shape
  dx = np.zeros_like(x)
  
  _,_, H_out, W_out = dout.shape
  
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  #notice that we took the max of the depth dimension of each individual region
  for n in range(N):
      for i in range(H_out):
          for j in range(W_out):
              region = x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]

              max_vals = np.max(region, axis= (1, 2))
              for w, val in enumerate(max_vals):
                  max_inds = np.where(region == val)[1:]
                  dx[n, w, i*stride + max_inds[0], j*stride + max_inds[1]] += dout[n, w, i, j]
              
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def get_reshaped_strides(reshaped, size):
  strides = np.zeros(len(reshaped), dtype=np.uint8)
  for i, dim in enumerate(reshaped[:-2]):
      strides[i] = np.prod(reshaped[i+1:]) * size
  strides[-2] = reshaped[-1] * size
  strides[-1] = size
  return tuple(strides)

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines. Use np.transpose.            #
  #############################################################################
    
  (N, C, H, W) = x.shape

  x_view = x.transpose((0, 2, 3, 1))
  x_view = x_view.reshape(-1, C)
  out, cache = batchnorm_forward(x_view, gamma[None, :], beta[None, :], bn_param)
  #reshape to (N, H, W, C) then permute axis
  out = out.reshape(((N, H, W, C))).transpose((0, 3, 1, 2))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  (N, C, H, W) = dout.shape
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
  dx = dx.reshape((N, H, W, C)).transpose(0, 3, 1, 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss_(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

def softmax_loss(scores, y):
  
  N = len(y)
  #shift scores
  scores = np.exp(scores - np.max(scores, axis = 1, keepdims = True)) # subtract max to prevent overflow with exp
    
  sum_exp = np.sum( scores , axis = 1, keepdims = True)
  scores /= sum_exp
  #now get the correct vals
  loss = -np.sum(np.log(scores[np.arange(N), y])) / N

  #dscores
  #dsquig is -sum / correct
  #fill it with the values of the gradients of the sum
  dscores = scores.copy()
  #we have already "timesd" the dsquig
  dscores[np.arange(N), y] -= 1
  dscores /= N
    
  return loss, dscores