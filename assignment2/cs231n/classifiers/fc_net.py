import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    self.layer_shapes = [input_dim, hidden_dim, num_classes]

    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    for i, l in enumerate(self.layer_shapes[:-1]):
        W_n = np.random.normal(scale = weight_scale, size = (l, self.layer_shapes[i + 1]))
        self.params['W' + str(i + 1)] = W_n
        b_n = np.zeros(self.layer_shapes[i + 1])
        self.params['b' + str(i + 1)] = b_n
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = X
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    layers = 2
    caches = []
    reg_sum = 0
    for l in range(layers):
        w = self.params['W' + str(l + 1)]
        b = self.params['b' + str(l + 1)]
        reg_sum += np.sum(np.square(w))
        scores, cache = affine_relu_forward(scores, w, b)
        caches.append(cache)
        

    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
       
    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #backward pass
    
    loss, dscores = softmax_loss(scores, y)
    loss += reg_sum * self.reg * .5
    #compute the gradients
    #Our process is simple, we first take dscores and compute the affine-relu backward pass
    #After this, we just need to take that dout and calculate the dout for the next layer down
    dout = dscores
    for l in range(layers - 1, -1, -1):
        dx, dw, db = affine_relu_backward(dout, caches[l])
        w = self.params['W' + str(l + 1)]
        reg = self.reg * w
        dw = dw + reg
        grads['W' + str(l + 1)] = dw
        grads['b' + str(l + 1)] = db
        dout = dx
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

#helps debug
def get_shapes(c):
    sh = lambda g : g.shape
    if not isinstance(c, tuple) and not isinstance(c, np.ndarray):
        return type(c)
    
    if isinstance(c[0], tuple) and len(c) > 1:
        return list(map(lambda g : get_shapes(g), c))
    return list(map(sh, list(c)))

#define helper functions

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    out, aff_cache = affine_forward(x, w, b)
    out, norm_cache = batchnorm_forward(out, gamma, beta, bn_param)
    out, rel_cache = relu_forward(out)
    cache = (aff_cache, norm_cache, rel_cache)
    return out, cache
    
def affine_batchnorm_relu_backward(dout, cache):
    aff_cache, norm_cache, rel_cache = cache
    drel = relu_backward(dout, rel_cache)
    dnorm, dg, dbeta = batchnorm_backward_alt(drel, norm_cache)
    daff, dw, db = affine_backward(dnorm, aff_cache)
    return daff, dw, db, dg, dbeta

def layer_dropout_forward(inps, func, param):
    scores, cache_f = func(*inps)
    scores, cache_d = dropout_forward(scores, param)
    return scores, cache_f + cache_d

def layer_dropout_backward(dout, func, cache):

    cache_f, cache_d = cache[:-2], cache[-2:]
    dout = dropout_backward(dout, cache_d)
    grads = func(dout, cache_f)
    return grads

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None, init=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    -init: The initialization for the networks. Possible = {'xavier', 'relu' (for relu activation funcs)}
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.init = init

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    #initializations
    hidden_dims = [input_dim] + hidden_dims + [num_classes]
    for i, l in enumerate(hidden_dims[:-1]):
        W_n = np.random.normal(scale = weight_scale, size = (l, hidden_dims[i + 1]))
        if self.init == 'xavier':
            W_n /= np.sqrt(l)
        if self.init == 'relu':
            W_n *= np.sqrt(2 / l)
            
        self.params['W' + str(i + 1)] = W_n
        b_n = np.zeros(hidden_dims[i + 1])
        self.params['b' + str(i + 1)] = b_n
        if self.use_batchnorm and i < len(hidden_dims) - 2:
            gamma_n = np.ones((hidden_dims[i + 1],), dtype = self.dtype)
            self.params['gamma' + str(i + 1)] = gamma_n
            beta_n = np.zeros((hidden_dims[i + 1],), dtype = self.dtype)
            self.params['beta' + str(i + 1)] = beta_n

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = X
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    caches = []
    reg_sum = 0
    for l in range(self.num_layers):
        w = self.params['W' + str(l + 1)]
        b = self.params['b' + str(l + 1)]
        reg_sum += np.sum(np.square(w))
        if l < self.num_layers - 1:
            
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(l + 1)]
                beta = self.params['beta' + str(l + 1)]
                scores, cache = affine_batchnorm_relu_forward(scores, w, b, gamma, beta, self.bn_params[l])
            else:
                scores, cache = affine_relu_forward(scores, w, b)
                
            if self.use_dropout:
                scores, cache_d = dropout_forward(scores, self.dropout_param)
                caches.append(cache + cache_d)
                
            else:
                caches.append(cache)
                
        else:
            scores, cache = affine_relu_forward(scores, w, b)
        
            caches.append(cache)
                
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += reg_sum * self.reg * .5
    #compute the gradients
    #Our process is simple, we first take dscores and compute the affine-relu backward pass
    #After this, we just need to take dout and substitute it for the gradient of x (the dout for the second layer in)
    dout = dscores
    for l in range(self.num_layers - 1, -1, -1):
        if l < self.num_layers - 1:
            if self.use_dropout:
                dout = dropout_backward(dout, caches[l][-2:])
                cache_f = caches[l][:-2]
            else:
                cache_f = caches[l]

            if self.use_batchnorm:
                dx, dw, db, dg, dbeta = affine_batchnorm_relu_backward(dout, cache_f)
                grads['gamma' + str(l + 1)] = dg
                grads['beta' + str(l + 1)] = dbeta
               
            else:
                dx, dw, db = affine_relu_backward(dout, cache_f)
                
        else:
            dx, dw, db = affine_relu_backward(dout, caches[l])
            
        w = self.params['W' + str(l + 1)]
        reg = self.reg * w
        dw += reg
        
        grads['W' + str(l + 1)] = dw
        grads['b' + str(l + 1)] = db
        dout = dx
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
