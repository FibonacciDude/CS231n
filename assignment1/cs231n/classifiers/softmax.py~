import numpy as np
from random import shuffle

z = 0

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  softmax = np.zeros(num_train)
  for i in range(num_train):
        correct_class = y[i]
        scores = X[i].dot(W)

        #now we compute softmax
        exp_sum = np.sum(np.exp(scores))
        correct_exp = np.exp(scores[correct_class])
        softmax[i] = -np.log(correct_exp / exp_sum)
        
        #The function F we just computed was: 
        # -np.log ( np.exp( scores[correct] ) * (np.sum( np.exp( scores ) ))^-1  )
        
        #dF / dsquig (F = -log(---))
        dsquig = -1 / (correct_exp / exp_sum)
        #dsquig / dscores (squig = e ^ correct * ( Sum: e ^ s )^-1)
        #When we take the derivative of the sum on the right, we just take the value from the left as our gradient from above
        #Then, with this in mind, we just multiply that gradient from above by -1 / ( sum -- ) ^ 2
        #We then mul by THAT gradient from above, and we just distribute it (from the sum function)
        #When we do that we encounter individual exp(s) so we just mul the top grad by our our value e ^ s for all s in X[i]
        #At the end we get e ^ correct * -1/exp_sum**2 * e ^ s

        dscores = np.full(scores.shape, -1 / np.square(exp_sum) * correct_exp)

        #Now, for the correct class, we just multiply by the other wire. This is 1 / exp_sum. 
        #But, don't forget the fact that we need to add the gradients along the outputs! This is since we have
        #the correct one being used for both the exp and the sum.
        #At the end we get (1 / exp_sum + whatever we had before (the gradient from the sum func) )
        dscores[correct_class] += 1 / exp_sum

        #NOW, we can start multiplying (we had to add first THEN multiply for the correct value)
        dscores *= np.exp(scores)
        #we multiply that by the top (dsquig)
        dscores *= dsquig
        
        #we then get the dW_i
        dW_i = X[i][:, np.newaxis]@dscores[np.newaxis, :]
        #don't forget that pesky dsquig!

        dW += dW_i
        
  dW /= num_train
  loss = softmax.mean()
            
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  #compute the scores
  scores = X@W
    
  #shift scores
  scores = scores - np.max(scores, axis = 1)[:, np.newaxis]
  #now get the correct vals
  correct_exp = np.exp(scores[np.arange(num_train), y])
  sum_exp = np.sum(np.exp(scores), axis = 1)
  loss = -np.log( correct_exp / sum_exp )
  
  loss = np.sum(loss) / num_train
  loss += np.sum(np.square(W)) * reg * .5


    
    
    
  #dF / dsquig (F = -log(---))
  dsquig = -1 / (correct_exp / sum_exp)

  #fill it with the values of the gradients of the sum
  dscores = np.full(scores.T.shape, -1 / np.square(sum_exp) * correct_exp).T

  
  dscores[np.arange(num_train), y] += 1 / sum_exp

  dscores = dscores * np.exp(scores)

  dscores *= dsquig[:, np.newaxis]

  dW = X.T@dscores
  dW /= num_train
   
  #regularization
  dW += reg * W
    
  #now lets calculate the gradient 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

