import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    #here we are taking a simple linear combination of the rows of W (representing one dimension and its effect of the classes)
    scores = X[i].dot(W)
    #scores here has dimension C (classes)
    correct_class_score = scores[y[i]]
    
    for j in range(num_classes):
      #if we have the correct class here, skip
      if j == y[i]:
        continue
      #get the margin score (difference between the actual and the correct score + 1) 
      margin = scores[j] - correct_class_score + 1 # note delta = 1
        
      #if we have exceeded the min difference, then we add that to the loss
      if margin > 0:
        #if the margin is greater than 0, we care to update the gradient
        #meaning that for this specific class j, we want to make the gradient the values of X[i]
        dW[:, j] = dW[:, j] + X[i]
        #now, we have added the gradient x to dW, which corresponds to the scores[j] part that contributes to margin.
        #however, the correct_class_score also contributes to margin (the +1 is a constant so it doesn't affect the input)
        #So scores[j] was a function of X[i] dot W[j], but what is correct_class_score a function of?
        #You guessed it! Its a function of - X[i] dot W[y[i]] (W with correct class). Thus, we have to subtract X[i] from the
        #The gradient in column y[i]!
        dW[:, y[i]] = dW[:, y[i]] - X[i]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #take average of that
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += W * reg
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  """
  print(X.shape, W.shape, y.shape)
  dims, classes = W.shape
  amt, dims = X.shape
  dmargins = np.ones((amt, classes))
  dmargins[np.arange(amt), y] *= 0
  
  dW = X.T.dot(dmargins)
  print(dW)
  dW += reg * W
  """
  return loss, dW

cnt = 0


def svm_loss_vectorized(W, X, y, reg):
  
  #Structured SVM loss function, vectorized implementation.

  #Inputs and outputs are the same as svm_loss_naive.
  
  global cnt
  cnt += 1
  dW = np.zeros(W.shape) # initialize the gradient as zero
  N, D = X.shape
  D, C = W.shape
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  #now we have a N X C matrix
  margins = np.maximum( scores - scores[np.arange(N), y][:, np.newaxis] + 1, 0)
  #We now take the scores and subtract the correct one (np.choose(y, scores))
  margins[np.arange(N), y] = 0
  #^^^this is not favorable...
  loss = np.sum(margins) / N
  loss += .5 * reg * np.sum(np.square(W))
  #if cnt % 50 == 0:
      #print("norm is ", np.linalg.norm(scores, ord = "fro"))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #now to compute the vectorized grad
  #we need the amount of times we need to subtract from the correct column (remember that ?)
  val_correct = np.count_nonzero( (margins > 0).astype("uint8"), axis = 1 )
  dmargins = np.zeros((N, C))
  dmargins[margins > 0] = 1
  #make ones corresponding to the correct ones be -val_correct
  dmargins[np.arange(N), y] = -val_correct
  #now we just need to matmul
  dW = X.T.dot(dmargins)
  dW = dW / N + reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


"""
def svm_loss_vectorized(W, X, y, reg):
  
  #Structured SVM loss function, vectorized implementation.
  #Inputs and outputs are the same as svm_loss_naive.

  loss = 0.0
  num_train = X.shape[0]
  num_classes = W.shape[1]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1)
  margins = np.maximum(0, scores - np.tile(correct_class_scores, (1,num_classes)) + 1)
  margins[range(num_train), list(y)] = 0
  
  loss = np.sum(margins)
  loss /= num_train
  # Add regularization to the loss.
  
  loss += 0.5 * reg * np.sum(W * W)

  coeff_matrix = np.zeros((num_train, num_classes))
  coeff_matrix[margins > 0] = 1
  coeff_matrix[range(num_train), list(y)] = 0
  coeff_matrix[range(num_train), list(y)] = -np.sum(coeff_matrix, axis=1)

  dW = (X.T).dot(coeff_matrix)
  dW = dW/num_train + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
"""