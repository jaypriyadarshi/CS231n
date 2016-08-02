import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores = np.dot(X[i,:],W)

    #avoid numerical instability
    logC = np.max(scores)
    scores -= logC

    sum_i = 0.0
    for ij in scores:
      sum_i += np.exp(ij)
    loss += -scores[y[i]] + np.log(sum_i)

    for j in range(num_classes):
      p = np.exp(scores[j])/sum_i
      dW[:,j] += (p - (j == y[i])) * X[i,:]

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_classes = W.shape[1]
  num_train =X.shape[0]

  scores = np.dot(X,W)
  scores -= np.max(scores)

  correct_class = scores[range(num_train),y]
  loss = -np.mean(np.log(np.exp(correct_class)/np.sum(np.exp(scores),axis=1)))

  p = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(num_train,1)
  ind = np.zeros(p.shape)
  ind[range(num_train),y] = 1
  dW = np.dot(X.T, (p-ind))
  dW /= num_train

  loss += 0.5 * reg * np.sum(W *W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

