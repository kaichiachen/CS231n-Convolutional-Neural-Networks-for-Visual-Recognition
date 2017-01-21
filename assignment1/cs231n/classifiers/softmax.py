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
  D = W.shape[0]
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    f_i = W.T.dot(X[i])

    log_c = np.max(f_i)
    f_i -= log_c

    sum_i = 0.0
    for f_i_j in f_i:
      sum_i += np.e ** f_i_j
    loss += -1 * f_i[y[i]] + np.log(sum_i)

    # http://www.jianshu.com/p/004c99623104
    for j in range(num_classes):
      p = np.exp(f_i[j])/sum_i
      dW[:, j] += (p-(j == y[i])) * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # N * C
  scores -= np.c_[np.max(scores, axis = 1)]

  f_y = scores[np.arange(num_train), y]
  e_f_i = np.log(np.sum(np.exp(scores), axis = 1))

  loss = np.sum(-f_y + e_f_i)

  p = np.exp(scores).T / np.sum(np.exp(scores), axis = 1).T
  p = p.T
  binary_y = np.zeros((num_train, num_classes), dtype=int)
  binary_y[np.arange(num_train), y] = 1
  dW = X.T.dot(p-binary_y)

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW

