import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    A convolutional network with the following architecture:
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,dtype=np.float32, use_batchnorm=False, dropout=0, seed=None):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.dtype = dtype
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout
        self.num_layers = 3
        self.params = {}

        C, H, W = input_dim
        F = num_filters
        filter_height = filter_size
        filter_width = filter_size
        stride_conv = 1
        P = (filter_size - 1) / 2

        Hc = (H + 2*P - filter_height) / stride_conv + 1
        Wc = (W + 2*P - filter_width) / stride_conv + 1

        W1 = np.random.normal(loc=0.0, scale=weight_scale, size=(F, C, filter_height, filter_width))
        b1 = np.zeros((F, ))

        width_pool = 2
        height_pool = 2
        stride_pool = 2

        Hp = Hc / height_pool
        Wp = Wc / width_pool

        W2 = np.random.normal(loc=0.0, scale=weight_scale, size=(F * Hp * Wp, hidden_dim))
        b2 = np.zeros(hidden_dim)

        W3 = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, hidden_dim))
        b3 = np.zeros(hidden_dim)

        W4 = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        b4 = np.zeros(num_classes)

        self.params.update({'W1': W1,
	                        'W2': W2,
	                        'W3': W3,
	                        'W4': W4,
	                        'b1': b1,
	                        'b2': b2,
	                        'b3': b3,
	                        'b4': b4
	                        })

        self.bn_params = []
        if self.use_batchnorm:
	      self.bn_params = [{'mode': 'train'} for i in xrange(3)]
	      gamma0 = np.ones((C, filter_height, filter_width))
	      betas0 = np.zeros(num_filters)

	      gamma1 = np.ones(hidden_dim)
	      betas1 = np.zeros(hidden_dim)

	      gamma2 = np.ones(hidden_dim)
	      betas2 = np.zeros(hidden_dim)

	      self.params.update({
	      	'gamma0': gamma0,
	      	'gamma1': gamma1,
	      	'gamma2': gamma2,
	      	'beta0': beta0,
	      	'beta1': beta1,
	      	'beta2': beta2
	      	})

        self.dropout_param = {}
        if self.use_dropout:
	      self.dropout_param = {'mode': 'train', 'p': dropout}
	      if seed is not None:
	        self.dropout_param['seed'] = seed

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):

    	X = X.astype(self.dtype)
    
        mode = 'test' if y is None else 'train'

        if self.dropout_param is not None:
	      self.dropout_param['mode'] = mode   
        if self.use_batchnorm is not None:
	      for bn_param in self.bn_params:
	        bn_param[mode] = mode

    	N = X.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma0, beta0 = self.params['gamma0'], self.params['beta0']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        filter_size = W1.shape[2]

        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        cache_stack = []

        """
        forward

        """

        if self.use_batchnorm:
        	X, cache = spatial_batchnorm_forward(X, gamma0, beta0, self.bn_param)
        	cache_stack.append(cache)
        	
        out, cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        cache_stack.append(cache)

        out,  cache= affine_relu_forward(out, W2, b2)
        cache_stack.append(cache)

        if self.use_dropout:
        	out, cache = dropout_forward(out, self.dropout_param)
	        cache_stack.append(cache)

        if self.use_batchnorm:
	        out, cache = batchnorm_forward(out, gamma1, beta1, self.bn_param)
	        cache_stack.append(cache)

        out,  cache= affine_relu_forward(out, W3, b3)
        cache_stack.append(cache)

        if self.use_dropout:
        	out, cache = dropout_forward(out, self.dropout_param)
	        cache_stack.append(cache)

        if self.use_batchnorm:
	        out, cache = batchnorm_forward(out, gamma2, beta2, self.bn_param)
	        cache_stack.append(cache)

        out,  cache = affine_relu_forward(out, W4, b4)
        cache_stack.append(cache)

        scores = out

        """
        loss function

        """

        loss, dx = softmax_loss(scores, y)
        W_sum = np.sum(W4*W4) + np.sum(W3*W3) + np.sum(W2*W2) + np.sum(W1*W1)
        loss /= N
        loss += 0.5 * reg * W_sum

        """
        backward

        """
        
        dx, dw, db = affine_relu_backward(dx, cache_stack.pop())
        dW4 = dw + reg*W4
        db4 = db

        if self.use_batchnorm:
        	dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache_stack.pop())
        	dgamma2 = dgamma
        	dbeta2 = dbeta

        if self.use_dropout:
        	dx = dropout_backward(dx, cache_stack.pop())

        dx, dw, db = affine_relu_backward(dx, cache_stack.pop())
        dW3 = dw + reg*W3
        db3 = db

        if self.use_batchnorm:
        	dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache_stack.pop())
        	dgamma1 = dgamma
        	dbeta1 = dbeta

        if self.use_dropout:
        	dx = dropout_backward(dx, cache_stack.pop())

        dx, dw, db = affine_relu_backward(dx, cache_stack.pop())
        dW2 = dw + reg*W2
        db2 = db

        dx, dw, db = conv_relu_pool_backward(dx, cache_stack.pop())
        dW1 = dw + reg*W1
        db1 = db

        dx, dgamma, dbeta = spatial_batchnorm_backward(dx, cache_stack.pop())
        dgamma0 = dgamma
        dbeta0 = dbeta

        grads = {}
        grads.update({
        	'W4': dW4,
        	'W3': dW3,
        	'W2': dW2,
        	'W1': dW1,
        	'b4': db4,
        	'b3': db3,
        	'b2': db2,
        	'b1': db1,
        	'gamma2': dgamma2,
        	'gamma1': dgamma1,
        	'gamma0': dgamma0,
        	'beta2': dbeta2,
        	'beta1': dbeta1,
        	'beta0': dbeta0,
        	})

        return loss, grads

pass
