import numpy as np
import time

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


""" helper layer for batch normalization"""
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    relu, relu_cache = relu_forward(bn)

    return relu, (fc_cache, bn_cache, relu_cache)


def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache

    da = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dbn, fc_cache)

    return dx, dw, db, dgamma, dbeta


def average_pooling_forward(x, pool_param):
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  
  same_size = pool_height == pool_width == stride
  tiles = H % pool_height == 0 and W % pool_width == 0
  if same_size and tiles:
    out, reshape_cache = max_pool_forward_reshape(x, pool_param)
    cache = ('reshape', reshape_cache)
  else:
    out, im2col_cache = max_pool_forward_im2col(x, pool_param)
    cache = ('im2col', im2col_cache)

  out = np.mean(x, axis=(2, 3)) # np.mean with (H, W) axis
  cache = (x)
  return out, cache


def average_pooling_backward(dout, cache):
  x = cache
  N, C, H, W = x.shape

  dx = np.zeros(x.shape)
  # naive
  for n in range(N):
    for c in range(C):
      dx[n, c, :, :] = 0.5 * dout[n, c]
  # TODO : more fast version
  return dx


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):

  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  b, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (conv_cache, bn_cache, relu_cache)
  return out, cache

def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  b, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(b)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  #out, pool_cache = average_pooling_forward(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_bn_relu_backward(dout, cache):
  conv_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, conv_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_pool_backward(dout, cache):
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, conv_cache)
  return dx, dw, db, dgamma, dbeta


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  #Affline is FC?
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(46,46,3), num_filters=(32), filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
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
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_classes = num_classes
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # first conv layer
    W1 = np.random.randn(num_filters[0], input_dim[0], filter_size, filter_size) / np.sqrt(filter_size**2*input_dim[0] / 2)
    b1 = np.zeros(num_filters[0])
    
    # second conv layer
    W2 = np.random.randn(num_filters[1], num_filters[0], filter_size, filter_size) / np.sqrt(filter_size**2*num_filters[0] / 2)
    b2 = np.zeros(num_filters[1])
    
    # third conv layer
    W3 = np.random.randn(num_filters[2], num_filters[1], filter_size, filter_size) / np.sqrt(filter_size**2*num_filters[1] / 2)
    b3 = np.zeros(num_filters[2])
    
    # fourth conv layer
    W4 = np.random.randn(num_filters[3], num_filters[2], filter_size, filter_size) / np.sqrt(filter_size**2*num_filters[2] / 2)
    b4 = np.zeros(num_filters[3])

    # moved this part in here
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    H = (input_dim[1]-filter_size+2*conv_param["pad"]) / conv_param["stride"] + 1
    H = (H - pool_param["pool_height"]) / pool_param["stride"] + 1
    W = (input_dim[2]-filter_size+2*conv_param["pad"]) / conv_param["stride"] + 1
    W = (W - pool_param["pool_width"]) / pool_param["stride"] + 1

    # fifth fc layer
    W5 = np.random.randn(int(num_filters[3]*H*W), hidden_dim) / np.sqrt(num_filters[3]*H*W / 2.0)
    b5 = np.zeros(hidden_dim)
    
    # fc-10 layer
    W6 = np.random.randn(hidden_dim, num_classes) / np.sqrt(hidden_dim / 2)
    b6 = np.zeros(num_classes)
    
    self.params["W1"] = W1; self.params["W2"] = W2; self.params["W3"] = W3
    self.params["W4"] = W4; self.params["W5"] = W5; self.params["W6"] = W6
    self.params["b1"] = b1; self.params["b2"] = b2; self.params["b3"] = b3
    self.params["b4"] = b4; self.params["b5"] = b5; self.params["b6"] = b6
    self.conv_param = conv_param; self.pool_param = pool_param

    self.params["gamma1"] = np.full(num_filters[0], 1, dtype=np.float32)
    self.params["beta1"] = np.zeros(num_filters[0])
    self.params["gamma2"] = np.full(num_filters[1], 1, dtype=np.float32)
    self.params["beta2"] = np.zeros(num_filters[1])
    self.params["gamma3"] = np.full(num_filters[2], 1, dtype=np.float32)
    self.params["beta3"] = np.zeros(num_filters[2])
    self.params["gamma4"] = np.full(num_filters[3], 1, dtype=np.float32)
    self.params["beta4"] = np.zeros(num_filters[3])
    self.params["gamma5"] = np.full(hidden_dim, 1, dtype=np.float32)
    self.params["beta5"] = np.zeros(hidden_dim)
    
    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in range(5)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    for bn_param in self.bn_params:
      bn_param[mode] = mode

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']

#    ff, _, HHf, WWf = W1.shape
    
    gamma1, beta1 = self.params["gamma1"], self.params["beta1"]
    gamma2, beta2 = self.params["gamma2"], self.params["beta2"]
    gamma3, beta3 = self.params["gamma3"], self.params["beta3"]
    gamma4, beta4 = self.params["gamma4"], self.params["beta4"]
    gamma5, beta5 = self.params["gamma5"], self.params["beta5"]
    
    conv_param, pool_param, bn_params = self.conv_param, self.pool_param, self.bn_params

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv1, conv1_cache = conv_bn_relu_forward(X, W1, b1, gamma1, beta1, conv_param, bn_params[0])
    conv2, conv2_cache = conv_bn_relu_pool_forward(conv1, W2, b2, gamma2, beta2, conv_param, pool_param, bn_params[1])
    
    conv3, conv3_cache = conv_bn_relu_forward(conv2, W3, b3, gamma3, beta3, conv_param, bn_params[2])
    conv4, conv4_cache = conv_bn_relu_forward(conv3, W4, b4, gamma4, beta4, conv_param, bn_params[3])
    
    # average pooling - dropout - fc
    #ap5, ap5_cache = average_pooling_forward(conv4)
    fc5, fc5_cache = affine_bn_relu_forward(conv4, W5, b5, gamma5, beta5, bn_params[4])
    fc6, fc6_cache  = affine_forward(fc5, W6, b6)
    
    scores = fc6
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    
    #5,6
    dx6, dw6, db6 = affine_backward(dout, fc6_cache)
    dx5, dw5, db5, dgamma5, dbeta5 = affine_bn_relu_backward(dx6, fc5_cache)
    #dx5 = average_pooling_backward(dx6, ap5_cache)
    
    #3,4
    dx4, dw4, db4, dgamma4, dbeta4 = conv_bn_relu_backward(dx5, conv4_cache)
    dx3, dw3, db3, dgamma3, dbeta3 = conv_bn_relu_backward(dx4, conv3_cache)
    #1,2
    dx2, dw2, db2, dgamma2, dbeta2 = conv_bn_relu_pool_backward(dx3, conv2_cache)
    dx1, dw1, db1, dgamma1, dbeta1 = conv_bn_relu_backward(dx2, conv1_cache)

    grads["W1"] = dw1 + self.reg*W1
    grads["W2"] = dw2 + self.reg*W2
    grads["W3"] = dw3 + self.reg*W3
    grads["W4"] = dw4 + self.reg*W4
    grads["W5"] = dw5 + self.reg*W5
    grads["W6"] = dw6 + self.reg*W6
    grads["b1"] = db1
    grads["b2"] = db2
    grads["b3"] = db3
    grads["b4"] = db4
    grads["b5"] = db5
    grads["b6"] = db6
    grads["gamma1"] = dgamma1; grads["beta1"] = dbeta1
    grads["gamma2"] = dgamma2; grads["beta2"] = dbeta2
    grads["gamma3"] = dgamma3; grads["beta3"] = dbeta3
    grads["gamma4"] = dgamma4; grads["beta4"] = dbeta4
    grads["gamma5"] = dgamma5; grads["beta5"] = dbeta5
    loss += 0.5 * self.reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2)+np.sum(W4**2)+np.sum(W5**2)+np.sum(W6**2))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    end = time.time()
    return loss, grads
