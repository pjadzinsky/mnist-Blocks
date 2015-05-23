# I'll build a convolutional neural network to work on mnist

# imports
import pdb
import theano.tensor as T

from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalActivation, Convolutional, MaxPooling
from blocks.bricks import Rectifier, Tanh, Softmax, MLP, Identity
from blocks.initialization import IsotropicGaussian, Constant

filter_size = 3

# define the model
x = T.tensor4('features')
# starting wtih a gray image of 28x28 pixels (N,1,28,28)
# 1st implement a convolution followed by reLU
cl = ConvolutionalActivation(
        Rectifier().apply,
        name='conv_1',
        filter_size=(filter_size, filter_size),
        num_filters=3,
        num_channels=1,
        image_size=(28,28),
        border_mode='full',
        weights_init=IsotropicGaussian(.1),
        biases_init=Constant(0),
        )

overshoot = (filter_size-1)/2

cl_out = cl.apply(x)#[:,:, overshoot:-overshoot, overshoot:-overshoot]

"""
# then an affine layer
affine = MLP(
        [Rectifier().apply], 
        [28*28, 10],
        weights_init=IsotropicGaussian(.1),
        biases_init=Constant(0),
        )

affine_out = MLP.apply(cl_out)
"""
