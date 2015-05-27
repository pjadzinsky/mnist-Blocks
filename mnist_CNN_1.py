# I'll build a convolutional neural network to work on mnist

##################################################################
# Building the model
##################################################################
import pdb
import theano.tensor as T
import numpy as np

from blocks.bricks import Rectifier, Softmax, MLP
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence, MaxPooling
from blocks.bricks.conv import ConvolutionalActivation, Flattener
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.roles import WEIGHT, FILTER, INPUT
from blocks.graph import ComputationGraph, apply_dropout

batch_size = 128
filter_size = 3
num_filters = 4
initial_weight_std = .01

x = T.tensor4('features')
y = T.lmatrix('targets')

# Convolutional Layers
conv_layers = [
        ConvolutionalLayer(Rectifier().apply, (3,3), 16, (2,2), name='l1'),
        ConvolutionalLayer(Rectifier().apply, (3,3), 32, (2,2), name='l2')]

convnet = ConvolutionalSequence(
        conv_layers, num_channels=1, image_size=(28,28),
        weights_init=IsotropicGaussian(0.1),
        biases_init=Constant(0)
        )

convnet.initialize()

output_dim = np.prod(convnet.get_dim('output'))
print(output_dim)

# Fully connected layers
features = Flattener().apply(convnet.apply(x))

mlp = MLP(
        activations=[Rectifier(), None],
        dims=[output_dim, 100, 10],
        weights_init=IsotropicGaussian(0.01),
        biases_init=Constant(0)
        )
mlp.initialize()

y_hat = mlp.apply(features)


# numerically stable softmax
cost = Softmax().categorical_cross_entropy(y.flatten(), y_hat)
cost.name = 'nll'
#cost = MisclassificationRate().apply(y, y_hat)
#cost.name = 'error_rate'

cg = ComputationGraph(cost)

#pdb.set_trace()
weights = VariableFilter(roles=[FILTER, WEIGHT])(cg.variables)
l2_regularization = 0.005 * sum((W**2).sum() for W in weights)

cost_l2 = cost + l2_regularization
cost.name = 'cost_with_regularization'

# Print sizes to check
print("Representation sizes:")
for layer in convnet.layers:
    print(layer.get_dim('input_'))

##################################################################
# Training
##################################################################
from blocks.dump import load_parameter_values
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.extensions import SimpleExtension, FinishAfter, Printing
from blocks.algorithms import GradientDescent, Scale, Momentum
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint, LoadFromDump
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

rng = np.random.RandomState(1)

mnist = MNIST('train')
training_stream = DataStream.default_stream(
    mnist,
    iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=batch_size))

algorithm = GradientDescent(cost=cost, params=cg.parameters,
        step_rule=Scale(learning_rate=0.1))

mnist_test = MNIST("test")
validation_stream = DataStream.default_stream(
    mnist_test,
    iteration_scheme=SequentialScheme(
        mnist_test.num_examples, batch_size=1024)
    )

model = Model(cost_l2)
algorithm = GradientDescent(
        cost=cost_l2,
        params=model.parameters,
        step_rule=Momentum(
            learning_rate=1e-2,
            momentum=0.9)
        )

main_loop = MainLoop(
        model = model,
        data_stream = training_stream,
        algorithm = algorithm,
        extensions = [
            FinishAfter(after_n_epochs=10),
            TrainingDataMonitoring(
                [cost],
                prefix='train',
                after_epoch=True),
            DataStreamMonitoring(
                [cost],
                validation_stream,
                prefix='valid'),
            Checkpoint('mnist.pkl', after_epoch=True),
            #EarlyStoppingDump('/Users/jadz/Documents/Micelaneous/Coursework/Blocks/mnist-blocks/', 'valid_error_rate'),
            Printing()
            ]
        )

main_loop.run()
