# imports
from blocks.bricks import MLP, Tanh, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
import theano.tensor as T

# data
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

# Construct the model
mlp = MLP(activations=[Tanh(), Softmax()], dims=[784, 100, 10], weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
mlp.initialize()

# Calculate the loss function
x = T.matrix('features')
y = T.lmatrix('targets')
y_hat = mlp.apply(x)
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
error_rate = MisclassificationRate().apply(y.flatten(), y_hat)

# load training data using Fuel
mnist_train = MNIST("train")
train_stream = Flatten(DataStream.default_stream(
                        dataset=mnist_train,
                        iteration_scheme=SequentialScheme(mnist_train.num_examples, 128)),
                       )

# load testing data
mnist_test = MNIST("test")
test_stream = Flatten(DataStream.default_stream(
                        dataset=mnist_test,
                        iteration_scheme=SequentialScheme(mnist_test.num_examples, 1024)),
                      )

# train the model
from blocks.model import Model
main_loop = MainLoop(
    model=Model(cost),
    data_stream=train_stream,
    algorithm=GradientDescent(
    cost=cost, params=ComputationGraph(cost).parameters,
    step_rule=Scale(learning_rate=0.1)),
    extensions=[FinishAfter(after_n_epochs=5),
        DataStreamMonitoring(
            variables=[cost, error_rate],
            data_stream=test_stream,
            prefix="test"),
        Printing()])

main_loop.run()
