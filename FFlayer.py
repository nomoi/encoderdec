import theano
import theano.tensor as T
import numpy as np

class FFlayer:

    def __init__(self, inputSize, outputSize, name = "FF Layer"):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.name = name

        V = np.random.uniform(-np.sqrt(1.0 / inputSize), np.sqrt(1.0 / inputSize), (outputSize, inputSize))
        c = np.zeros(outputSize)
        self.V = theano.shared(name = name + '.V', value = V.astype(theano.config.floatX))
        self.c = theano.shared(name = name + '.c', value = c.astype(theano.config.floatX))

    def step(self, x):
        return T.nnet.softmax(self.V.dot(x) + self.c)[0]


    def getUpdates(self, cost, learning_rate):
        dV = T.grad(cost, self.V)
        dc = T.grad(cost, self.c)

        return [(self.V, self.V - learning_rate * dV),
                (self.c, self.c - learning_rate * dc)]

    def getParameters(self):
        return self.name, [self.V.get_value(), 
                           self.c.get_value()]

    def loadParameters(self, params):
        self.E.set_value(params[0])
        self.b.set_value(params[1])
