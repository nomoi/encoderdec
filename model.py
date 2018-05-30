import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
from theano.compile.debugmode import DebugMode
import time
import operator
import layers

import data
import constants
import arguments
import util

args = arguments.parse()

token2index, index2token = data.load_mappings(args.input)

class Model:
    
    def __init__(self, inputSize, hiddenSize, outputSize, modules = None):

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        if modules == None:
            encoder = layers.GRU(inputSize, hiddenSize, "encoder-GRU")
            decoder = layers.GRU(outputSize, hiddenSize, "decoder-GRU")
            ffout = layers.FF(hiddenSize, inputSize, "ffout")
            self.modules = { "encoder" : encoder, "decoder" : decoder, "ffout" : ffout }
        else:
            self.modules = modules 
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        start_time = util.now()
        print("[Compiling model graph...]")

        print("[Loading modules]")
        encoder, decoder, ffout = self.modules["encoder"], self.modules["decoder"], self.modules["ffout"]

        x_e = T.ivector('x_e')
        y = T.ivector("y")

        y_d = y[1:]
        y_x = y[:-1]

        print("[Encoder loop]")
        s_e, updates = theano.scan(
            encoder.step,
            sequences = x_e,
            outputs_info = T.zeros(self.hiddenSize))

        vector_rep = s_e[-1]
        def decoder_out(y_x, prev):
            s = decoder.step(y_x, prev)
            o_t = ffout.step(s)
            return o_t, s

        [o_d, s_d], updates = theano.scan(
            decoder_out,
            sequences=y_x,
            outputs_info=[None, dict(initial=vector_rep)])

        def decoder_out_free(y_x, prev):
            s = decoder.step(y_x, prev)
            o_t = ffout.step(s)

            return (T.argmax(o_t), s), theano.scan_module.until(T.eq(T.argmax(o_t), T.constant(token2index["END_TOKEN"])))

        print("[Decoder loop]")
        [o_d_free, s_d_free], updates = theano.scan(
            decoder_out_free,
            outputs_info=[dict(initial=T.cast(T.constant(token2index["START_TOKEN"]), "int64")), dict(initial=vector_rep)],
            n_steps = 40)

        cost = T.sum(T.nnet.categorical_crossentropy(o_d, y_d))

        print("[Prediction]")
        self.predict_class = theano.function([x_e], o_d_free)
        self.vector_rep = theano.function([x_e], vector_rep)
        self.ce_error = theano.function([x_e, y], cost)

        lr = T.scalar('learning rate')

        updates = ffout.getUpdates(cost, lr) + decoder.getUpdates(cost, lr) + encoder.getUpdates(cost, lr)

        print("[Stochastic Gradient Descent]")
        self.SGD = theano.function(
            [x_e, y, lr],
            updates = updates)

        print("[Took %d milliseconds]" % (util.now() - start_time))

    def calculate_total_loss(self, test_x, test_y):
        return np.sum([self.ce_error(x_e, y) for x_e, y in zip(test_x, test_y)])

    def calculate_loss(self, test_x, test_y):
        num_tokens = np.sum([len(y) for y in test_y])
        return self.calculate_total_loss(test_x, test_y)/float(num_tokens)

    @staticmethod
    def save(model, filestr):
        print("[Saving model to %s...]" % filestr)

        modules = {name: model.modules[name].getParameters() for name in model.modules}
        sizes = [model.inputSize, model.hiddenSize, model.outputSize]

        np.savez("models/" + filestr, sizes = sizes, **modules)

    @staticmethod
    def load(filestr):
        print("[Loading model from %s...]" % filestr)
        f = np.load(filestr)
        module_values = {name: f[name] for name in f.files}
        sizes = module_values.pop("sizes")

        #reconstructing modules

        modules = {}
        for name in module_values:
            moduleType = module_values[name][-1]
            if moduleType == "GRU":
                modules[name] = layers.GRU(*module_values[name])
            elif moduleType == "FF":
                modules[name] = layers.FF(*module_values[name])

        m = Model(*sizes, modules = modules) 

        return m
