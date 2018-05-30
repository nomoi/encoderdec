#!/usr/bin/env python
import numpy as np
import sys
import random
from datetime import datetime
from model import Model

import arguments
import constants
import data
import util

args = arguments.parse()
train_x, train_y, test_x, test_y = data.load_pairs(args.input)
token2index, index2token = data.load_mappings(args.input)

HIDDEN_SIZE = args.hidden_size
LEARNING_RATE = args.learning_rate
EVALUATION_RATE = args.evaluation_rate
SAVE_RATE = args.save_rate

print(HIDDEN_SIZE, LEARNING_RATE, EVALUATION_RATE)
print(len(train_x), len(test_x), len(train_y), len(test_y))

m = Model(len(index2token), HIDDEN_SIZE, len(index2token))

def train_with_sgd(m, learning_rate, evaluation_rate, save_rate):
    # We keep track of the losses so we can plot them later
    losses = [( 0, 1000.0 )]
    num_examples_seen = 0

    print("Beginning training...")
    while losses[-1][1] > 0.1:
        # Optionally evaluate the loss
        if (num_examples_seen % evaluation_rate == 0):
            loss = m.calculate_loss(test_x, test_y)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%m-%d-%H-%M-%S')
            date = datetime.now().strftime('%m-%d')
            print("")
            print("[%s: Loss after %d examples = %f]" % (time, num_examples_seen, loss))
            if (num_examples_seen % save_rate == 0):
                save_path = sys.argv[-1] + "-" + date + "-" + str(loss)[:3]
                Model.save(m, save_path)

            for r in range(3):
                i = random.randint(0, len(test_x) - 1)

                model_out = m.predict_class(test_x[i])
                out = [index2token[x] for x in model_out]
                inp = [index2token[x] for x in test_x[i]]
                tar = [index2token[y] for y in test_y[i]]
                trans = "T:%s\n%s =>\n   %s" % ("".join(tar), "".join(inp), "".join(out))

                print(" * " + trans)

            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()

        # For each training example...
        for x, y in zip(train_x, train_y):
            m.SGD(x, y, learning_rate)
            num_examples_seen += 1

train_with_sgd(m, LEARNING_RATE, EVALUATION_RATE, SAVE_RATE)
