# Theano Encoder-Decoder
Generic encoder-decoder RNN implemented in Theano. Based on [the original 2014 paper](http://arxiv.org/abs/1406.1078) by Cho *et al*. The code is made to be readily extendable without abstracting away too much of Theano's functionality.

Modules (e.g. a GRU or LSTM) can be defined in the `initModules()` function in `model.py`:
```python
def initModules():
  return {
    "encoder": layers.GRU(inputSize, hiddenSize), 
    "decoder": layers.GRU(outputSize, hiddenSize), 
    "ffout"  : layers.FF(hiddenSize, inputSize)
  }
```
which are initialized and stored in `self.modules`. These can be accessed for use with Theano's scan function:
```python
encoder = self.modules["encoder"]

states, updates = theano.scan(encoder.step, 
  sequences = x, 
  outputs_info = T.zeros(self.hidden))
```
Modules are autodifferentiated by calling their `getUpdates()` method, given some `cost`:
```python
lr = T.scalar('learning rate')

updates = encoder.getUpdates(cost, lr) + 
          decoder.getUpdates(cost, lr) + 
          ffout.getUpdates(cost, lr)

self.SGD = theano.function([x, y, lr], updates = updates)
```
Model parameters can be saved and loaded with `Model.save()` and `Model.load()`. These will be stored in .npz format.

## Examples
Predicting the next sentence. 1000 hidden units. Outputs from training on the Project Gutenberg KJV overnight:

*input*: `so abram departed, as the lord had spoken unto him; and lot went with him  and abram was seventy and five years old when he departed out of haran.`

*output*: `and the king said unto him , i will give thee to the king of israel , and i will give thee to the king of babylon , and i will give thee to the king of babylon , and`

*input*: `the house is on fire`

*output*: `and they said unto him , i will not hear my voice , and i will be my god .`

Notice that while the network has successfully learned grammar and a biblical style, its responses are totally generic. This is probably because the sentences are too long for the encoder to really contribute much. *Try: training on clauses rather than sentences and reverse the inputs like in [this paper](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).*
