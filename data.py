import sys
import operator
import util
import numpy as np

def shuffle(a, b):
    start_time = util.now()
    print("[Shuffling...]", )
    sys.stdout.flush()

    assert len(a) == len(b), "Length of arrays is not equal." 
    combined = np.asarray([[x,y] for x,y in zip(a, b)])
    np.random.shuffle(combined)

    print("[Took %d milliseconds]" % (util.now() - start_time))
    return combined[:, 0], combined[:, 1]

def divide(a, b, ratio):
    start_time = util.now()
    print("[Dividing len %d data by %f ratio...]" % (len(a), ratio))
    sys.stdout.flush()

    assert len(a) == len(b), "Length of arrays is not equal." 
    index = round(len(a) * ratio)
    a_1, a_2 = a[:index], a[index:]
    b_1, b_2 = b[:index], b[index:]

    print("[Division of %d:%d]" % (len(a_1), len(a_2)),)
    print("[Took %d milliseconds]" % (util.now() - start_time))
    return a_1, a_2, b_1, b_2

def npz_load(inp, name):
    data = util.loadFile(inp)

    start_time = util.now()
    print("[Loading %s:%s...]" % (inp, name),)
    sys.stdout.flush()

    loaded = data[name]
    print("[Took %d milliseconds]" % (util.now() - start_time))
    return loaded

def load_pairs(inp):
    
    pairs = npz_load(inp, "pairs")
    data_x, data_y = shuffle(pairs[:, 0], pairs[:, 1])

    return divide(data_x, data_y, 0.995)

def load_mappings(inp):
    
    index2token = npz_load(inp, "index2token")

    token2index = {}
    for i,e in enumerate(index2token):
        token2index[e] = i

    return token2index, index2token
