import util
from constants import *
import sys
from model import Model
import data
import glob
import os
import nltk

filestr = sys.argv[-1]

if filestr == "load":
    filestr = max(glob.iglob("models/*.npz"), key = os.path.getmtime)

m = Model.load(filestr)

def interactiveInput(model):
    inp = ""
    while inp != "q":
        inp = raw_input(">")
        if inp[0] == "#":
            vec = [data.token2index["START_TOKEN"]]+[data.token2index[x] for x in nltk.token_tokenize(inp)] + [data.token2index["END_TOKEN"]]
            out = model.vector_rep(vec)
            print(out)
        else:
            vec = [data.token2index["START_TOKEN"]]+[data.token2index[x] for x in nltk.token_tokenize(inp)] + [data.token2index["END_TOKEN"]]
            out = model.predict_class(vec)
            print(" ".join(data.index2token[x] for x in out))

interactiveInput(m)
