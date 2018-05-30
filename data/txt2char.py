import argparse
import random
import nltk
import operator
import itertools
import numpy
import re

parser = argparse.ArgumentParser(description="script for converting plaintext files into training char data for seq2seq")
parser.add_argument("input", help="file to be read")
parser.add_argument("-p", "--preview-size", type=int, help="number of lines to preview for quality check", default=0)
parser.add_argument("-e", "--equal-pairing", action='store_true', help="make both pairs the same")
parser.add_argument("-i", "--include", type=str, help="characters to include in addition to tokenbet", default="")
args = parser.parse_args()

index2token = list(" abcdefghijklmnopqrstuvwxyz',.!?;:" + args.include)
index2token.extend(["START_TOKEN", "END_TOKEN"])
VOCAB_SIZE = len(index2token)
print(index2token)

token2index = {}
for i,e in enumerate(index2token):
    token2index[e] = i

FILENAME = args.input

tokenizer = nltk.tokenize.TweetTokenizer()

def clean():
    with open(FILENAME) as f:

        formatted = []
        lines = [list(x.strip()) for x in f.readlines()]

        for line in lines:
            new_line = []
            for char in line:
                if char in token2index:
                    new_line.append(char)
            formatted.append(new_line)

        token_freq = nltk.FreqDist(itertools.chain(*formatted))

        print("Found %d unique tokens" % len(token_freq.items()))
        vocab = sorted(token_freq.items(), key=lambda x: (x[1], x[0]), reverse = True)[:VOCAB_SIZE - 2]
        print("With vocab size %d, the least frequent token is %s with %d occs." % (VOCAB_SIZE, vocab[-1][0], vocab[-1][1]))

        print("numifying...")

        pairs = []
        print("Start token index: %d" % token2index["START_TOKEN"])

        def toNum(sent):
            return [token2index[x] for x in sent]

        for i, line in enumerate(formatted[:-1]):

            next_line = formatted[i+1]

            if args.equal_pairing:
                pairs.append([toNum(line), toNum(line)])
            else:
                pairs.append([toNum(line), toNum(next_line)])

            if i < int(args.preview_size):
                print(pairs[-1])

        numpy.savez("num_bible", pairs = pairs, index2token = index2token, token2index = token2index)

        print("%d pairs saved" % len(pairs))

clean()
