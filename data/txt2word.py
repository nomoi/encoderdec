import argparse
import random
import nltk
import operator
import itertools
import numpy
import re

parser = argparse.ArgumentParser(description="script for converting plaintext files into training token data for seq2seq")
parser.add_argument("input", help="file to be read")
parser.add_argument("-s", "--size-vocab", type=int, help="tokens not common enough will become 'UNK'", required=True)
parser.add_argument("-p", "--preview-size", type=int, help="number of lines to preview for quality check", default=0)
parser.add_argument("-e", "--equal-pairing", action='store_true', help="make both pairs the same")
parser.add_argument("--max-length", type=int, help="longest token length of line to store", default=1000)
parser.add_argument("--min-length", type=int, help="minimum token length of line to store", default=1)
args = parser.parse_args()

START_TOKEN = "START_TOKEN"
END_TOKEN = "END_TOKEN"
UNK = "UNK"
VOCAB_SIZE = args.size_vocab
FILENAME = args.input

tokenizer = nltk.tokenize.TweetTokenizer()


def clean():
    with open(FILENAME) as f:

        lines = [ tokenizer.tokenize(x.strip()) for x in f.readlines()]
        formatted = []
        for line in lines:

            if len(line) < args.min_length or len(line) > args.max_length:
                formatted_line = [START_TOKEN, END_TOKEN]
            else:
                formatted_line = [START_TOKEN] + line + [END_TOKEN]

            formatted.append(formatted_line)

        token_freq = nltk.FreqDist(itertools.chain(*formatted))

        print("Found %d unique tokens" % len(token_freq.items()))
        vocab = sorted(token_freq.items(), key=lambda x: (x[1], x[0]), reverse = True)[:VOCAB_SIZE - 2]
        print("With vocab size %d, the least frequent token is %s with %d occs." % (VOCAB_SIZE, vocab[-1][0], vocab[-1][1]))

        sortedVocab = sorted(vocab, key = operator.itemgetter(1))
        index2token = [UNK] + [x[0] for x in sortedVocab]

        print("Replacing rare tokens")
        common = []
        for line in formatted:
            common.append([w if w in index2token else UNK for w in line])

        numpy.savez("cleaned_bible", common = common, index2token = index2token)


def numify():
    print("numifying...")
    f = numpy.load("cleaned_bible.npz")
    index2token = f["index2token"]
    common = f["common"]

    token2index = {}
    for i,e in enumerate(index2token):
        token2index[e] = i

    pairs = []
    print("Start token index: %d" % token2index["START_TOKEN"])

    def toNum(sent):
        return [token2index[x] for x in sent]

    skipped = 0
    for i, line in enumerate(common[:-1]):

        next_line = common[i+1]

        if len(line) == 2 or len(next_line) == 2:
            skipped += 1
            continue

        if args.equal_pairing:
            pairs.append([toNum(line), toNum(line)])
        else:
            pairs.append([toNum(line), toNum(next_line)])

        if i < int(args.preview_size):
            print(pairs[-1])

    numpy.savez("num_bible", pairs = pairs, index2token = index2token, token2index = token2index)

    print("Skipped %d pairs" % skipped)
    print("%d pairs saved" % len(pairs))

clean()
numify()
