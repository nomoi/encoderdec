import time
import sys
import numpy as np

def now():
    return int(time.time() * 1000)

def loadFile(filestr):
    print("[Loading %s...]" % filestr,)
    sys.stdout.flush()

    time_start = now()
    f = np.load(filestr)
    print("[Took %d milliseconds]" % (now() - time_start))
    return f

def iprint(x):
    print(x, end="", flush=True)

