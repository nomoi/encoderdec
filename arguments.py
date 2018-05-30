import argparse

def parse():
    parser = argparse.ArgumentParser(description="code for running theano")
    parser.add_argument("input", help=".npz file to be read")
    parser.add_argument("-hi", "--hidden-size", type=int, help="number of hidden units", default=500)
    parser.add_argument("-l", "--learning-rate", type=float, help="scaling factor for step", default=0.005)
    parser.add_argument("-e", "--evaluation-rate", type=int, help="epochs per error evaluation", default=1)
    parser.add_argument("-s", "--save-rate", type=int, help="epochs per model backup", default=20)
    args = parser.parse_args()
    return args

