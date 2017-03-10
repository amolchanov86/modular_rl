#!/usr/bin/env python
import sys
import os
import argparse
import atexit
from collections import defaultdict
import yaml

import numpy as np
from matplotlib import pyplot as plt


def load_params(filename):
    yaml_stream = file(filename, 'r')
    return yaml.load(yaml_stream)

# ===========================
#   Main function
# ===========================
def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p","--params",
        default="config/train_params.yaml",
        help="Config yaml file with parameters"
    )
    args = parser.parse_args()

    params = load_params(args.params)
    print 'Params = ', params


if __name__ == '__main__':
    main(sys.argv)