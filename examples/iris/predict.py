#!/usr/bin/env python2
# encoding: utf-8

import sys
import argparse
import h5py
import numpy as np
from itertools import izip


PATH_TO_DEPLOY = './examples/iris/deploy.prototxt'
PATH_TO_MODEL = './examples/iris/snapshot_iter_1000.caffemodel'

IRIS_INDEX = {'Iris-setosa' : 0,
              'Iris-versicolor' : 1,
              'Iris-virginica' : 2,}

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_engine(path_to_deploy, path_to_model):
    net  = caffe.Classifier(path_to_deploy, path_to_model)
    return net

def predict(net, features):
    testcase = np.ndarray(shape=(1, 1, 4), dtype='float32')

    for i in range(len(features)):
        testcase[0][0][i] = features[i]
    testcase.dtype = 'float32'
    p = net.predict([testcase])
    argmax = lambda array: max(izip(array, xrange(len(array))))[1]
    return argmax(p[0])

def predict_csv(net, filename):
    with open(filename, 'r') as csv_file:
        for line in csv_file:
            if not line.strip():
                continue
            features = [float(i) for i in line.split(',')[:4]]
            correct = IRIS_INDEX[line.split(',')[4].strip()]
            prediction = predict(net, features)
            if correct == prediction:
                sys.stdout.write(bcolors.OKGREEN)
            else:
                sys.stdout.write(bcolors.FAIL)
            print("{} -> {} (Correct : {})".format(str(features),
                                                    prediction,
                                                    correct))
            sys.stdout.write(bcolors.ENDC)

def predict_stdin(net):
    for line in sys.stdin.readlines():
        features = [float(i) for i in line.split()]
        if len(features) == 4:
            print("{} -> {}".format(str(features), predict(net, features)))

def set_args():
    parser = argparse.ArgumentParser('The prediction script for the Iris set.')
    parser.add_argument('-model', help='Path to the model (*.caffemodel file)')
    parser.add_argument('-deploy', help='Path to the deploy file')
    parser.add_argument('-csv', help='Path to the csv file containing testcases')
    parser.add_argument('-pycaffe', help='Path to caffe/python dictionary')
    return parser.parse_args()

def main(argv):
    """ main """
    args = set_args()
    PATH_TO_MODEL = args.model
    PATH_TO_DEPLOY = args.deploy
    sys.path.insert(0,  args.pycaffe)
    import caffe

    net = caffe.Classifier(PATH_TO_DEPLOY, PATH_TO_MODEL)
    if args.csv:
        predict_csv(net, args.csv)
    else:
        predict_stdin(net)

if __name__ == '__main__':
    main(sys.argv)
