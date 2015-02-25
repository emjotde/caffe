#!/usr/bin/env python2
# encoding: utf-8

import sys

sys.path.insert(0,  './python')

import caffe
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

def main(argv):
    """ main """
    net = load_engine(PATH_TO_DEPLOY, PATH_TO_MODEL)
    if len(argv) == 1:
        for line in sys.stdin.readlines():
            features = [float(i) for i in line.split()]
            if len(features) == 4:
                print("{} -> {}".format(str(features), predict(net, features)))
    elif len(argv) == 2:
        with open(argv[1]) as csv_file:
            for line in csv_file:
                if  not line.strip():
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





if __name__ == '__main__':
    main(sys.argv)
