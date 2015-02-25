#!/bin/bash

./examples/iris/predict.py -model examples/iris/snapshot_iter_1000.caffemodel \
    -deploy examples/iris/deploy.prototxt -csv examples/iris/iris.data -pycaffe python
