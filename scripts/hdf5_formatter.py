#!/usr/bin/python

import os
import sys

import h5py
import numpy
from joblib import Parallel, delayed
import argparse


INPUT_FORMATS = 'libsvm'.split()

BATCH_SIZE = 1024
THREADS = 8
COMPRESSION_LEVEL = 1 # between 0 and 9 

DEBUG_COUNTER = 5000
DEBUG = False


def main():
    args = parse_user_args()

    data, labels, vec_size = read_libsvm_data(args.input)
    debug("number of data: {}".format(len(data)))

    vec_size = max(vec_size, args.vector_size)
    debug("number of features: {}".format(vec_size))

    batches = (len(data) / args.batch_size) + 1
    debug("number of batches: {}".format(batches))

    debug("number of threads: {}".format(args.threads))
    jobs = []

    for batch in range(batches):
        n = batch * args.batch_size
        m = min(n + args.batch_size, len(data))
        hdf5_file = "{0}.{1:03d}".format(os.path.abspath(args.output), batch)
        print hdf5_file

        debug("split data from {} to {}".format(n, m))
        jobs.append(delayed(save_data_to_hdf5)(hdf5_file, data[n:m], labels[n:m], vec_size))

    Parallel(n_jobs=args.threads, verbose=DEBUG)(jobs)


def read_libsvm_data(libsvm_file):
    debug("reading file {} in libsvm format...".format(libsvm_file))

    data = []
    labels = []
    max_idx = 0

    with open(libsvm_file, 'r') as file:
        for line in file:
            fields = line.strip().split()

            datum = {}
            for field in fields[1:]:
                idx, val = field.split(':')
                # in libsvm format first dimension is a label
                datum[int(idx) - 1] = float(val)
            max_idx = max(max_idx, max(datum.keys()))
            data.append(datum)

            # labels in libsvm format are counted from 1
            labels.append(int(fields[0]) - 1)

    debug("last feature index: {}".format(max_idx))

    return (data, labels, max_idx + 1)

def save_data_to_hdf5(hdf5_file, data, labels, vec_size):
    debug("creating compressed HDF5 file {}...".format(hdf5_file))

    with h5py.File(hdf5_file, 'w') as file:
        dset = file.create_dataset('data', shape=(len(data), vec_size, 1, 1),
                                   dtype=numpy.float16, fillvalue=0.0,
                                   compression='gzip', 
                                   compression_opts=COMPRESSION_LEVEL)

        for i, row in enumerate(data):
            for idx, val in row.iteritems():
                dset[i, idx, 0, 0] = val
            if (i+1) % DEBUG_COUNTER == 0:
                debug("[{}]".format(i+1))

        lset = file.create_dataset('label', shape=(len(data), 1), 
                                   dtype=numpy.float16,
                                   compression='gzip', 
                                   compression_opts=COMPRESSION_LEVEL)

        for i, lbl in enumerate(labels):
            lset[i, 0] = lbl

    debug("file {} is ready".format(hdf5_file))


def debug(msg):
    if DEBUG:
        print >> sys.stderr, msg
        
def parse_user_args():
    parser = argparse.ArgumentParser(description="Converts data from " \
        "LibSVM format to HDF5 format. Script writes list of paths to " \
        "created HDF5 files into STDOUT.")

    parser.add_argument('input', help="input file")
    parser.add_argument('output', help="prefix for output HDF5 files")

    #parser.add_argument('-f', '--format', choices=INPUT_FORMATS, 
        #default='libsvm', help="format of input data")
    parser.add_argument('-v', '--vector-size', type=int,
        help="size of feature vector")
    parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE,
        help="batch size per HDF5 file (= {})".format(BATCH_SIZE))
    parser.add_argument('-t', '--threads', type=int, default=THREADS,
        help="number of CPUs (= {})".format(THREADS))

    parser.add_argument('-d', '--debug', action='store_true', 
        help="debug mode")

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    return args

if __name__ == '__main__':
    main()
