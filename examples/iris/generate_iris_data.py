#!/usr/bin/python

import os
import sys
import numpy as np
import h5py
import random

num_cols = 4
num_rows = 0
height = 1
width = 1
total_size = num_rows * num_rows * height * width

IRIS_INPUT_FILE = os.path.dirname(os.path.abspath(__file__)) + '/iris.data'
H5_OUTPUT_FILE = os.path.dirname(os.path.abspath(__file__)) + '/iris.h5'
H5GZIP_OUTPUT_FILE = os.path.dirname(os.path.abspath(__file__)) + '/iris.gzip.h5'

TEST_FRAC = 0.2

print >> sys.stderr, "h5 output = {}".format(H5_OUTPUT_FILE)
print >> sys.stderr, "h5gzip output = {}".format(H5GZIP_OUTPUT_FILE)

iris_data = []

num_of_species = 0
species_to_classes = {}

with open(IRIS_INPUT_FILE, 'r') as iris_file:
    for line in iris_file:
        fields = line.strip().split(',')
        if len(fields) == 5:
            features = [float(v) for v in fields[:-1]]

            if fields[-1] not in species_to_classes:
                species_to_classes[fields[-1]] = num_of_species
                num_of_species += 1
    
            label = species_to_classes[fields[-1]]

            iris_data.append(features + [label])
            num_rows += 1

random.shuffle(iris_data)

for cls, lbl in species_to_classes.iteritems():
    print >> sys.stderr, cls, ': ', lbl

features = [row[0:-1] for row in iris_data]
labels = [row[-1] for row in iris_data]

for i in range(0,10):
    feats = [ str(v) for v in features[i] ]
    print >> sys.stderr, "{}: {} = {}".format(i, ' '.join(feats), labels[i])

print >> sys.stderr, "blob = ({} {} {} {})".format(num_rows, num_cols, height, width)

num_rows_test = int(num_rows * TEST_FRAC)
num_rows_train = num_rows - num_rows_test

test_size = num_rows_test * num_cols * height * width
train_size = num_rows_train * num_cols * height * width

print >> sys.stderr, "test examples: {}".format(test_size)
print >> sys.stderr, "train examples: {}".format(train_size)

train_data = np.array(features[:num_rows_train])
train_data = train_data.reshape(num_rows_train, num_cols, height, width)
train_data = train_data.astype('float32')

test_data = np.array(features[num_rows_train:])
test_data = test_data.reshape(num_rows_test, num_cols, height, width)
test_data = test_data.astype('float32')

# We had a bug where data was copied into label, but the tests weren't
# catching it, so let's make label 1-indexed.
train_label = np.array(labels[:num_rows_train])[:, np.newaxis]
train_label = train_label.astype('float32')

test_label = np.array(labels[num_rows_train:])[:, np.newaxis]
test_label = test_label.astype('float32')


with h5py.File(H5_OUTPUT_FILE + '.train', 'w') as f:
    f['data'] = train_data
    f['label'] = train_label
with h5py.File(H5_OUTPUT_FILE + '.test', 'w') as f:
    f['data'] = test_data
    f['label'] = test_label

with open(H5_OUTPUT_FILE + '.train.txt', 'w') as f:
    f.write(H5_OUTPUT_FILE + ".train\n")
    f.write(H5_OUTPUT_FILE + ".train\n")
with open(H5_OUTPUT_FILE + '.test.txt', 'w') as f:
    f.write(H5_OUTPUT_FILE + ".test\n")

#with h5py.File(H5GZIP_OUTPUT_FILE + , 'w') as f:
    #f.create_dataset(
        #'data', data=data + total_size,
        #compression='gzip', compression_opts=1
    #)
    #f.create_dataset(
        #'label', data=label,
        #compression='gzip', compression_opts=1
    #)
