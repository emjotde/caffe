#!/bin/bash

# This script downloads the enron corpus. The enron corpus is divided into six packages.
# You can provide by the argument the number of package to download. Default, it is set to six.
# The whole corpora contains ~16K spam emails and ~17K ham emails.

NPACKAGE=${1:-6}

for i in $(seq 1 $NPACKAGE)
do
	wget http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron$i.tar.gz
	tar xvf enron$i.tar.gz --transform "s|^enron$i||"
done

