This repo includes a python implementation of a k-means classification algorithm which can operate on .mat files.
The actual algorithm, contained in kmeans.py, can take any n x d data in a .mat file, but the test script contained
in main.py only uses the first two dimensions because it plots the clusters with corresponding means.