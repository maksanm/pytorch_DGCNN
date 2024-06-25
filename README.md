PyTorch DGCNN
=============

About
-----

Adjusted fork of the PyTorch implementation of DGCNN (Deep Graph Convolutional Neural Network). Check https://github.com/muhanzhang/DGCNN for more information.


Installation
------------

This implementation is based on Hanjun Dai's structure2vec graph backend. Under the "lib/" directory, type

    make -j4

to compile the necessary c++ files.

After that, under the root directory of this repository, type

    ./run_DGCNN.sh

to run DGCNN on dataset MUTAG with the default setting, or type 

    ./run_DGCNN.sh $DATASET_NAME $bsize

to run 10-fold cross-validation on dataset = `$DATASET_NAME` using batch size = `$bsize`.

Check "run_DGCNN.sh" for more options.

Datasets
--------

Default graph datasets are stored in "data/DSName/DSName.txt". Check the "data/README.md" for the format. 

In addition to the support of discrete node labels (tags), DGCNN now supports multi-dimensional continuous node features. One example dataset with continuous node features is "Synthie". Check "data/Synthie/Synthie.txt" for the format. 

There are two preprocessing scripts in MATLAB: "mat2txt.m" transforms .mat graphs (from Weisfeiler-Lehman Graph Kernel Toolbox), "dortmund2txt.m" transforms graph benchmark datasets downloaded from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

How to use your own data
------------------------

The first step is to transform your graphs to the format described in "data/README.md". You should put your testing graphs at the end of the file. Then, there is an option -test_number X, which enables using the last X graphs from the file as testing graphs. You may also pass X as the third argument to "run_DGCNN.sh" by

    ./run_DGCNN.sh $DATASET_NAME $bsize
