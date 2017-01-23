# Batch Learn

Batch-Learn is an implementation of ML algorithms which may be applied to on-disk data batch-by-batch, without loading full dataset to memory.

Algorithms included:
* FFM
* NN / MLP

It's extracted from the [code](https://github.com/alno/kaggle-outbrain-click-prediction) written during [Outbrain Click Prediction](https://www.kaggle.com/c/outbrain-click-prediction/) competition on Kaggle and now is undergoing some rewrite and refactoring.

## Installation

Batch-learn uses [CMake](https://cmake.org/) as a build tool and depends on following libraries:
* boost-program-options
* boost-iostreams

To compile code you need to install boost libraries and then call:

    mkdir build
    cd build
    cmake ..
    make

## Usage

First, you need to convert to batch-learn format:

    batch-learn convert -f ffm -b 24  ffm_dataset.txt -O bl_dataset

To train ffm model and make predictions on test dataset:

    batch-learn ffm --train tr1 --test te1 --pred pred.txt

You also may specify validation dataset:

    batch-learn ffm --train tr1 --test te1 --val va1 --pred pred.txt

To get list of available commands just run:

    batch-learn help

To get help about some specific command:

    batch-learn help ffm
