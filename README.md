# Batch Learn

Batch-Learn is an implementation of ML algorithms which may be applied to on-disk data batch-by-batch, without loading full dataset to memory.

Algorithms included:
* FFM
* NN / MLP

## Installation

## Usage

First, you need to convert to batch-learn format:

    batch-learn convert -f ffm -b 24  ffm_dataset.txt -O bl_dataset
