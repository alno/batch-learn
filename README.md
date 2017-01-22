# Batch Learn

Batch-Learn is an implementation of ML algorithms which may be applied to on-disk data batch-by-batch, without loading full dataset to memory.

Algorithms included:
* FFM
* NN / MLP

## Installation

## Usage

First, you need to convert to batch-learn format:

    batch-learn convert -f ffm -b 24  ffm_dataset.txt -O bl_dataset

To train ffm model and make predictions on test dataset:

    batch-learn ffm --train tr1 --test te1 --pred pred.txt

You also may specify validation dataset:

    batch-learn ffm --train tr1 --test te1 --val va1 --pred pred.txt
