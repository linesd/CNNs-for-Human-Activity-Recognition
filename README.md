# CNN's for Human Activity Recognition

This repository is a PyTorch implementation of Human Activity Recognition using convolutional neural networks.

 **Notes:**
- Tested for python >= 3.6
- Only tested for CPU

**Table of Contents:**
1. [Install](#install)
2. [Run](#run)
3. [Data](#data)
3. [Models](#models)
4. [Results](#results)

## Install

```
# clone repo
pip install -r requirements.txt
```

## Run

Use `python main.py` to run the preset configuration to train and evaluate the model. The preset 
configuration can be found in `hyperparams.ini`.

To run a custom experiment use `python main.py <experiment name> <params>`. For example:

```
python main.py -n har_1 -d har -b 64 --lr 0.0001 
```

You can evaluate a pre-trained model with the following:

```
python main.py -n har_1 --is-eval-only
```

### Output
Running will create a directory `results/<saving-name>/` which contains:
* **model.pt**: The trained model.
* **specs.json**: The parameters used to run the program (default and those modified with the CLI)

### Help
To get the help menu run `python main.py -h` which yields the following output:

```
usage: main.py [-h] [-d {har,newdataset}] [-b BATCH_SIZE] [--lr LR]
               [-e EPOCHS] [-s IS_STANDARDIZED] [-m {Cnn1,Cnn2}] [-n NAME]
               [--is-eval-only] [--no-test]

PyTorch implementation of CNN's for Human Activity Recognition

optional arguments:
  -h, --help            show this help message and exit

Training specific options:
  -d, --dataset {har,newdataset}
                        Path to training data. (default: har)
  -b, --batch-size BATCH_SIZE
                        Batch size for training. (default: 32)
  --lr LR               Learning rate. (default: 0.0005)
  -e, --epochs EPOCHS   Maximum number of epochs to run for. (default: 20)
  -s, --is_standardized IS_STANDARDIZED
                        Whether to standardize the data. (default: True)

Model specific options:
  -m, --model-type {Cnn1,Cnn2}
                        Type of encoder to use. (default: Cnn2)

General options:
  -n, --name NAME       Name of the model for storing and loading purposes.
                        (default: HAR_1)

Evaluation specific options:
  --is-eval-only        Whether to only evaluate using precomputed model
                        `name`. (default: False)
  --no-test             Whether or not to compute the test losses.` (default:
                        False)

```

## Data

The repository uses the Human Activity Recognition Using Smartphones Data Set available at:
- [HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) 

## Models

Two similar models have been implemented: Cnn1 & Cnn2. The only difference is that Cnn1 has a filter size of 3 and
Cnn2 has a filter of size 5.

Cnn2 architecture:

```
CNN(
  (encoder): Cnn2(
    (conv1): Conv1d(9, 64, kernel_size=(5,), stride=(1,))
    (conv2): Conv1d(64, 64, kernel_size=(5,), stride=(1,))
    (drop): Dropout(p=0.6, inplace=False)
    (pool): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (lin1): Linear(in_features=3840, out_features=100, bias=True)
    (lin2): Linear(in_features=100, out_features=6, bias=True)
  )
)
```

## Results

Pre-trained models for `HAR_Cnn1` and `HAR_Cnn2` can be found in the results folder. 

The following results were achieved on the `fashionMNIST` dataset:
- Epochs: 15
- learning rate: 5e-4
- batch_size: 64

```
***************************************************
*            Evaluating Train Accuracy            *
***************************************************

Train accuracy of the network on the 60000 test images: 98 %

Accuracy of T-shirt/top : 96 %
Accuracy of Trouser : 100 %
Accuracy of Pullover : 95 %
Accuracy of Dress : 98 %
Accuracy of  Coat : 98 %
Accuracy of Sandal : 99 %
Accuracy of Shirt : 95 %
Accuracy of Sneaker : 99 %
Accuracy of   Bag : 99 %
Accuracy of Ankle boot : 96 %

***************************************************
*            Evaluating Test Accuracy             *
***************************************************

Test accuracy of the network on the 10000 test images: 91 %

Accuracy of T-shirt/top : 81 %
Accuracy of Trouser : 100 %
Accuracy of Pullover : 81 %
Accuracy of Dress : 95 %
Accuracy of  Coat : 88 %
Accuracy of Sandal : 97 %
Accuracy of Shirt : 78 %
Accuracy of Sneaker : 97 %
Accuracy of   Bag : 96 %
Accuracy of Ankle boot : 92 %
```

And for the `MNIST` dataset:
- Epochs: 10
- learning rate: 5e-4
- batch_size: 64

```
***************************************************
*            Evaluating Train Accuracy            *
***************************************************

Train accuracy of the network on the 60000 test images: 99 %

Accuracy of 0 - zero : 100 %
Accuracy of 1 - one : 100 %
Accuracy of 2 - two : 99 %
Accuracy of 3 - three : 99 %
Accuracy of 4 - four : 99 %
Accuracy of 5 - five : 99 %
Accuracy of 6 - six : 100 %
Accuracy of 7 - seven : 99 %
Accuracy of 8 - eight : 99 %
Accuracy of 9 - nine : 99 %

***************************************************
*            Evaluating Test Accuracy             *
***************************************************

Test accuracy of the network on the 10000 test images: 99 %

Accuracy of 0 - zero : 100 %
Accuracy of 1 - one : 100 %
Accuracy of 2 - two : 99 %
Accuracy of 3 - three : 98 %
Accuracy of 4 - four : 98 %
Accuracy of 5 - five : 100 %
Accuracy of 6 - six : 100 %
Accuracy of 7 - seven : 99 %
Accuracy of 8 - eight : 99 %
Accuracy of 9 - nine : 98 %
```