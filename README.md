# Htt Machine Learning Scripts

This repository holds scripts used for training and application of various machine learning methods used for Htt analyses.

## Set up the software stack

CVMFS is used to source the appropriate software stacks. Locate the CVMFS mount point on your system and run the following setup script. Please note that you have to select the appropriate architecture for your system.

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_91/x86_64-slc6-gcc62-opt/setup.sh
```

All scripts are written with an `argparse` layer on top, which enables help calling `*.py --help`.

## Create the toy dataset

The repository comes with a toy dataset, which can be used for testing the training and application scripts. Run the following command to create `ROOT` files for a signal process and two background processes drawn from multivariate normal distributions.

```bash
./dataset/write_example_dataset.py $PWD
```

## Training

The training is performed using TMVA or plain Keras. TMVA is used so that a rapid evaluation of different machine learning methods is possible. The preferred methods are neural networks implemented with the `keras` framework, which can be trained using the `PyMVA` interface of `ROOT.TMVA`.

### Create a training dataset

For training, we need to extract a training dataset from the Monte Carlo and data files. To do so, run the script `create_training_dataset.py` with the following command line options.

```bash
./dataset/create_training_dataset.py dataset/example_dataset_config.yaml
```

The script creates a two-fold dataset using even and odd event numbers, which are stored separately in `fold*.root` files.

### Set up Keras

Keras is a wrapper for Theano and TensorFlow, which is also available through CVMFS. The following environment variables can be set to control the most important options.

```bash
# Set the keras backend, either 'theano' or 'tensorflow'
export KERAS_BACKEND=theano
# Define the number of cores to be used, works only for theano backend
export OMP_NUM_THREADS=12
# This may be needed on SLC6 architectures
export THEANO_FLAGS=gcc.cxxflags=-march=corei7
```

### TMVA

The script `TMVA_training.py` implements a TMVA workflow for training multiclass machine learning methods on the toy dataset. It displays the training of `keras` models using `PyMVA` and a comparison with the TMVA BDT implementation. The configuration of variables and classes happens in the config file `example_training_config.yaml`. The last command line parameter defines the fold to be trained.

```bash
./training/TMVA_training.py training/example_training_config.yaml 0
./training/TMVA_training.py training/example_training_config.yaml 1
```

### Keras

The training with plain Keras is performed with almost the same calls than the TMVA training with common configs. For preprocessing, it is used `sklearn.model_selection.preprocessing`, which results are serialized to disk using the Python `pickle` module.

```bash
./training/keras_training.py training/example_training_config.yaml 0
./training/keras_training.py training/example_training_config.yaml 1
```

## Testing

The testing of a multiclass application can be done using confusion matrices. These matrices are filled with event weights respective to the true class and the predicted class. Furthermore, the matrix can be normalized respective to the rows, columns or the principal axis. These representations of the confusion matrix are called the purity and efficiency representations.

### Keras

The following command produces the confusion matrices for the Keras workflow.

```bash
./testing/keras_confusion_matrix.py \
    training/example_training_config.yaml testing/example_keras_testing_config.yaml 0
./testing/keras_confusion_matrix.py \
    training/example_training_config.yaml testing/example_keras_testing_config.yaml 1
```

## Application

For application, we use two approaches. The classification of the analysis ntuples using the `TMVA.Reader` allows for a rapid prototyping and fast results on a small dataset. However, the approach is not suitable for millions of events if the `keras` wrapper of `PyMVA` is used. To run quickly over the full dataset with a `keras` model, we recommend to use the [`lwtnn`](https://www.github.com/lwtnn/lwtnn) package.

### TMVA

The script `TMVA_application.py` implements the `TMVA.Reader` using the information in the previously used config files. Additionally, the application config points to the to be used classifiers and defines the names of the newly created branches appended to the input tree. The following command shows the usage for the toy dataset.

```bash
./application/TMVA_application.py \
    dataset/example_dataset_config.yaml \
    training/example_training_config.yaml \
    application/example_TMVA_application_config.yaml \
    example_data.root \
    ntuple
```

Have a look at the toy data file `example_data.root`, which holds new branches with the response of the applied machine learning methods. As discriminating variable in the analysis, it is intended to use the variable `*max_score` with the cut string `*max_index==CLASS_NUMBER`, which selects the desired process.

### Keras

The Keras application script maintains the same arguments than the TMVA based approach. The only difference is the entry `preprocessing` in the application config, which points to the serialized preprocessors. The following command shows the usage for the toy dataset.

```bash
./application/keras_application.py \
    dataset/example_dataset_config.yaml \
    training/example_training_config.yaml \
    application/example_keras_application_config.yaml \
    example_data.root \
    ntuple
```

### Keras with `lwtnn` as inference engine

TODO
