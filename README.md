# Htt Machine Learning Scripts

This repository holds scripts used for training and application of various machine learning methods used for Htt analyses.

## Set up the software stack

`CVMFS` is used to source the appropriate software stacks. Locate the `CVMFS` mount point on your system and run the following setup script. Please note that you have to select the appropriate architecture for your system.

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

The training is performed using `TMVA` so that a rapid evaluation of different machine learning methods is possible. The preferred methods are neural networks implemented with the `keras` framework, which can be trained using the `PyMVA` interface of `ROOT.TMVA`.

### Create a training dataset

For training, we need to extract a training dataset from the Monte Carlo and data files. To do so, run the script `create_training_dataset.py` with the following command line options.

```bash
./dataset/create_training_dataset.py /dataset/example_dataset_config.yaml
```

The script creates a two-fold dataset using even and odd event numbers, which are stored separately in `fold*.root` files.

### Run the training

The script `TMVA_training.py` implements a TMVA workflow for training multiclass machine learning methods on the toy dataset. It displays the training of `keras` models using `PyMVA` and a comparison with the TMVA BDT implementation. The configuration of variables and classes happens in the config file `example_training_config.yaml`. The last command line parameter defines the fold to be trained.

```bash
./training/TMVA_training.py training/example_training_config.yaml 0
```

## Application

TODO: TMVA, lwtnn
