#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
import root_numpy
import numpy as np
np.random.seed(1234)

import argparse
import yaml
import os
import pickle

from sklearn import preprocessing, model_selection
import keras_models

import logging
logger = logging.getLogger("keras_training")


def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(
        description="Train machine Keras models for Htt analyses")
    parser.add_argument("config", help="Path to training config file")
    parser.add_argument("fold", type=int, help="Select the fold to be trained")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Parse config.")
    return yaml.load(open(filename, "r"))


def setup_logging(level, output_file=None):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not output_file == None:
        file_handler = logging.FileHandler(output_file, "w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def main(args, config):
    # Extract list of variables
    variables = config["variables"]
    logger.debug("Use variables:")
    for v in variables:
        logger.debug("%s", v)

    # Load training dataset
    filename = config["datasets"][args.fold]
    logger.debug("Load training dataset from %s.", filename)
    x = []
    y = []
    w = []
    rfile = ROOT.TFile(filename, "READ")
    classes = config["classes"]
    for i_class, class_ in enumerate(classes):
        logger.debug("Process class %s.", class_)
        tree = rfile.Get(class_)
        if tree == None:
            logger.fatal("Tree %s not found in file %s.", class_, filename)
            raise Exception

        # Get inputs for this class
        x_class = np.zeros((tree.GetEntries(), len(variables)))
        x_conv = root_numpy.tree2array(tree, branches=variables)
        for i_var, var in enumerate(variables):
            x_class[:, i_var] = x_conv[var]
        x.append(x_class)

        # Get weights
        w_class = np.zeros((tree.GetEntries(), 1))
        w_conv = root_numpy.tree2array(
            tree, branches=[config["event_weights"]])
        w_class[:, 0] = w_conv[config["event_weights"]] * config[
            "class_weights"][class_]
        w.append(w_class)

        # Get targets for this class
        y_class = np.zeros((tree.GetEntries(), len(classes)))
        y_class[:, i_class] = np.ones((tree.GetEntries()))
        y.append(y_class)

    # Stack inputs, targets and weights to a Keras-readable dataset
    x = np.vstack(x)  # inputs
    y = np.vstack(y)  # targets
    w = np.vstack(w) * config["global_weight_scale"]  # weights
    w = np.squeeze(w)  # needed to get weights into keras

    # Perform input variable transformation and pickle scaler object
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
        logger.debug("Preprocessing (variable, mean, std): %s, %s, %s", var,
                     mean, std)

    path_preprocessing = os.path.join(
        config["output_path"],
        "fold{}_keras_preprocessing.pickle".format(args.fold))
    logger.info("Write preprocessing object to %s.", path_preprocessing)
    pickle.dump(scaler, open(path_preprocessing, 'wb'))

    # Split data in training and testing
    x_train, x_test, y_train, y_test, w_train, w_test = model_selection.train_test_split(
        x, y, w, test_size=1.0 - config["train_test_split"], random_state=1234)

    # Train model
    model = keras_models.example(len(variables), len(classes))
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        validation_data=(x_test, y_test, w_test),
        batch_size=100,
        nb_epoch=10,
        shuffle=True)

    # Save model
    path_model = os.path.join(config["output_path"],
                              "fold{}_keras_model.h5".format(args.fold))
    logger.info("Write model to %s.", path_model)
    model.save(path_model)


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
