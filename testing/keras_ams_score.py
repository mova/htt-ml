#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
from array import array
import yaml
import pickle
import numpy as np
import os

from keras.models import load_model

import logging
logger = logging.getLogger("keras_ams_score")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate AMS score.")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_testing", help="Path to testing config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))


def main(args, config_test, config_train):
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.debug("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.debug("Load keras model %s.", path)
    model = load_model(path)

    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.debug("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)
    true_positive = []
    for i_class, class_ in enumerate(config_train["classes"]):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception
        values = []
        for variable in config_train["variables"]:
            values.append(array("f", [-999]))
            tree.SetBranchAddress(variable, values[-1])
        weight = array("f", [-999])
        tree.SetBranchAddress(config_test["weight_branch"], weight)

        values_preprocessed = np.zeros(
            (tree.GetEntries(), len(values)), dtype=np.float32)
        weights = np.zeros(tree.GetEntries(), dtype=np.float32)
        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)
            values_stacked = np.hstack(values).reshape(1, len(values))
            values_preprocessed[i_event, :] = preprocessing.transform(
                values_stacked)
            weights[i_event] = weight[0]

        response = model.predict(values_preprocessed)
        response = np.squeeze(response)
        max_index = np.argmax(response, axis=-1)
        mask = (max_index == i_class).astype(np.float)
        true_positive.append(np.sum(mask * weights))

    for i_class, class_ in enumerate(config_train["classes"]):
        tp = float(
            np.sum([x for i, x in enumerate(true_positive) if i == i_class]))
        fp = float(
            np.sum([x for i, x in enumerate(true_positive) if i != i_class]))

        fp_reg = 10.0
        ams = np.sqrt(
            2 * ((tp + fp + fp_reg) * np.log(1.0 + tp / (fp + fp_reg)) - tp))
        logger.debug("TP (%s): %f", class_, tp)
        logger.debug("FP (%s): %f", class_, fp)
        logger.info("AMS (%s): %f", class_, ams)


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
