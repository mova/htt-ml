#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
from array import array
import yaml
import pickle
import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.models import load_model

import logging
logger = logging.getLogger("keras_check_performance")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Produce confusion matrice")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_testing", help="Path to testing config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    parser.add_argument("target_class", type=str, help="Class to be tested.")
    parser.add_argument(
        "cut", type=float, help="Cut 'score>cut' on score of target class.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))


def main(args, config_test, config_train):
    logger.fatal("DEPRECATED")
    raise Exception
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.info("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.info("Load keras model %s.", path)
    model = load_model(path)

    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.info("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)

    i_target_class = config_train["classes"].index(args.target_class)
    counts = {c: 0.0 for c in config_train["classes"]}
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

        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)
            values_stacked = np.hstack(values).reshape(1, len(values))
            values_preprocessed = preprocessing.transform(values_stacked)
            response = model.predict(values_preprocessed)
            response = np.squeeze(response)
            if response[i_target_class] > args.cut:
                counts[class_] += weight[0]

    num_signal = 0.0
    num_background = 0.0
    for c in counts:
        if c == args.target_class:
            num_signal += counts[c]
        else:
            num_background += counts[c]

    logger.info("Target class %s with cut score > %s selected.",
                args.target_class, args.cut)
    logger.info("Number of signal events: %s", num_signal)
    logger.info("Number of background events: %s", num_background)
    logger.info("Purity: %s", num_signal / (num_signal + num_background))
    logger.info("Significance: %s",
                num_signal / np.sqrt(num_signal + num_background))


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
