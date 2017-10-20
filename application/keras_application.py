#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
from array import array
import yaml
import pickle
import numpy as np

from keras.models import load_model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Apply machine learning methods for Htt analyses")
    parser.add_argument("config_dataset", help="Path to dataset config file")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument(
        "config_application", help="Path to application config file")
    parser.add_argument(
        "input", help="Path to input file, where response will be added.")
    parser.add_argument("tree", help="Path to tree in the ROOT input file")
    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"))


def main(args, config_dataset, config_training, config_application):
    # Load keras model and preprocessing
    classifiers = []
    preprocessing = []
    for c, p in zip(config_application["classifiers"],
                    config_application["preprocessing"]):
        classifiers.append(load_model(c))
        preprocessing.append(pickle.load(open(p, "rb")))

    # Open input file and register branches with input and output variables
    file_ = ROOT.TFile(args.input, "UPDATE")
    if file_ == None:
        raise Exception("File is not existent: {}".format(args.input))

    tree = file_.Get(args.tree)
    if tree == None:
        raise Exception("Tree {} is not existent in file: {}".format(
            args.tree, args.input))

    values = []
    for variable in config_training["variables"]:
        values.append(array("f", [-999]))
        tree.SetBranchAddress(variable, values[-1])

    response_branches = []
    response_single_scores = []
    prefix = config_application["branch_prefix"]
    for class_ in config_training["classes"]:
        response_single_scores.append(array("f", [-999]))
        response_branches.append(
            tree.Branch("{}{}".format(prefix, class_), response_single_scores[
                -1], "{}{}/F".format(prefix, class_)))

    response_max_score = array("f", [-999])
    response_branches.append(
        tree.Branch("{}max_score".format(prefix), response_max_score,
                    "{}max_score/F".format(prefix)))

    response_max_index = array("f", [-999])
    response_branches.append(
        tree.Branch("{}max_index".format(prefix), response_max_index,
                    "{}max_index/F".format(prefix)))

    # Loop over events and add method's response to tree
    for i_event in range(tree.GetEntries()):
        # Get current event
        tree.GetEntry(i_event)

        # Get event number and calculate method's response
        event = int(getattr(tree, config_dataset["event_branch"]))
        values_stacked = np.hstack(values).reshape(1, len(values))
        values_preprocessed = preprocessing[event %
                                            2].transform(values_stacked)
        response = classifiers[event % 2].predict(values_preprocessed)
        response = np.squeeze(response)

        # Find max score and index
        response_max_score[0] = -999.0
        for i, r in enumerate(response):
            response_single_scores[i][0] = r
            if r > response_max_score[0]:
                response_max_score[0] = r
                response_max_index[0] = i

        # Fill branches
        for branch in response_branches:
            branch.Fill()

    # Write new branches to input file
    file_.Write()
    file_.Close()


if __name__ == "__main__":
    args = parse_arguments()
    config_dataset = parse_config(args.config_dataset)
    config_training = parse_config(args.config_training)
    config_application = parse_config(args.config_application)
    main(args, config_dataset, config_training, config_application)
