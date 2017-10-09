#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
import argparse
import yaml
import numpy as np
import os
from array import array

np.random.seed(1234)  # makes the output files reproducible


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Write example dataset used for testing the scripts")
    parser.add_argument("output_path", help="Path to output files")
    return parser.parse_args()


def main(args):
    # Processes, which contribute to the dataset
    # The observables of the processes are the variables "x" and "y", which follow a Gaussian 2D distributions with mean "mean", unity variance and correlation "corr".
    processes = {
        "signal": {
            "num_entries": 1000,
            "norm": 0.1,
            "mean": [0.0, 0.0],
            "corr": 0.0
        },
        "background_1": {
            "num_entries": 1000,
            "norm": 0.6,
            "mean": [0.5, 0.5],
            "corr": -0.7
        },
        "background_2": {
            "num_entries": 1000,
            "norm": 0.3,
            "mean": [-0.5, -0.5],
            "corr": 0.7
        }
    }

    # Write Monte Carlo-like output files
    for process in processes:
        file_ = ROOT.TFile("{}.root".format(process), "RECREATE")
        tree = ROOT.TTree(process, process)

        x = array('f', [-999])
        tree.Branch('x', x, 'x/F')
        y = array('f', [-999])
        tree.Branch('y', y, 'y/F')
        weights = array('f', [-999])
        tree.Branch('weights', weights, 'weights/F')

        covariance_matrix = [[1.0, processes[process]["corr"]],
                             [processes[process]["corr"], 1.0]]
        values = np.random.multivariate_normal(
            processes[process]["mean"],
            covariance_matrix,
            size=processes[process]["num_entries"])

        for value in values:
            x[0] = value[0]
            y[0] = value[1]
            weights[0] = processes[process]["norm"] / float(
                processes[process]["num_entries"])
            tree.Fill()

        file_.Write()
        file_.Close()

    # Write data-like output file
    file_ = ROOT.TFile("data.root", "RECREATE")
    tree = ROOT.TTree("data", "data")

    x = array('f', [-999])
    tree.Branch('x', x, 'x/F')
    y = array('f', [-999])
    tree.Branch('y', y, 'y/F')

    for process in processes:
        covariance_matrix = [[1.0, processes[process]["corr"]],
                             [processes[process]["corr"], 1.0]]
        values = np.random.multivariate_normal(
            processes[process]["mean"],
            covariance_matrix,
            size=int(processes[process]["num_entries"] *
                     processes[process]["norm"]))

        for value in values:
            x[0] = value[0]
            y[0] = value[1]
            tree.Fill()

    file_.Write()
    file_.Close()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
