#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
import yaml
import os
import subprocess

import logging
logger = logging.getLogger("create_training_dataset")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(description="Create training dataset")
    parser.add_argument("config", help="Datasets config file")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load YAML config: {}".format(filename))
    return yaml.load(open(filename, "r"))


def main(args, config):
    for num_fold in range(2):
        logger.info("Merge input files for fold {}.".format(num_fold))
        created_files = []
        for process in config["processes"]:
            logger.debug("Collect events of process {} for fold {}.".format(
                process, num_fold))
            # Create output file
            created_files.append(
                os.path.join(config["output_path"],
                             "merge_fold{}_{}.root".format(num_fold, process)))
            file_ = ROOT.TFile(created_files[-1], "RECREATE")

            # Collect all files for this process in a chain
            chain = ROOT.TChain(config["tree_path"])
            for filename in config["processes"][process]["files"]:
                chain.AddFile(os.path.join(config["base_path"], filename))
            logger.debug("Found {} events for process {}.".format(
                chain.GetEntries(), process))

            # Skim the events with the cut string
            cut_string = "({EVENT_BRANCH}%2=={NUM_FOLD})&&({CUT_STRING})".format(
                EVENT_BRANCH=config["event_branch"],
                NUM_FOLD=num_fold,
                CUT_STRING=config["processes"][process]["cut_string"])
            logger.debug("Skim events with cut string: {}".format(cut_string))
            chain_skimmed = chain.CopyTree(cut_string)
            logger.debug("Found {} events for process {} after skimming.".
                         format(chain.GetEntries(), process))

            # Rename chain to process name and write to output file
            logger.debug("Write output file for this process and fold.")
            chain_skimmed.SetName(process)
            chain_skimmed.Write()
            file_.Close()

        # Combine all skimmed files using `hadd`
        logger.debug("Call `hadd` to combine files of processes for fold {}.".
                     format(num_fold))
        output_file = os.path.join(config["output_path"], "fold{}_{}".format(
            num_fold, config["output_filename"]))
        subprocess.call(["hadd", "-f", output_file] + created_files)
        logger.info("Created output file: {}".format(output_file))


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
