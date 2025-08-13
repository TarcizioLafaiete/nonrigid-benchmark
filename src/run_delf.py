from easy_local_features.feature.baseline_delf import DELF_baseline 
from nonrigid_benchmark.utils import eval_loop
from functools import partial

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import multiprocessing
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    return parser.parse_args()



if __name__ == "__main__":


    args = parse()
    baseline = DELF_baseline()
    eval_loop(args.dataset,args.output,baseline.match,[args.dataset_type])