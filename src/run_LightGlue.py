from easy_local_features.matching.baseline_lightglue import LightGlue_baseline
from nonrigid_benchmark.ransac import nr_RANSAC
from nonrigid_benchmark.utils import extract,eval_loop
from nonrigid_benchmark.evaluate import eval_pair
from nonrigid_benchmark.io import load_benchmark
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
    baseline = LightGlue_baseline()
    extractor = baseline.extractor
    extract_fn = partial(extract, detectordescriptor=extractor)
    eval_loop(args.dataset,args.output, baseline.match,[args.dataset_type])
