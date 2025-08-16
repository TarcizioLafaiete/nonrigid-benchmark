from easy_local_features.feature.baseline_dedode import  DeDoDe_baseline
from nonrigid_benchmark.utils import eval_loop
from functools import partial

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import torch
import multiprocessing
from tqdm import tqdm

baseline = DeDoDe_baseline()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    return parser.parse_args()

def match(image1, image2):

    matches = baseline.match(image1,image2)

    kpts1 = matches['mkpts0'].cpu().numpy().tolist()
    kpts2 = matches['mkpts1'].cpu().numpy().tolist()

    matches_index = []


    i = 0
    for kp in kpts1:
        matches_index.append((i,i))
        i+=1

    return kpts1,kpts2,matches_index    

if __name__ == "__main__":


    args = parse()
    eval_loop(args.dataset,args.output,match,[args.dataset_type])