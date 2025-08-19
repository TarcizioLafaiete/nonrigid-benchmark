from easy_local_features.feature.baseline_dalf import DALF_baseline
from nonrigid_benchmark.ransac import nr_RANSAC
from nonrigid_benchmark.utils import eval_loop
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

baseline = DALF_baseline()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    return parser.parse_args()


def match(image1, image2):
    kpts1,desc1 = baseline.detectAndCompute(image1)
    kpts2,desc2 = baseline.detectAndCompute(image2)

    matches = baseline.matcher({
    'descriptors0': desc1,
    'descriptors1': desc2
    })

    matches_list = matches['matches0'].cpu().numpy().tolist()

    i = 0
    matches_idx = []
    matches_idx_valid = []
    for idx in matches_list[0]:
        matches_idx.append((i,idx))
        i += 1

    # print(matches_idx)

    for idx0,idx1 in matches_idx:
        if idx1 >= 0:
            matches_idx_valid.append((idx0,idx1))


    kpts1 = kpts1.cpu().numpy()
    kpts2 = kpts2.cpu().numpy() 


    return kpts1.reshape((kpts1.shape[1],kpts1.shape[2])).tolist(),kpts2.reshape((kpts2.shape[1],kpts2.shape[2])).tolist(),matches_idx_valid



if __name__ == "__main__":

    args = parse()
    eval_loop(args.dataset,args.output,match,[args.dataset_type])    