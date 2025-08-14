from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from nonrigid_benchmark.utils import eval_loop
from functools import partial

import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import multiprocessing
from tqdm import tqdm

sift = cv2.SIFT_create()
matcher = NearestNeighborMatcher()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    return parser.parse_args()

def match(image1,image2):

    kpts1,desc1 = sift.detectAndCompute(image1,None)
    kpts2,desc2 = sift.detectAndCompute(image2,None)

    desc1_t = torch.from_numpy(desc1).float().unsqueeze(0)  # (1, N1, D)
    desc2_t = torch.from_numpy(desc2).float().unsqueeze(0)  # (1, N2, D)

    matches = matcher({
        'descriptors0': desc1_t,
        'descriptors1': desc2_t
        })

    matches_list = matches['matches0'].cpu().numpy().tolist()

    # Prepara índices válidos
    matches_idx_valid = [
        (i, idx) for i, idx in enumerate(matches_list[0]) if idx >= 0
    ]

    # Extrai coordenadas dos keypoints
    kpts1_np = np.array([kp.pt for kp in kpts1], dtype=np.float32)
    kpts2_np = np.array([kp.pt for kp in kpts2], dtype=np.float32)

    # Retorna listas para serialização
    return kpts1_np.tolist(), kpts2_np.tolist(), matches_idx_valid

if __name__ == "__main__":


    args = parse()
    eval_loop(args.dataset,args.output,match,[args.dataset_type])