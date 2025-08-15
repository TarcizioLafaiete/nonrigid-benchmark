from easy_local_features.feature.baseline_sosnet import SOSNet_baseline 
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

baseline = SOSNet_baseline()
sift = cv2.SIFT_create()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    return parser.parse_args()

def cv2kp_to_tensor(kp):
    pts = np.array([k.pt for k in kp], dtype=np.float32)  # N x 2
    return torch.from_numpy(pts)  # (N, 2)

def detect(img):
    
    kp, _ = sift.detectAndCompute(img,None)
    return cv2kp_to_tensor(kp)

def compute(img,kp):
    return baseline.compute(img,kp)



def match(image1, image2):
        
        kp1 = detect(image1)
        kp2 = detect(image2)

        kpts1,desc1 = compute(image1,kp1)
        kpts2,desc2 = compute(image2,kp2)

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