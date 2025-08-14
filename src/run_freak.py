from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from nonrigid_benchmark.utils import eval_loop

import cv2
import torch
import argparse
import numpy as np

detector = cv2.FastFeatureDetector_create()
freak = cv2.xfeatures2d.FREAK_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    return parser.parse_args()
    

def match(image1,image2):

    kp1 = detector.detect(image1,None)
    kp2 = detector.detect(image2,None)

    kp1,desc1 = freak.compute(image1,kp1)
    kp2,desc2 = freak.compute(image2,kp2)

    matches = matcher.match(desc1,desc2)

    matches = sorted(matches,key=lambda x: x.distance)
    matches_idx = [(m.queryIdx,m.trainIdx) for m in matches]

    kpts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)
    kpts2 = np.array([kp.pt for kp in kp2], dtype=np.float32)

    return kpts1.tolist(), kpts2.tolist(), matches_idx

if __name__ == "__main__":


    args = parse()
    eval_loop(args.dataset,args.output,match,[args.dataset_type])