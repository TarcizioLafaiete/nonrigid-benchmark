from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from nonrigid_benchmark.utils import eval_loop

import cv2
import torch
import argparse
import numpy as np
from skimage.feature import daisy

sift = cv2.SIFT_create()
matcher = NearestNeighborMatcher()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    return parser.parse_args()


def compute_daisy(img, kpts):
    # Computa todos os descritores da imagem de uma só vez
    desc_map = daisy(
        img,
        step=1,
        radius=15,
        rings=3,
        histograms=8,
        orientations=8,
        visualize=False
    )  # shape: (H', W', D)

    # Ajustar keypoints para o grid da saída do daisy
    descs = []
    for kp in kpts:
        y, x = int(kp.pt[1]), int(kp.pt[0])
        if 0 <= y < desc_map.shape[0] and 0 <= x < desc_map.shape[1]:
            descs.append(desc_map[y, x])
    
    return np.array(descs, dtype=np.float32)
    

def match(image1,image2):

    kpts1,desc1 = sift.detectAndCompute(image1,None)
    kpts2,desc2 = sift.detectAndCompute(image2,None)

    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    desc1 = compute_daisy(image1,kpts1)
    desc2 = compute_daisy(image2,kpts2)

    desc1_t = torch.from_numpy(desc1).unsqueeze(0)
    desc2_t = torch.from_numpy(desc2).unsqueeze(0)

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