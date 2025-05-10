from nonrigid_benchmark.baselines.xfeat import warpXFeat
from nonrigid_benchmark.utils import extract,eval_loop_cached
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
    parser.add_argument("--input", type=str, required=True, help="Path to the matches file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--split", type=str, required=True, help="Split to evaluate")
    parser.add_argument("--matching_th", type=float, default=3, help="Matching threshold")
    parser.add_argument("--nproc", type=int, default=1, help="Parallel processors")
    parser.add_argument("--plot", action='store_true', help="Plot the results")
    parser.add_argument("--dataset_type", default='single_object',help="Dataset type")
    parser.add_argument("--metric_file",default="metrics.json",help="Doc where the evaluation will be stored")
    return parser.parse_args()



if __name__ == "__main__":


    args = parse()
    extractor = warpXFeat()
    extract_fn = partial(extract, detectordescriptor=extractor)
    eval_loop_cached(args.dataset,args.output, extract_fn, extractor.match,[args.dataset_type])

    selected_pairs = load_benchmark(os.path.join(args.dataset, args.dataset_type))

    nproc = args.nproc
    
    predictions = json.load(open(args.input, 'r'))
    
    outfile_path = args.metric_file
    split = args.split
    
    metrics = {
        'ms': [],
        'ma': [],
        'rr': [],
        'arap_3d_accurracy_0': [],
        'arap_3d_accurracy_1': [],
        'arap_3d_accurracy_2': [],
        'arap_2d_accurracy_0': [],
        'arap_2d_accurracy_1': [],
        'arap_2d_accurracy_2': []
    }

    if nproc > 1:    
        with multiprocessing.Pool(nproc) as pool:
            args = [(pair, prediction, args.matching_th, args.plot) for pair, prediction in zip(selected_pairs[split], predictions)]
            results = list(tqdm(pool.imap(eval_pair, args), total=len(args)))
            for result in results:
                metrics['ms'].append(result['ms'])
                metrics['ma'].append(result['ma'])
                metrics['rr'].append(result['rr'])
                metrics['arap_3d_accurracy_0'].append(result['arap_3d_accurracy'][0])
                metrics['arap_3d_accurracy_1'].append(result['arap_3d_accurracy'][1])
                metrics['arap_3d_accurracy_2'].append(result['arap_3d_accurracy'][2])
                metrics['arap_2d_accurracy_0'].append(result['arap_2d_accurracy'][0])
                metrics['arap_2d_accurracy_1'].append(result['arap_2d_accurracy'][1])
                metrics['arap_2d_accurracy_2'].append(result['arap_2d_accurracy'][2])
    else:
        for pair, prediction in zip(selected_pairs[split], predictions):
            result = eval_pair((pair, prediction, args.matching_th, args.plot))
            metrics['ms'].append(result['ms'])
            metrics['ma'].append(result['ma'])
            metrics['rr'].append(result['rr'])
            metrics['arap_3d_accurracy_0'].append(result['arap_3d_accurracy'][0])
            metrics['arap_3d_accurracy_1'].append(result['arap_3d_accurracy'][1])
            metrics['arap_3d_accurracy_2'].append(result['arap_3d_accurracy'][2])
            metrics['arap_2d_accurracy_0'].append(result['arap_2d_accurracy'][0])
            metrics['arap_2d_accurracy_1'].append(result['arap_2d_accurracy'][1])
            metrics['arap_2d_accurracy_2'].append(result['arap_2d_accurracy'][2])
        
    # mean score
    ms = np.mean(metrics['ms'])
    ma = np.mean(metrics['ma'])
    rr = np.mean(metrics['rr'])
    arap2d = np.mean(metrics['arap_2d_accurracy_0']),np.mean(metrics['arap_2d_accurracy_1']),np.mean(metrics['arap_2d_accurracy_2'])
    arap3d = np.mean(metrics['arap_3d_accurracy_0']),np.mean(metrics['arap_3d_accurracy_1']),np.mean(metrics['arap_3d_accurracy_2'])
    
    with open(outfile_path, 'w') as f:
        f.write(f"Matching Score: {ms},Matching Acurracy: {ma},Repeatibility: {rr},Arap Registration 2D Acurracy: {arap2d},Arap Registration 3D Acurray: {arap3d}")
