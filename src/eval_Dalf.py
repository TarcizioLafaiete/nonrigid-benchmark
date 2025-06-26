from nonrigid_benchmark.baselines.xfeat import warpXFeat
from nonrigid_benchmark.baselines.superglue import SuperGlue_baseline,SuperPoint_baseline
from nonrigid_benchmark.ransac import nr_RANSAC
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
    parser.add_argument("--dataset_type", default='test_single_obj',help="Dataset type")
    parser.add_argument("--metric_file",default="metrics",help="Doc where the evaluation will be stored")
    return parser.parse_args()



if __name__ == "__main__":


    args = parse()
    extractor = warpXFeat()
    extract_fn = partial(extract, detectordescriptor=extractor)
    eval_loop_cached(args.dataset,args.output, extract_fn, extractor.match,[args.dataset_type])

    selected_pairs = load_benchmark(os.path.join(args.dataset, args.dataset_type))

    nproc = args.nproc
    
    predictions_file = [f for f in os.listdir(args.input) if f.endswith('.json')]
    split_names = [os.path.splitext(f)[0] for f in predictions_file]

    if os.path.exists("progress.json"):
        with open("progress.json", 'r') as f:
            data = json.load(f)
    else:
        data = {"passed": []}
    
    for pred_file in predictions_file:

        skip = 0
        total = 0

        if pred_file in data['passed']: 
            continue

        

        print(f"Evalution: {pred_file}")
        predictions = json.load(open(args.input + pred_file,'r'))
        split = os.path.splitext(pred_file)[0]
        outfile_path = args.metric_file + "metric_" +  split + ".json"
        
        metrics = {
            'ms': [],
            'ma': [],
            'rr': [],
            'arap_3d_accurracy_0': [],
            'arap_3d_accurracy_1': [],
            'arap_3d_accurracy_2': [],
            'arap_2d_accurracy_0': [],
            'arap_2d_accurracy_1': [],
            'arap_2d_accurracy_2': [],
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
                total += 1
                if result['arap_skip']:
                    skip +=1
            
        # mean score
        ms = np.mean(metrics['ms'])
        ma = np.mean(metrics['ma'])
        rr = np.mean(metrics['rr'])
        arap2d = np.mean(metrics['arap_2d_accurracy_0']),np.mean(metrics['arap_2d_accurracy_1']),np.mean(metrics['arap_2d_accurracy_2'])
        arap3d = np.mean(metrics['arap_3d_accurracy_0']),np.mean(metrics['arap_3d_accurracy_1']),np.mean(metrics['arap_3d_accurracy_2'])

        output_dict = {
            "Matching Score": ms,
            "Matching Accuracy": ma,
            "Repeatability": rr,
            "ARAP Registration 2D Accuracy": arap2d,
            "ARAP Registration 3D Accuracy": arap3d,
            "ARAP Skipping": skip,
            "Skip Proportion": skip / total if total != 0 else None
        }

        with open(outfile_path, 'w') as f:
            json.dump(output_dict, f, indent=4)

        data['passed'].append(pred_file)
        with open("progress.json", 'w') as f:
            json.dump(data, f, indent=4)

        
    
