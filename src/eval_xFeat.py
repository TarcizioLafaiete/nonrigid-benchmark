from nonrigid_benchmark.baselines.xfeat import warpXFeat
from nonrigid_benchmark.utils import extract,eval_loop_cached
from nonrigid_benchmark.evaluate import eval_pair
from functools import partial

import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

extractor = warpXFeat()
extract_fn = partial(extract, detectordescriptor=extractor)
# eval_loop_cached(sys.argv[1], sys.argv[2], extract_fn, extractor.match,['single_object'])

sc0 = "../assets/sample_dataset/single_object/sequence_000/scenario_000/rgba_00000.png"
sc0t3 = "../assets/sample_dataset/single_object/sequence_000_deformed_timestep_00003/scenario_000/rgba_00000.png"
pair = [sc0,sc0t3]

predict = "results/single_object/deformation_3.json"
p = {}
with open(predict,'r') as f:
    p = json.load(f)


args = pair,p,3,False
print(eval_pair(args))