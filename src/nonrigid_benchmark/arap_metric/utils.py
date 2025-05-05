import numpy
import time
import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import os
import re
import cv2
import sys

import kornia
from adalam import AdalamFilter
import torch

import matplotlib.pyplot as plt

import ipywidgets as iw
from IPython.display import display
from scipy.interpolate import Rbf

tps_repo_path = 'modules'
if not os.path.exists(tps_repo_path):
    raise RuntimeError('TPS repository is required')
sys.path.insert(0, tps_repo_path)
from tps import pytorch_og as tps_torch

plots3d = []

#Set AdaLAM outlier filter
DEFAULT_CONFIG = {
    'area_ratio': 25,  # Ratio between seed circle area and image area. Higher values produce more seeds with smaller neighborhoods.
    'search_expansion': 5,  # Expansion factor of the seed circle radius for the purpose of collecting neighborhoods. Increases neighborhood radius without changing seed distribution
    'ransac_iters': 240,  # Fixed number of inner GPU-RANSAC iterations
    'min_inliers': 4,  # Minimum number of inliers required to accept inliers coming from a neighborhood
    'min_confidence': 40,  # Threshold used by the confidence-based GPU-RANSAC
    'orientation_difference_threshold': None,  # Maximum difference in orientations for a point to be accepted in a neighborhood. Set to None to disable the use of keypoint orientations.
    'scale_rate_threshold': None,  # Maximum difference (ratio) in scales for a point to be accepted in a neighborhood. Set to None to disable the use of keypoint scales.
    'detected_scale_rate_threshold': 4,  # Prior on maximum possible scale change detectable in image couples. Affinities with higher scale changes are regarded as outliers.
    'refit': True,  # Whether to perform refitting at the end of the RANSACs. Generally improves accuracy at the cost of runtime.
    'force_seed_mnn': False,  # Whether to consider only MNN for the purpose of selecting seeds. Generally improves accuracy at the cost of runtime. You can provide a MNN mask in input to skip MNN computation and still get the improvement.
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Device to be used for running AdaLAM. Use GPU if available.
}
adalam_matcher = AdalamFilter(DEFAULT_CONFIG)


def plot_inliers2(img_ref, img_tgt, kp_ref, kp_tgt, dist_mat, idx_ref, idx_tgt):
    #print(dist_mat.shape)
    est_tgt = np.argmin(dist_mat, axis=1)
    #print(est_tgt.shape)
    gt_tgt = np.ones(len(kp_ref)) * -1

    gt_tgt[idx_ref] = idx_tgt

    correct = 0
    wrong = 0

    for i in range(len(kp_ref)):
        if not est_tgt[i] == gt_tgt[i]:
            x,y = kp_ref[i].pt
            cv2.drawMarker(img_ref, (int(x),int(y)), (225,0,0), cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA)

            x,y = kp_tgt[est_tgt[i]].pt
            cv2.drawMarker(img_tgt, (int(x),int(y)), (225,0,0), cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA) 
            wrong +=1    
        else:     

            x1,y1 = kp_ref[i].pt
            x1, y1 = int(x1), int(y1)
            cv2.drawMarker(img_ref, (x1, y1), (0,225,0), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

            x2,y2 = kp_tgt[est_tgt[i]].pt
            x2, y2 = int(x2), int(y2)
            cv2.drawMarker(img_tgt, (x2,y2), (0,225,0), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
            correct +=1


    print("MATCHING SCORE: ", correct / min(len(kp_ref), len(kp_tgt)))

    canvas = np.hstack([img_ref, img_tgt])

    for i in range(len(kp_ref)):
        if est_tgt[i] == gt_tgt[i]:
            x1,y1 = kp_ref[i].pt
            x1, y1 = int(x1), int(y1)

            x2,y2 = kp_tgt[est_tgt[i]].pt
            x2, y2 = int(x2), int(y2)

            #Draw lines
            offset = img_ref.shape[1]
            cv2.line(canvas, (x1,y1), (x2+offset, y2), (0,225,0), 1, cv2.LINE_AA)

    plt.figure(figsize=(12,12))
    plt.imshow(img_ref); plt.show()

    plt.figure(figsize=(24,24))
    plt.imshow(np.rot90(canvas)); plt.show()
            


def plot_inliers(img_ref, img_tgt, kp_ref, kp_tgt, idx_inliers):

    img_ref = np.copy(img_ref)
    img_tgt = np.copy(img_tgt)
    kp_ref = np.array([kp.pt for kp in kp_ref]).astype(int)
    kp_tgt = np.array([kp.pt for kp in kp_tgt]).astype(int)

    print(len(idx_inliers) / min(len(kp_ref), len(kp_tgt) ))

    mask = np.zeros(len(kp_ref))
    mask[idx_inliers[:,0]] = 1
    mask = mask.astype(bool)
    kp_ref_inliers = kp_ref[mask]
    kp_ref_outliers = kp_ref[~mask]

    mask = np.zeros(len(kp_tgt))
    mask[idx_inliers[:,1]] = 1
    mask = mask.astype(bool)
    kp_tgt_inliers = kp_tgt[mask]
    kp_tgt_outliers = kp_tgt[~mask]

    for kp in kp_ref_inliers:
        x,y = kp.astype(int)
        #cv2.circle(img_ref, (x,y), 6, (0,225,0), 2, cv2.LINE_AA)
        cv2.drawMarker(img_ref, (x,y), (0,255,0), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

    # for kp in kp_ref_outliers:
    #     x,y = kp.astype(int)
    #     cv2.drawMarker(img_ref, (x,y), (225,0,0), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

    for kp in kp_tgt_inliers:
        x,y = kp.astype(int)
        #cv2.circle(img_tgt, (x,y), 6, (0,225,0), 2, cv2.LINE_AA)
        cv2.drawMarker(img_tgt, (x,y), (0,255,0), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

    # for kp in kp_tgt_outliers:
    #     x,y = kp.astype(int)
    #     cv2.drawMarker(img_tgt, (x,y), (225,0,0), cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
    

    img_ref = img_ref[:370, :]
    img_tgt = img_tgt[150:, :]

    canvas = np.vstack([img_ref, img_tgt])

    kp_ref = kp_ref[idx_inliers[:,0]]
    kp_tgt = kp_tgt[idx_inliers[:,1]]

    offset = img_ref.shape[0]
    
    for m in range(len(kp_ref)):
        cv2.line(canvas, kp_ref[m], (kp_tgt[m][0], kp_tgt[m][1]+offset - 150), (0,225,0), 1, cv2.LINE_AA)

    plt.figure(figsize=(12,12))
    plt.imshow(img_ref), plt.show()
    plt.figure(figsize=(12,12))
    plt.imshow(img_tgt), plt.show()

    plt.figure(figsize=(12,12))
    plt.imshow(canvas), plt.show()
    



def filter_matches_adalam(kp1, kp2, dist_mat, shape_im1, shape_im2, top_k = 200):

    print('Keypoints:', min(len(kp1), len(kp2)))

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.empty_cache()

    k1 = torch.tensor([k.pt for k in kp1], device = dev)
    k2 = torch.tensor([k.pt for k in kp2], device = dev)
    dist_mat = torch.tensor(dist_mat, device = dev)

    dd12, nn12 = torch.topk(dist_mat, k=2, dim=1, largest=False)  # (n1, 2)

    idxs = nn12[:, 0]
    scores = dd12[:, 0] / dd12[:, 1].clamp_min_(1e-3)

    sort_idx = torch.argsort(scores)
    idxs = idxs[sort_idx][:top_k]
    scores = scores[sort_idx][:top_k]
    k1 = k1[sort_idx][:top_k]

    # scores = scores.cpu().numpy()
    # idxs = idxs.cpu().numpy()
    # idx_s = np.arange(len(idxs))[scores < 0.9]
    # idx_t = idxs[scores < 0.9]
    # idxs = np.array([idx_s, idx_t]).T

    idxs = adalam_matcher.filter_matches(
               k1, k2,
               idxs, scores ).cpu().numpy()

    k1 = k1.cpu().numpy()[idxs[:,0]]
    k2 = k2.cpu().numpy()[idxs[:,1]]

    print(len(k1), len(k2))    
    return k1, k2, idxs


def cv_kps_from_csv(csv):
	keypoints = []
	for line in csv:
		k = cv2.KeyPoint(line['x'], line['y'], line['size']*1., line['angle'])
		keypoints.append(k)

	return keypoints


def compute_2d_error(coords1, coords2, im_shape1, im_shape2, tps_file):
    factor1 = np.array([im_shape1[1], im_shape1[0]], dtype= np.float32)
    factor2 = np.array([im_shape2[1], im_shape2[0]], dtype= np.float32)
    coords1_norm = coords1 / factor1
    coords2_norm = coords2 / factor2
    warped = compute_gt_warp(coords1_norm, coords2_norm, tps_file) * factor1

    error =  np.linalg.norm( coords1 - warped, axis = 1)
    print(error.mean())
    return error

def compute_gt_warp(tgt_coords, file_path):
    device = torch.device('cpu')
    theta_np = np.load(file_path + '_theta.npy').astype(np.float32)
    ctrl_pts = np.load(file_path + '_ctrlpts.npy').astype(np.float32)
    tgt_coords = tgt_coords.astype(np.float32)

    theta = torch.tensor(theta_np, device= device)
    warped_coords = tps_torch.tps_sparse(theta, torch.tensor(ctrl_pts, device=device), torch.tensor(tgt_coords, 
                                                                    device=device)).squeeze(0).cpu().numpy()
    return warped_coords


def read_matches(filepath):
	csv = np.recfromcsv(filepath + '.match', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
	return csv['idx_ref'], csv['idx_tgt']

def readDistMat(abs_filename):
    '''
        Read distance matrix 
    '''
    
    filename, _ = os.path.splitext(os.path.basename(abs_filename))
    name_src, name_tgt, name_desc = re.split('__',filename)
    
    with open(abs_filename) as f:
        #print('Loading ', abs_filename)
        n_src,n_tgt = list(map(int,f.readline().rstrip('\n').split(' '))) #read length of keypoints 1 and 2		
        lin_matrix = list(map(float,f.readline().rstrip('\n').split(' ')[:-1]))
        
        dist_mat = np.array(lin_matrix, dtype=np.float32)
        dist_mat = dist_mat.reshape(n_src, n_tgt)
        dist_mat[np.where(np.isnan(dist_mat))] = -1
            
    return dist_mat


def plot_meshes(xyz, ijk, xyz2, ijk2, save_name = ''):
    marker_opts = dict( size=2.5,
                        #color='green',          # set color to an array/list of desired values
                        colorscale='Viridis',   # choose a colorscale
                        opacity=0.8,
                        color=xyz[:,2],
                    )


    tri_points = xyz[ijk]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])

    #define the trace for triangle sides
    lines1 = go.Scatter3d(
                    x=Xe,
                    y=Ye,
                    z=Ze,
                    mode='lines',
                    name='',
                    line=dict(color= 'rgb(0,225,0)', width=1.5))  



    tri_points = xyz2[ijk2]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])

    #define the trace for triangle sides
    lines2 = go.Scatter3d(
                    x=Xe,
                    y=Ye,
                    z=Ze,
                    mode='lines',
                    name='',
                    line=dict(color= 'rgb(225,0,0)', width=1.5))  


    mesh1 = go.Mesh3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], i=ijk[:,0], j=ijk[:,1], k=ijk[:,2], color='green', opacity=0.5, name = "Target", showlegend=True if "REF" in save_name else False)
    mesh2 = go.Mesh3d(x=xyz2[:,0], y=xyz2[:,1], z=xyz2[:,2], i=ijk2[:,0], j=ijk2[:,1], k=ijk2[:,2], color='red', opacity=0.5, name = "Reference", showlegend=True if "REF" in save_name else False)
    
    fig = go.Figure(data=[mesh1, mesh2])
                        #lines1, lines2])

    #fig.add_scatter()

    #fig = go.FigureWidget()
    #fig = go.Figure(data=[go.Mesh3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2]])
    fig.update_layout(
        margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
                    ),
        scene_camera = dict(
                            eye=dict(x=0., y=0., z=-2),
                            up=dict(x=0., y=-1., z=0)
                            ),
        scene_dragmode='orbit',
        scene = dict(
            # aspectratio=dict(x=1, y=1, z=1), # <---- tried this too
            aspectmode='data',
            xaxis= {
            #'range': [0.2, 1],
            'showgrid': False, # thin lines in the background
            'zeroline': False, # thick line at x=0
            'visible': False,  # numbers below
            },
            yaxis= {
            #'range': [0.2, 1],
            'showgrid': False, # thin lines in the background
            'zeroline': False, # thick line at x=0
            'visible': False,  # numbers below
            },
            zaxis= {
            #'range': [0.2, 1],
            'showgrid': False, # thin lines in the background
            'zeroline': False, # thick line at x=0
            'visible': False,  # numbers below
            }

        ),
        #template='plotly_dark',
        autosize=False,
        width=640,
        height=480,

    )
    #fig.show()

    out_dir = '/tmp/' + save_name
    
    if save_name!= '':
        os.makedirs(out_dir, exist_ok = True)
        for i in range(90,360+90,5):
            fig.update_layout(scene_camera = dict(
                                up=dict(x=0., y=-1, z=0),
                                eye=dict(x = 2.4* np.cos(i * np.pi / 180.), y=0, z=2.4* np.sin(i * np.pi / 180.))
                                ))
            fig.write_image(out_dir + '/test%05d.png'%(i))
    else:
        fig.show()
        #fig.show()


    #fig.write_html("aligned.html")



def plot_mesh(xyz, ijk):
    marker_opts = dict( size=2.5,
                        #color='green',          # set color to an array/list of desired values
                        colorscale='Viridis',   # choose a colorscale
                        opacity=0.8,
                        color=xyz[:,2]
                    )

    fig = go.Figure(data=[go.Mesh3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], i=ijk[:,0], j=ijk[:,1], k=ijk[:,2], color='green', opacity=0.6)])

    fig.update_layout(
        scene = dict(
            # aspectratio=dict(x=1, y=1, z=1), # <---- tried this too
            aspectmode='data'
        ),
        #template='plotly_dark',
        autosize=False,
        width=800,
        height=600,
    )
    fig.show()

def plot_pcd(XYZ):
    x, y, z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                        mode='markers',
                                           marker=dict(
                                                    size=1.0,
                                                    color=z,                # set color to an array/list of desired values
                                                    colorscale='Viridis',   # choose a colorscale
                                                    opacity=0.8
                                                ))])
    fig.update_layout(
        scene = dict(
            aspectmode='data'
        ),
        template='plotly_dark',
        autosize=False,
        width=800,
        height=600,
    )
    fig.show()



###################### NON-RIGID RANSAC ####################################

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_corresp_ransac(src_kps, tgt_kps, dist_mat):
    src_kps = torch.tensor([k.pt for k in src_kps], device = dev)
    tgt_kps = torch.tensor([k.pt for k in tgt_kps], device = dev)
    with torch.no_grad():
        dist_mat = torch.tensor(dist_mat, device = dev)
        dd12, nn12 = torch.topk(dist_mat, k=1, dim=1, largest=False)  # (n1, 2)
        idxs = nn12[:, 0]

    tgt_kps = tgt_kps[idxs]
    inliers = nr_RANSAC(src_kps, tgt_kps, device = dev)
    print(inliers.sum())

    return src_kps[inliers].cpu().numpy(), tgt_kps[inliers].cpu().numpy()


def normalize(pts):
    pts = pts - pts.mean()
    norm_avg = (pts**2).sum(axis=1).sqrt().mean()
    pts = pts / norm_avg * np.sqrt(2.)
    return pts

def random_choice(max_idx, batch, dev):
    return torch.randint(max_idx, (batch, 3), device = dev)

def nr_RANSAC(ref_pts, tgt_pts, device, batch = 8_000, thr = 0.05):

    with torch.no_grad():
        ref_pts = ref_pts.to(device)
        tgt_pts = tgt_pts.to(device)

        ref_pts = normalize(ref_pts)
        tgt_pts = normalize(tgt_pts)
        pts = torch.cat((ref_pts, tgt_pts), axis=1)
        choices = random_choice(len(pts), batch, dev = dev)
        batched_pts = pts[choices]
        batched_pts = batched_pts.permute(0,2,1)
        mean_vec = batched_pts.mean(axis=2)

        batched_pts = batched_pts - mean_vec.view(-1,4,1)

        U, S, Vh = torch.linalg.svd(batched_pts)
        A = U[:, :, :2]

        #check if hypothesis is not ill-conditioned (has 2 sing vals > eps)
        good_mask = S[:, 1] > 1e-3

        pts_expanded = pts.expand(batch,-1,-1).permute(0,2,1)
        M = torch.bmm(A, A.permute(0,2,1))
        # print(mean_vec.shape)
        # print(pts_expanded.shape)
        # print(torch.bmm(A, A.permute(0,2,1)).shape)
        # print((pts_expanded - mean_vec.view(-1,4,1)).shape)
        residuals = pts_expanded - torch.bmm( torch.bmm(A, A.permute(0,2,1)) , (pts_expanded - mean_vec.view(-1,4,1)) )  - mean_vec.view(-1,4,1)
        residuals = torch.linalg.norm(residuals, dim=1)


        inliers = residuals < thr
        inliers = inliers[good_mask]
        count = inliers.sum(dim=1)
        best = count.argmax()

    # print(residuals.shape)
    # print(count)
    # print(count.max())
    # print(inliers.shape)

    return inliers[best].cpu().numpy()


def fit_tps(source_kp, target_kp):
    # source_kp, target_kp: Nx2 arrays
    x_src, y_src = source_kp[:, 0], source_kp[:, 1]
    x_tgt, y_tgt = target_kp[:, 0], target_kp[:, 1]

    # RBF interpoladores com função TPS
    fx = Rbf(x_src, y_src, x_tgt, function='thin_plate')
    fy = Rbf(x_src, y_src, y_tgt, function='thin_plate')
    return fx, fy

def apply_tps_to_grid(fx, fy, shape):
    h, w = shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    # Aplica TPS
    warped_x = fx(grid_x_flat, grid_y_flat)
    warped_y = fy(grid_x_flat, grid_y_flat)

    # Reshape para formar um campo vetorial (GT)
    flow_map = np.stack([warped_x.reshape(h, w), warped_y.reshape(h, w)], axis=-1)
    return flow_map
