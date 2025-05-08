
from . import utils

import numpy as np
import cv2
import open3d as o3d
import time
from scipy.spatial import KDTree
import pdb
import matplotlib.pyplot as plt


class ARAP:
    '''
        Class that wraps ARAP method from Open3D.
        Given a depth image pair and intrinsics, it computes the mesh
        (possibly with invalid vals at Z = 0), then, given the correspondences,
        it computes the ARAP transformation from source to target.
        
    '''

    def __init__(self):
        self.model = o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed

    def proj_2d3d(self, K, D):
        '''
            input: K, intrinsics ndarray(3,3)
                D, depth ndarray(H, W)
            returns: ndarray (H, W, 3) structured pointcloud
        '''
        H, W = D.shape
        xg, yg = np.meshgrid(np.arange(W), np.arange(H))
        xy1 = np.dstack([xg, yg, np.ones((H, W))]).reshape(-1, 3)
        xyz = np.linalg.inv(K) @ xy1.T
        xyz = xyz.T.reshape(H, W, 3)

        #project pts to 3D
        D = D.astype(np.float64)[:, :, np.newaxis]
        xyz *= D

        return xyz

    def pyrDown(self, img, iters):
        img = img.astype(np.float32)
        for i in range(iters):
            img = cv2.bilateralFilter(img, -1, 100, 3)
            img = cv2.resize(img, None, fx=0.5, fy = 0.5, interpolation = cv2.INTER_NEAREST)

        return img


    def gen_trimesh_idx(self, H, W):
        '''
        @Brief:
        Generates a triangle mesh configured as shown below:
        (x,y)   *--* (x+1,y)
                | /|
                |/ |
        (x,y+1) *--* (x+1,y+1)
        '''
        X, Y = np.meshgrid(np.arange(W-1), np.arange(H-1))
        X1 = X + 1
        Y1 = Y + 1

        UpperT = np.dstack((X,Y,X1,Y,X,Y1))
        LowerT = np.dstack((X,Y1,X1,Y,X1,Y1))

        UpperT = UpperT.reshape(-1,6).reshape(-1,3,2)
        LowerT = LowerT.reshape(-1,6).reshape(-1,3,2)
        return np.vstack((UpperT,LowerT)) # stack upper and lower triangles in a single array


    def build_trimesh_from_xyz(self, pts3d):
        '''
            build a triangular mesh from a structured point cloud, while filtering
            invalid values (where Z is zero) from facets and vertices
            Args:
                pts3d: ndarray(H, W, 3) points in mm
            Return:
                xyz: ndarray(N, 3) - linearized pointcloud 
                triangles: ndarray(N, 3) - indices of triangular mesh 
        '''
        t0 = time.time()
        pts3d = pts3d.copy()
        h, w = pts3d.shape[:2] 

        #filter too close pts from camera to avoid numeric errors
        pts3d[ pts3d[:, :, 2] < 1] = 0

        #pts3d[100:120, 100:120, 2] = 0
        xyz1 = pts3d.reshape(-1,3)

        triangles = self.gen_trimesh_idx(h, w)
        triangles = triangles[:, :, 1] * w + triangles[:, :, 0]

        valid_mask_tri = (xyz1[triangles][:, :, 2] > 0).all(axis=1)
        triangles1 = triangles[valid_mask_tri] #select valid triangles
        touched_v = np.zeros(len(xyz1), dtype=bool)
        #touch all vertices using triangles
        touched_v[triangles1] = True
        #rm untouched vertices
        xyz1[~touched_v] = 0

        #compute offsets from invalid values for re-indexation
        xyz_idx = np.arange(len(xyz1))
        valid_mask_vert = xyz1[:,2] > 0
        xyz1 = xyz1[valid_mask_vert] #select valid vertices
        new_idx = xyz_idx[valid_mask_vert]
        xyz_idx[:] = -1
        xyz_idx[new_idx] = np.arange(len(new_idx))
        triangles1 = xyz_idx[triangles1] #re-index triangles 

        #print('done in %.3f'%(time.time() - t0))
        #plot_mesh(xyz1, triangles1)
        return xyz1, triangles1, xyz_idx

    #build_trimesh_from_xyz(pts3d)


    def perspective_project(self, pts3d, K):
        pts2d = pts3d / pts3d[:,2:]
        pts2d = (K @ pts2d.T).T
        return pts2d[:, :2]


    def blob_filter(self, depth):
        bmask = (depth > 1).astype(np.uint8)
        contours, _= cv2.findContours(bmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        largest_contour = np.array([len(c) for c in contours]).argmax()
        mask = np.zeros_like(bmask)
        cv2.fillPoly(mask, pts=[contours[largest_contour]],color=(255))
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)

        #plt.figure(figsize=(16,16))
        #plt.imshow(mask, cmap='gray'), plt.show()
        depth[~mask.astype(bool)] = 0


    def register_pair(self, kps_src, kps_tgt, D_ref, D_tgt, K, samples,warp_func,
                        mask_ref = None, mask_tgt = None, npyr = 2,
                        thr_2d = [2.0, 3.0, 5.0], 
                        thr_3d = [5., 10., 15.],
                        plot_ref = False,
                        save_name = 'test'):
        '''
            Register 2 pointclouds using the as rigid as possible deformation method
            Args:
                kps_src: ndarray(N, 2) 2D source points (x,y)
                kps_tgt: ndarray(N, 2) 2D target points (x,y)
                D_ref: ndarray(H, W) depth image in mm
                D_tgt: ndarray(H, W) depth image in mm
                mask_ref: ndarray(H, W) boolean reference mask to filter invalid depth
                mask_tgt: ndarray(H, W) boolean tgt mask to filter invalid depth
                K: ndarray(3,3), intrinsics of depth camera
                npyr: int, number of pyramid levels for smoothing depth and reducing pcl
                thr_2d: 2D threshold in pixels for accuracy computation
                thr_3d: 3D threshold in mm for accuracy computation

            Return:
                2d and 3d accuracy of registration
        '''

        t0 = time.time()
        scale_factor = 1/(2.**npyr)
        K = K.copy()
        K[:2, :] *= scale_factor

        src_sample,tgt_sample = samples

        shape_im1 = D_ref.shape[:2]
        shape_im2 = D_tgt.shape[:2]

        if mask_ref is not None:
            D_ref[~mask_ref.astype(bool)] = 0
        if mask_tgt is not None:
            D_tgt[~mask_tgt.astype(bool)] = 0 

        kps_src=(kps_src*scale_factor+0.5).astype(np.int32)
        kps_tgt=(kps_tgt*scale_factor+0.5).astype(np.int32)

        D_ref = self.pyrDown(D_ref, npyr) 
        D_tgt = self.pyrDown(D_tgt, npyr)

        self.blob_filter(D_ref)
        self.blob_filter(D_tgt)

        #print("took ", time.time() - t0)
        pts3d_ref = self.proj_2d3d(K, D_ref)
        pts3d_tgt = self.proj_2d3d(K, D_tgt)

        #utils.plot_pcd(pts3d_tgt.reshape(-1,3))

        xyz1, tri1, remap_idx1 = self.build_trimesh_from_xyz(pts3d_ref)
        xyz2, tri2, remap_idx2 = self.build_trimesh_from_xyz(pts3d_tgt)

        #remap match indexes from 2D coord to linear idx and update them
        idx_src = kps_src[:,0] + kps_src[:,1]*D_ref.shape[1]
        idx_tgt = kps_tgt[:,0] + kps_tgt[:,1]*D_tgt.shape[1]
        idx_src = remap_idx1[idx_src]
        idx_tgt = remap_idx2[idx_tgt]
        valid_mask = (idx_src >= 0) & (idx_tgt >= 0)
        idx_src = idx_src[valid_mask]
        idx_tgt = idx_tgt[valid_mask]


        if len(idx_src) < 8 and len(idx_tgt) < 8:
            print(" #######  WARNING: Too few matching points, skipping... ####### \n\n")
            return [0,0,0], [0,0,0]

        #build open3d structure
        mesh1 = o3d.geometry.TriangleMesh()
        mesh1.vertices = o3d.utility.Vector3dVector(xyz1)
        mesh1.triangles = o3d.utility.Vector3iVector(tri1)

        mesh2 = o3d.geometry.TriangleMesh()
        mesh2.vertices = o3d.utility.Vector3dVector(xyz2)
        mesh2.triangles = o3d.utility.Vector3iVector(tri2)

        #constraint_ids = np.random.choice(len(xyz1), 300, replace=False).astype(np.int32)
        constraint_ids_t_s = idx_tgt.astype(np.int32)
        constraint_pos_t_s = np.asarray(mesh1.vertices)[idx_src]
        constraint_ids_t_s = o3d.utility.IntVector( constraint_ids_t_s )
        constraint_pos_t_s = o3d.utility.Vector3dVector( constraint_pos_t_s )


        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Warning ) as cm: #Debug
            mesh_prime_t_s = mesh2.deform_as_rigid_as_possible(constraint_ids_t_s,
                                                        constraint_pos_t_s,
                                                        max_iter=25 ,
                                                        energy = self.model,
                                                        smoothed_alpha=0.1
                                                        )
        

        xyz = np.asarray(mesh_prime_t_s.vertices)
        ijk = np.asarray(mesh_prime_t_s.triangles)

        xyz2 = np.asarray(mesh1.vertices)
        ijk2 = np.asarray(mesh1.triangles)

        xyz_o = np.asarray(mesh2.vertices)
        ijk_o = np.asarray(mesh2.triangles)
        
        #print('done in %.2f'%(time.time() - t0))
        if plot_ref:
            utils.plot_meshes(xyz2, ijk2, xyz_o, ijk_o, save_name + '-REF' if save_name != '' else '')
            
        utils.plot_meshes(xyz2, ijk2, xyz, ijk, save_name)
        #input('...')

        #Query 3D points for correctness
        tree = KDTree(xyz2)
        dists_3d, idxs = tree.query(xyz)

        #Query 2D ground-truth TPS for correctness
        undeformed_tgt_2d = self.perspective_project(xyz, K) / scale_factor #Produzido a partir da deformacao de mesh2 -> mesh1 ou tgt -> src
        tgt_2d = self.perspective_project(xyz_o, K) / scale_factor # Sao os pontos do tgt em 2D
        img_shape = np.array([D_tgt.shape[1], D_tgt.shape[0]], dtype= np.float32) / scale_factor
        result = warp_func(tgt_2d,
                           tgt_sample['uv_coords'],
                           src_sample['uv_coords'],
                           tgt_sample['segmentation'],
                           src_sample['segmentation'],
                           300)
        # gt_undeformed = utils.compute_gt_warp(tgt_2d / img_shape, tps_path) * img_shape # As duas imagens tem pontos
        valid_mask = (result['keypoints'][:, 0] != -1)  # Filtra pontos inválidos
        gt_undeformed = result['keypoints']
        dists_2d = np.linalg.norm(undeformed_tgt_2d[valid_mask] - gt_undeformed[valid_mask], axis = 1)

        #print(tgt_2d)
        #pdb.set_trace()

        accuracy_3d = [ (dists_3d < thr).sum() / len(dists_3d) for thr in thr_3d]
        accuracy_2d = [ (dists_2d < thr).sum() / len(dists_2d) for thr in thr_2d]

        return accuracy_2d, accuracy_3d