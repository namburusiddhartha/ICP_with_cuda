import cupoch as o3d
import torch
import numpy as np
import time
import pickle
from torch.utils.dlpack import to_dlpack
torch.set_printoptions(precision = 10)

def to_gpu(pts):
    pts_torch = torch.from_numpy(pts.astype(np.float32)).cuda()
    pack = to_dlpack(pts_torch)
    pcd = o3d.geometry.PointCloud()
    pcd.from_points_dlpack(pack)
    return pcd

estimated_point_clouds, init_poses, object_point_cloud = pickle.load(open('input.pkl', 'rb'))

def icp_refinement_akasha(estimated_point_clouds, init_poses, obj_cloud_gpu):

    est_cloud_arr = [to_gpu(x) for x in estimated_point_clouds]
    kdtree = o3d.geometry.KDTreeFlann(obj_cloud_gpu)
    reg_p2p = o3d.registration.registration_icp_akasha(
                        est_cloud_arr, obj_cloud_gpu, 1.0, kdtree, 28 , init_poses,
                        o3d.registration.TransformationEstimationPointToPoint(), 
                        o3d.registration.ICPConvergenceCriteria(1e-6,1e-6,100))
    return reg_p2p

def icp_refinement(estimated_point_clouds, init_poses, obj_cloud_gpu):
    final_poses = []
    for pose, pc in zip(init_poses, estimated_point_clouds):
        est_cloud_gpu = to_gpu(pc)
        reg_p2p = o3d.registration.registration_icp(
                            est_cloud_gpu, obj_cloud_gpu, 1.0, pose,
                            o3d.registration.TransformationEstimationPointToPoint(), 
                            o3d.registration.ICPConvergenceCriteria(max_iteration=100))
        pred_pose = np.array(reg_p2p.transformation)
        final_poses.append(pred_pose)
    return final_poses

obj_cloud_gpu = to_gpu(object_point_cloud)
t = time.time()
result = icp_refinement(estimated_point_clouds, init_poses, obj_cloud_gpu)
print("Total Time:", time.time() - t)
t = time.time()
result = icp_refinement_akasha(estimated_point_clouds, init_poses, obj_cloud_gpu)
print("Total Time akasha:", time.time() - t)
#result = np.asarray(result)



