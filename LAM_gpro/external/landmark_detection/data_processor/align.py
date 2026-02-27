import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes

from open3d.pipelines.registration import registration_ransac_based_on_correspondence


def rigid_transform_3D(A, B):
    assert A.shape == B.shape, "Input arrays must have the same shape"
    assert A.shape[1] == 3, "Input arrays must be Nx3"
    
    N = A.shape[0]  # Number of points

    # Compute centroids of A and B
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points around the centroids
    AA = A - centroid_A
    BB = B - centroid_B

    # H = AA^T * BB
    H = np.dot(AA.T, BB)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation (det(R) should be +1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Compute translation
    t = centroid_B - np.dot(R, centroid_A)

    # Construct the transform matrix (4x4)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    return transform_matrix


def compute_rigid_transform(points1, points2):
    """
    计算从points1到points2的刚体变换（包括尺度、旋转和平移）。
    
    参数:
    points1, points2: np.ndarray, 形状为(68, 3)的数组，分别为两组3D对应点。
    
    返回:
    scale: float, 尺度因子
    R: np.ndarray, 3x3的旋转矩阵
    t: np.ndarray, 3维的平移向量
    """
    # 中心化
    mean1 = np.mean(points1, axis=0)
    centered_points1 = points1 - mean1
    mean2 = np.mean(points2, axis=0)
    centered_points2 = points2 - mean2
    
    # 使用orthogonal_procrustes计算旋转和平移
    R, _ = orthogonal_procrustes(centered_points1, centered_points2)
    t = mean2 - R @ mean1  # 计算平移向量
    
    # 计算尺度因子
    scale = np.mean(np.linalg.norm(centered_points2, axis=1) / 
                    np.linalg.norm(centered_points1, axis=1))
    
    return scale, R, t


def compute_rigid_transform_new(points_A, points_B):
    # 中心化
    center_A = np.mean(points_A, axis=0)
    center_B = np.mean(points_B, axis=0)
    points_A_centered = points_A - center_A
    points_B_centered = points_B - center_B
    
    # 计算协方差矩阵
    cov_matrix = np.dot(points_A_centered.T, points_B_centered)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    # 确保旋转矩阵为正交且右手系，这里我们取Vt的转置作为旋转矩阵
    rotation_matrix = np.dot(Vt.T, U.T)
    
    # 检查行列式是否为-1（表示反射，不满足旋转矩阵要求），如果是，则调整一个列的符号
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2,:] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)
    
    # 计算尺度因子
    scale = np.trace(np.dot(points_A_centered.T, points_B_centered)) / np.trace(np.dot(points_A_centered.T, points_A_centered))
    
    # 计算平移向量
    translation_vector = center_B - scale * np.dot(rotation_matrix, center_A)
    
    return scale, rotation_matrix, translation_vector




# 示范用法
obj_A = '/home/gyalex/Desktop/our_face.obj'
obj_B = '/home/gyalex/Desktop/Neutral.obj'

mesh_A = o3d.io.read_triangle_mesh(obj_A)
mesh_B = o3d.io.read_triangle_mesh(obj_B)

vertices_A = np.asarray(mesh_A.vertices)
vertices_B = np.asarray(mesh_B.vertices)

list_A = list()
list_B = list()
with open('/home/gyalex/Desktop/our_marker.txt', 'r') as f:
    lines_A = f.readlines()
    for line in lines_A:
        hh = line.strip().split()
        list_A.append(int(hh[0]))

with open('/home/gyalex/Desktop/ARKit_landmarks.txt', 'r') as f:
    lines_B = f.readlines()
    for line in lines_B:
        hh = line.strip().split()
        list_B.append(int(hh[0]))

A = vertices_A[list_A,:]  # 第一组3D点
B = vertices_B[list_B,:]  # 第二组3D点

# scale, R, t = compute_rigid_transform(A, B)

# # 定义尺度变换矩阵
# scale_matrix = np.eye(4)
# scale_matrix[0, 0] = scale  # x轴方向放大2倍
# scale_matrix[1, 1] = scale  # y轴方向放大2倍
# scale_matrix[2, 2] = scale  # z轴方向放大2倍

# transform_matrix = np.eye(4)
# transform_matrix[:3, :3] = scale
# transform_matrix[:3, 3] = R*t

# mesh_A.transform(transform_matrix)
# # mesh_A.transform(scale_matrix)

# o3d.io.write_triangle_mesh('/home/gyalex/Desktop/our_face_new.obj', mesh_A)

pcd_source = o3d.utility.Vector3dVector(A)  # 示例源点云数据
pcd_target = o3d.utility.Vector3dVector(B)  # 示例目标点云数据 + 1偏移，仅作示例

corres_source = list()
for idx in range(68): corres_source.append(idx) 
corres_target = list()
for idx in range(68): corres_target.append(idx) 

# 根据对应点索引获取实际的对应点坐标
corres_source_points = pcd_source
corres_target_points = pcd_target

corres = o3d.utility.Vector2iVector([[src, tgt] for src, tgt in zip(corres_source, corres_target)])

# 应用RANSAC进行基于对应点的配准
reg_result = registration_ransac_based_on_correspondence(
    pcd_source,
    pcd_target,
    corres,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    ransac_n=3,
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, epsilon=1e-6)
)

# # 使用RANSAC进行配准
# convergence_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=50000, max_validation=500)
# ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
#     pcd_source,
#     pcd_target,
#     corres,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     3,  # RANSAC阈值，根据实际情况调整
#     convergence_criteria,
#     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
#     o3d.pipelines.registration.RANSACLoss())

# 应用变换到源mesh
# mesh_source_aligned = mesh_source.transform(reg_result.transformation)

a = 0