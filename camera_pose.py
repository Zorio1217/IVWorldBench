import numpy as np
import argparse

def compute_camera_pose_error(predicted_pose, gt_pose):
    """
    计算相机位姿误差，包括位置误差和旋转误差。
    
    参数:
    predicted_pose: 4x4 numpy array，预测的相机位姿矩阵
    gt_pose: 4x4 numpy array，真实的相机位姿矩阵
    
    返回:
    position_error: float，位置误差（单位：米）
    rotation_error: float，旋转误差（单位：度）
    """
    # 提取位置和旋转部分
    predicted_position = predicted_pose[:3, 3]
    gt_position = gt_pose[:3, 3]
    
    predicted_rotation = predicted_pose[:3, :3]
    gt_rotation = gt_pose[:3, :3]
    
    # 计算位置误差
    position_error = np.linalg.norm(predicted_position - gt_position)
    
    # 计算旋转误差
    rotation_diff = np.dot(predicted_rotation.T, gt_rotation)
    rotation_error = np.arccos((np.trace(rotation_diff) - 1) / 2) * (180.0 / np.pi)
    return np.sqrt(position_error * rotation_error)

parser = argparse.ArgumentParser(description='Evaluate camera pose estimation performance of a model.')
parser.add_argument('--gt_pose_path', type=str, required=True, help='Path to the ground truth camera pose file (numpy format).')
parser.add_argument('--predicted_pose_path', type=str, required=True, help='Path to the predicted camera pose file (numpy format).')
args = parser.parse_args()

gt_pose = np.load(args.gt_pose_path)  # 加载真实的相机位姿矩阵
predicted_pose = np.load(args.predicted_pose_path)  # 加载预测的相机位姿矩阵
error = compute_camera_pose_error(predicted_pose, gt_pose)
print(f"Camera Pose Error: {error:.4f}")
