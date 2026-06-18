import open3d as o3d
import numpy as np

print("Open3D版本:", o3d.__version__)
# 测试点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0,0,0], [1,1,1], [2,2,2]])

# 👉 修改：体素大小设为2.0
downpcd = pcd.voxel_down_sample(voxel_size=2.0)

print("降采样后点数:", len(downpcd.points))  # 现在会输出 1！
