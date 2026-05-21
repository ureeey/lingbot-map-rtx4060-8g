import open3d as o3d
import sys

pcd = o3d.io.read_point_cloud(sys.argv[1])
print(f"Points: {len(pcd.points)}")
print(f"Colors: {pcd.has_colors()}")
o3d.visualization.draw_geometries([pcd])