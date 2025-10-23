import sys
import open3d as o3d

base="/home/ma/Desktop/TTT/randla-net-tf2-main/data/semantic3d/original_ply/bildstein_station1_xyz_intensity_rgb.ply"
pcd = o3d.io.read_point_cloud(base)
o3d.visualization.draw_geometries([ply])


# pcd = o3d.io.read_point_cloud(sys.argv[1])
# o3d.visualization.draw_geometries([pcd])

