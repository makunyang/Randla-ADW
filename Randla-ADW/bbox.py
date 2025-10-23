from helper_tool import Plot
from os.path import join, exists
from helper_ply import read_ply
import numpy as np
import os
from sklearn.cluster import DBSCAN


def instance_segmentation(pc_xyz, pc_sem_ins):
    instance_labels = []
    for class_id in np.unique(pc_sem_ins):
        class_points = pc_xyz[pc_sem_ins == class_id]
        if len(class_points) > 0:
            # 使用DBSCAN进行聚类
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(class_points)
            instance_ids = clustering.labels_
            instance_labels.extend(instance_ids + np.max(instance_labels) + 1 if len(instance_labels) > 0 else instance_ids)
        else:
            instance_labels.extend([])
    instance_labels = np.array(instance_labels)
    return instance_labels


def compute_bounding_boxes(pc_xyz, instance_labels):
    bounding_boxes = []
    for instance_id in np.unique(instance_labels):
        if instance_id != -1:
            instance_points = pc_xyz[instance_labels == instance_id]
            min_coords = np.min(instance_points, axis=0)
            max_coords = np.max(instance_points, axis=0)
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            bounding_boxes.append((min_coords, max_coords, center, size))
    return bounding_boxes


def visualize_bounding_boxes(Plot, pc_xyz, bounding_boxes):
    # 绘制点云
    Plot.draw_pc(pc_xyz)
    # 绘制三维目标框
    for min_coords, max_coords, center, size in bounding_boxes:
        Plot.draw_bounding_box(center, size)


if __name__ == '__main__':
    path = '/home/ma/Desktop/TTT/randla-net-tf2-main/data'
    label_folder = '/home/ma/Desktop/TTT/randla-net-tf2-main/test/Log_2025-06-01_03-36-39/predictions/'

    label_to_names = {0: 'unlabeled',
                      1: 'man-made terrain',
                      2: 'natural terrain',
                      3: 'high vegetation',
                      4: 'low vegetation',
                      5: 'buildings',
                      6: 'hard scape',
                      7: 'scanning artefacts',
                      8: 'cars'}

    original_folder = '/home/ma/Desktop/TTT/randla-net-tf2-main/data/semantic3d/original_data'
    full_pc_folder = '/home/ma/Desktop/TTT/randla-net-tf2-main/data/semantic3d/original_ply'

    test_files_names = []
    cloud_names = [file_name[:-4] for file_name in os.listdir(original_folder) if file_name[-4:] == '.txt']
    for pc_name in cloud_names:
        if not exists(join(original_folder, pc_name + '.labels')):
            test_files_names.append(pc_name + '.ply')
    test_files_names = np.sort(test_files_names)
    # Ascii files dict for testing
    ascii_files = {
        'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
        'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
        'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
        'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
        'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
        'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
        'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
        'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
        'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
        'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
        'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
        'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
        'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
        'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
        'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
        'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
        'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
        'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
        'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

    plot_colors = Plot.random_colors(11, seed=2)
    for file in test_files_names:
        print(file)
        test_files = join(full_pc_folder, file)
        label_files = join(label_folder, ascii_files[file])
        data = read_ply(test_files)
        # 绘制原图
        pc_xyzrgb = np.vstack((data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'])).T
        Plot.draw_pc(pc_xyzrgb)
        # 绘制预测结果图
        pc_xyz = np.vstack((data['x'], data['y'], data['z'])).T
        pc_sem_ins = np.loadtxt(label_files)
        pc_sem_ins = pc_sem_ins.astype(int)
        Plot.draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors)

        # 实例分割
        instance_labels = instance_segmentation(pc_xyz, pc_sem_ins)

        # 计算三维目标框
        bounding_boxes = compute_bounding_boxes(pc_xyz, instance_labels)

        # 可视化三维目标框
        visualize_bounding_boxes(Plot, pc_xyz, bounding_boxes)
