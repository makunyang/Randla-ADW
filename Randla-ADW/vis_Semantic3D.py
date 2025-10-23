from helper_tool import Plot
from os.path import join, exists
from helper_ply import read_ply
import numpy as np
import os
 
 
if __name__ == '__main__':
    path = '/home/ma/Desktop/TTT/randla-net-tf2-main/data'
    label_folder = '/home/ma/Desktop/TTT/randla-net-tf2-main/test/Log_2025-07-01_01-16-19/predictions/'
 
    label_to_names = {0: 'unlabeled',
                               1: 'ground',
                               2: 'vegetation',
                               3: 'building',
                               4: 'water',
                               5: 'car',
                               6: 'boat'}
 
 
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
            '19842688.ply': '19842688.labels',
            '19842689.ply': '19842689.labels',
            '19842690.ply': '19842690.labels',
            '19842691.ply': '19842691.labels',
            '19842692.ply': '19842692.labels',
            '19842693.ply': '19842693.labels',
            '19842694.ply': '19842694.labels',
            '19842695.ply': '19842695.labels',
            '19852688.ply': '19852688.labels',
            '19852689.ply': '19852689.labels',
            '19852690.ply': '19852690.labels',
            '19852691.ply': '19852691.labels',
            '19852692.ply': '19852692.labels',
            '19852693.ply': '19852693.labels',
            '19852694.ply': '19852694.labels',
            '19852695.ply': '19852695.labels',
            '19862688.ply': '19862688.labels',
            '19862689.ply': '19862689.labels',
            '19862690.ply': '19862690.labels',
            '19862691.ply': '19862691.labels',
            '19862692.ply': '19862692.labels',
            '19862693.ply': '19862693.labels',
            '19862694.ply': '19862694.labels',
            '19862695.ply': '19862695.labels',
            '19872689.ply': '19872689.labels',
            '19872690.ply': '19872690.labels',
            '19872691.ply': '19872691.labels',
            '19872692.ply': '19872692.labels',
            '19872693.ply': '19872693.labels',
            '19872694.ply': '19872694.labels',
            '19872695.ply': '19872695.labels',
            '19882688.ply': '19882688.labels',
            '19882689.ply': '19882689.labels',
            '19882690.ply': '19882690.labels',
            '19882691.ply': '19882691.labels'}
    
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
        Plot.draw_pc_sem_ins(pc_xyz, pc_sem_ins,plot_colors)
