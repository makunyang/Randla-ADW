import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import argparse
from pathlib import Path
import sys
import os


class PointCloudInstanceSegmenter:
    def __init__(self, args):
        """初始化实例分割器"""
        self.args = args
        self.points = self._load_pointcloud(args.ply_path)
        self.labels = self._load_labels(args.label_path)
        self.instances = []
        self.label_to_name = self._parse_label_mapping(args.label_map)

        # 创建输出目录（如果不存在）
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

    def _load_pointcloud(self, path):
        """加载.ply点云文件"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"点云文件不存在: {path}")
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points)
            if len(points) == 0:
                raise ValueError(f"点云文件为空: {path}")
            return points
        except Exception as e:
            raise RuntimeError(f"加载点云失败: {e}")

    def _load_labels(self, path):
        """加载.labels文件"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"标签文件不存在: {path}")
        try:
            labels = np.loadtxt(path, dtype=np.int32)
            if len(labels) != len(self.points):
                raise ValueError(f"标签数量({len(labels)})与点云点数({len(self.points)})不匹配")
            return labels
        except Exception as e:
            raise RuntimeError(f"加载标签失败: {e}")

    def _parse_label_mapping(self, label_map_args):
        """解析命令行中的类别映射参数"""
        mapping = {}
        if not label_map_args:
            return mapping

        if len(label_map_args) % 2 != 0:
            print("警告: 类别映射参数数量应为偶数 (ID 名称 对)")
            return mapping

        for i in range(0, len(label_map_args), 2):
            try:
                id = int(label_map_args[i])
                name = label_map_args[i + 1]
                mapping[id] = name
            except:
                print(f"警告: 无法解析类别映射项: {label_map_args[i]} {label_map_args[i+1]}")
        return mapping

    def perform_instance_segmentation(self):
        """执行实例分割"""
        print(f"开始实例分割: {len(np.unique(self.labels))} 个语义类别")
        print(f"DBSCAN 参数: eps={self.args.eps}, min_samples={self.args.min_samples}")

        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            if label == self.args.ignore_label:
                continue

            label_mask = self.labels == label
            label_points = self.points[label_mask]

            if len(label_points) < self.args.min_samples:
                continue

            # 执行DBSCAN聚类
            dbscan = DBSCAN(
                eps=self.args.eps,
                min_samples=self.args.min_samples,
                metric='euclidean',
                n_jobs=self.args.n_jobs
            )
            instance_labels = dbscan.fit_predict(label_points)

            # 收集有效实例
            for instance_id in np.unique(instance_labels):
                if instance_id == -1:  # 跳过噪声点
                    continue

                instance_mask = instance_labels == instance_id
                instance_points = label_points[instance_mask]

                # 过滤极小实例
                if len(instance_points) < self.args.min_instance_points:
                    continue

                self.instances.append({
                    'class_id': label,
                    'points': instance_points,
                    'point_indices': np.where(label_mask)[0][instance_mask]
                })

        print(f"实例分割完成: 共识别出 {len(self.instances)} 个实例")
        return self.instances

    def generate_bounding_boxes(self):
        """生成三维边界框"""
        boxes = []
        for instance in self.instances:
            points = instance['points']

            # 计算轴对齐边界框(AABB)
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            center = (min_coords + max_coords) / 2
            dimensions = max_coords - min_coords

            boxes.append({
                'class_id': instance['class_id'],
                'center': center,
                'dimensions': dimensions,
                'point_indices': instance['point_indices']
            })

        return boxes

    def visualize_results(self):
        """可视化点云和边界框"""
        if not self.instances:
            print("警告: 没有实例可可视化")
            return

        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # 为点云着色（按语义类别）
        colors = np.zeros((len(self.points), 3))
        for instance in self.instances:
            color = self._get_color(instance['class_id'])
            colors[instance['point_indices']] = color
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 创建边界框
        geometries = [pcd]
        for box in self.generate_bounding_boxes():
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=box['center'] - box['dimensions'] / 2,
                max_bound=box['center'] + box['dimensions'] / 2
            )
            aabb.color = self._get_color(box['class_id'])
            geometries.append(aabb)

        # 设置可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="点云实例分割",
            width=self.args.window_width,
            height=self.args.window_height
        )

        # 添加所有几何体
        for geom in geometries:
            vis.add_geometry(geom)

        # 设置相机视角
        ctr = vis.get_view_control()
        ctr.set_front([-0.5, 0.5, -0.5])
        ctr.set_lookat(np.mean(self.points, axis=0))
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)

        # 添加交互选项
        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 1.0

        # 运行可视化
        vis.run()
        vis.destroy_window()

        # 保存结果（如果指定）
        if self.args.output_ply:
            output_path = Path(self.args.output_ply)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            o3d.io.write_point_cloud(str(output_path), pcd)
            print(f"可视化结果已保存到: {output_path}")

    def save_results_to_txt(self):
        """保存结果到TXT文件"""
        if not self.instances:
            print("警告: 没有实例可保存")
            return

        output_path = Path(self.args.output_txt)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        with open(output_path, 'w') as f:
            # 写入表头
            f.write("class_id,class_name,center_x,center_y,center_z,length,width,height,point_count\n")

            # 写入每个实例
            for box in self.generate_bounding_boxes():
                class_name = self.label_to_name.get(box['class_id'], f"class_{box['class_id']}")
                f.write(f"{box['class_id']},{class_name},"
                        f"{box['center'][0]:.6f},{box['center'][1]:.6f},{box['center'][2]:.6f},"
                        f"{box['dimensions'][0]:.6f},{box['dimensions'][1]:.6f},{box['dimensions'][2]:.6f},"
                        f"{len(box['point_indices'])}\n")

        print(f"实例结果已保存到: {output_path}")

    def _get_color(self, class_id):
        """获取类别的颜色"""
        # 预定义颜色映射（可根据需要扩展）
        color_map = {
            0: [1.0, 0.0, 0.0],  # 红色
            1: [0.0, 1.0, 0.0],  # 绿色
            2: [0.0, 0.0, 1.0],  # 蓝色
            3: [1.0, 1.0, 0.0],  # 黄色
            4: [1.0, 0.0, 1.0],  # 紫色
            5: [0.0, 1.0, 1.0],  # 青色
            6: [1.0, 0.5, 0.0],  # 橙色
            7: [0.5, 0.0, 1.0],  # 深紫色
            8: [0.0, 0.5, 1.0],  # 天蓝色
            9: [0.5, 1.0, 0.0],  # 黄绿色
            -1: [0.5, 0.5, 0.5]  # 灰色（默认）
        }

        # 对于超出预定义范围的类别，生成随机但可重复的颜色
        if class_id not in color_map:
            np.random.seed(class_id)  # 使用类别ID作为随机种子，确保颜色一致
            color_map[class_id] = np.random.rand(3).tolist()

        return color_map[class_id]


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于RandLA-Net语义分割结果的点云实例分割与三维框生成工具"
    )

    # 输入参数
    input_group = parser.add_argument_group("输入参数")
    input_group.add_argument(
        "-p", "--ply_path", required=True, type=str,
        help="输入点云文件路径 (.ply)"
    )
    input_group.add_argument(
        "-l", "--label_path", required=True, type=str,
        help="语义标签文件路径 (.labels)"
    )
    input_group.add_argument(
        "--label_map", nargs="+", default=[],
        help="类别ID到名称的映射 (例如: --label_map 0 Road 1 Building)"
    )

    # 输出参数
    output_group = parser.add_argument_group("输出参数")
    output_group.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录 (所有输出文件将保存在此目录下)"
    )
    output_group.add_argument(
        "--output_txt", type=str, default="instances.txt",
        help="输出实例信息的TXT文件路径"
    )
    output_group.add_argument(
        "--output_ply", type=str, default=None,
        help="输出带边界框的点云PLY文件路径"
    )

    # 实例分割参数
    seg_group = parser.add_argument_group("实例分割参数")
    seg_group.add_argument(
        "--eps", type=float, default=0.8,
        help="DBSCAN聚类的邻域半径 (米)"
    )
    seg_group.add_argument(
        "--min_samples", type=int, default=15,
        help="DBSCAN形成核心点所需的最小样本数"
    )
    seg_group.add_argument(
        "--min_instance_points", type=int, default=10,
        help="过滤小于此点数的实例"
    )
    seg_group.add_argument(
        "--ignore_label", type=int, default=-1,
        help="要忽略的标签ID"
    )
    seg_group.add_argument(
        "--n_jobs", type=int, default=1,
        help="DBSCAN并行处理的作业数 (-1表示使用所有CPU)"
    )

    # 可视化参数
    vis_group = parser.add_argument_group("可视化参数")
    vis_group.add_argument(
        "-v", "--visualize", action="store_true",
        help="显示可视化结果"
    )
    vis_group.add_argument(
        "--window_width", type=int, default=1280,
        help="可视化窗口宽度"
    )
    vis_group.add_argument(
        "--window_height", type=int, default=720,
        help="可视化窗口高度"
    )

    # 其他参数
    parser.add_argument(
        "--verbose", action="store_true",
        help="显示详细处理信息"
    )

    return parser.parse_args()


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 处理输出路径（如果指定了输出目录）
        if args.output_dir:
            if not args.output_txt.startswith(args.output_dir):
                args.output_txt = os.path.join(args.output_dir, args.output_txt)
            if args.output_ply and not args.output_ply.startswith(args.output_dir):
                args.output_ply = os.path.join(args.output_dir, args.output_ply)

        # 初始化实例分割器
        segmenter = PointCloudInstanceSegmenter(args)

        # 执行实例分割
        segmenter.perform_instance_segmentation()

        # 保存结果到TXT
        segmenter.save_results_to_txt()

        # 可视化（如果指定）
        if args.visualize:
            segmenter.visualize_results()

        print("处理完成!")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()