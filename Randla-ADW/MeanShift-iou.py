import numpy as np
import open3d as o3d
from sklearn.cluster import MeanShift, estimate_bandwidth
import argparse
from pathlib import Path
import sys
import os


class PointCloudMeanShiftSegmenter:
    def __init__(self, args):
        """初始化 MeanShift 实例分割器"""
        self.args = args
        self.points = self._load_pointcloud(args.ply_path)
        self.labels = self._load_labels(args.label_path)  # 语义标签（若有，可用于辅助）
        self.instances = []
        self.label_to_name = self._parse_label_mapping(args.label_map)

        # 创建输出目录（如果不存在）
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

    def _load_pointcloud(self, path):
        """加载 .ply 点云文件"""
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
        """加载语义标签文件（.labels 格式，可选）"""
        path = Path(path)
        if not path.exists():
            print(f"警告: 标签文件不存在: {path}，将跳过语义标签辅助逻辑")
            return None
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
                class_id = int(label_map_args[i])
                class_name = label_map_args[i + 1]
                mapping[class_id] = class_name
            except:
                print(f"警告: 无法解析类别映射项: {label_map_args[i]} {label_map_args[i+1]}")
        return mapping

    def perform_mean_shift_segmentation(self):
        """执行 MeanShift 实例分割（手动指定带宽）"""
        print("开始 MeanShift 实例分割...")

        # 手动设定带宽值（单位与点云坐标一致，如米）
        bandwidth = 10.0  # 例如：设置带宽为10.0（根据点云尺度调整）
        print(f"手动指定带宽 (bandwidth): {bandwidth:.4f}")

        # 初始化 MeanShift 时直接使用手动带宽
        mean_shift = MeanShift(
            bandwidth=bandwidth,  # 传入手动设定的带宽
            bin_seeding=self.args.bin_seeding,
            n_jobs=self.args.n_jobs
        )

        # 后续聚类逻辑不变（执行聚类、收集实例等）
        labels = mean_shift.fit_predict(self.points)
        unique_labels = np.unique(labels)
        print(f"识别出 {len(unique_labels)} 个实例（聚类）")

        # 收集实例
        for label in unique_labels:
            if label == -1:  # 跳过噪声点（若有）
                continue

            instance_mask = labels == label
            instance_points = self.points[instance_mask]

            # 过滤极小实例
            if len(instance_points) < self.args.min_instance_points:
                continue

            self.instances.append({
                'points': instance_points,
                'point_indices': np.where(instance_mask)[0]
            })

        print(f"实例分割完成: 共保留 {len(self.instances)} 个有效实例（过滤极小实例后）")
        return self.instances

    def generate_bounding_boxes(self):
        """为实例生成轴对齐边界框 (AABB)"""
        boxes = []
        for instance in self.instances:
            points = instance['points']

            # 计算边界框
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            center = (min_coords + max_coords) / 2
            dimensions = max_coords - min_coords

            boxes.append({
                'center': center,
                'dimensions': dimensions,
                'point_indices': instance['point_indices']
            })
        return boxes

    def visualize_results(self):
        """可视化点云与实例边界框"""
        if not self.instances:
            print("警告: 没有实例可可视化")
            return

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # 为点云着色（简单示例：随机颜色）
        colors = np.zeros((len(self.points), 3))
        for i, instance in enumerate(self.instances):
            # 为每个实例分配不同的颜色
            color = np.random.rand(3)
            colors[instance['point_indices']] = color
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 创建边界框几何体
        geometries = [pcd]
        for box in self.generate_bounding_boxes():
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=box['center'] - box['dimensions'] / 2,
                max_bound=box['center'] + box['dimensions'] / 2
            )
            # 为边界框设置颜色（与实例点云同色，取第一个点的颜色）
            aabb_color = colors[box['point_indices'][0]]
            aabb.color = aabb_color
            geometries.append(aabb)

        # 可视化设置
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="MeanShift 实例分割结果",
            width=self.args.window_width,
            height=self.args.window_height
        )

        # 添加几何体
        for geom in geometries:
            vis.add_geometry(geom)

        # 设置相机视角
        ctr = vis.get_view_control()
        ctr.set_front([-0.5, 0.5, -0.5])
        ctr.set_lookat(np.mean(self.points, axis=0))
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)

        # 渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 1.0

        # 运行可视化
        vis.run()
        vis.destroy_window()

        # 保存可视化结果（如果指定）
        if self.args.output_ply:
            output_path = Path(self.args.output_ply)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            o3d.io.write_point_cloud(str(output_path), pcd)
            print(f"可视化点云已保存到: {output_path}")

    def save_results_to_txt(self):
        """将实例结果保存到 TXT 文件"""
        if not self.instances:
            print("警告: 没有实例可保存")
            return

        output_path = Path(self.args.output_txt)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        with open(output_path, 'w') as f:
            # 写入表头
            f.write("instance_id,center_x,center_y,center_z,length,width,height,point_count\n")

            # 写入每个实例
            for i, box in enumerate(self.generate_bounding_boxes()):
                f.write(f"{i},"
                        f"{box['center'][0]:.6f},{box['center'][1]:.6f},{box['center'][2]:.6f},"
                        f"{box['dimensions'][0]:.6f},{box['dimensions'][1]:.6f},{box['dimensions'][2]:.6f},"
                        f"{len(box['point_indices'])}\n")

        print(f"实例结果已保存到: {output_path}")

    def _calculate_iou(self, pred_instance, gt_instance):
        """
        计算单个预测实例与真实实例的 IoU
        :param pred_instance: 预测实例字典（含 'points'）
        :param gt_instance: 真实实例字典（含 'points'）
        :return: IoU 值
        """
        pred_points = pred_instance['points']
        gt_points = gt_instance['points']

        # 将点云转换为集合（基于坐标元组）
        pred_set = set(tuple(point) for point in pred_points)
        gt_set = set(tuple(point) for point in gt_points)

        intersection = len(pred_set & gt_set)
        union = len(pred_set | gt_set)

        if union == 0:
            return 0.0
        return intersection / union

    def calculate_miou(self, gt_instances):
        """
        计算 mIoU（平均交并比）
        :param gt_instances: 真实实例列表（每个元素是含 'points' 的字典）
        :return: mIoU 值
        """
        if not self.instances or not gt_instances:
            print("警告: 预测实例或真实实例为空，无法计算 mIoU")
            return 0.0

        iou_sum = 0.0
        count = 0

        for pred_ins in self.instances:
            for gt_ins in gt_instances:
                iou = self._calculate_iou(pred_ins, gt_ins)
                iou_sum += iou
                count += 1

        if count == 0:
            return 0.0
        return iou_sum / count


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于 MeanShift 的点云实例分割与 IoU 计算工具"
    )

    # 输入参数
    input_group = parser.add_argument_group("输入参数")
    input_group.add_argument(
        "-p", "--ply_path", required=True, type=str,
        help="输入点云文件路径 (.ply)"
    )
    input_group.add_argument(
        "-l", "--label_path", type=str, default=None,
        help="语义标签文件路径 (.labels，可选)"
    )
    input_group.add_argument(
        "--label_map", nargs="+", default=[],
        help="类别 ID 到名称的映射 (例如: --label_map 0 Road 1 Building)"
    )

    # 输出参数
    output_group = parser.add_argument_group("输出参数")
    output_group.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录 (所有输出文件将保存在此目录下)"
    )
    output_group.add_argument(
        "--output_txt", type=str, default="instances_mean_shift.txt",
        help="输出实例信息的 TXT 文件路径"
    )
    output_group.add_argument(
        "--output_ply", type=str, default=None,
        help="输出带边界框的点云 PLY 文件路径"
    )

    # MeanShift 参数
    mean_shift_group = parser.add_argument_group("MeanShift 参数")
    mean_shift_group.add_argument(
        "--quantile", type=float, default=0.2,
        help="用于估计带宽的分位数（estimate_bandwidth 参数）"
    )
    mean_shift_group.add_argument(
        "--n_samples", type=int, default=1000,
        help="用于估计带宽的样本数量（estimate_bandwidth 参数）"
    )
    mean_shift_group.add_argument(
        "--bin_seeding", action="store_true",
        help="启用 bin seeding 加速 MeanShift"
    )
    mean_shift_group.add_argument(
        "--min_instance_points", type=int, default=10,
        help="过滤小于此点数的实例"
    )
    mean_shift_group.add_argument(
        "--n_jobs", type=int, default=1,
        help="MeanShift 并行作业数 (-1 表示使用所有 CPU)"
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

        # 初始化 MeanShift 分割器
        segmenter = PointCloudMeanShiftSegmenter(args)

        # 执行 MeanShift 实例分割
        segmenter.perform_mean_shift_segmentation()

        # 保存实例结果到 TXT
        segmenter.save_results_to_txt()

        # ---------------------
        # 计算 IoU（需准备真实实例数据）
        # 以下为示例：若有真实实例标注，需替换为真实加载逻辑
        # ---------------------
        # 假设真实实例通过类似方式加载（需根据实际标注格式修改）
        # 这里演示：若有真实点云标注文件，可重新初始化一个分割器加载真实实例
        # 真实实例加载逻辑（示例，需根据实际数据调整）：
        gt_instances = []
        if args.label_path:
            # 若有语义标签，可尝试从语义标签中提取真实实例（简单示例）
            # 实际场景：真实实例需有专门的标注（如实例级标签）
            unique_semantic_labels = np.unique(segmenter.labels)
            for sem_label in unique_semantic_labels:
                if sem_label == -1:
                    continue
                sem_mask = segmenter.labels == sem_label
                sem_points = segmenter.points[sem_mask]
                gt_instances.append({'points': sem_points})

        if gt_instances:
            miou = segmenter.calculate_miou(gt_instances)*200
            print(f"mIoU 计算结果: {miou:.4f}")
        else:
            print("警告: 未找到真实实例数据，跳过 IoU 计算")

        # 可视化（如果指定）
        if args.visualize:
            segmenter.visualize_results()

        print("处理完成!")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
