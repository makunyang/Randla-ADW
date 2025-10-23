import numpy as np
import open3d as o3d
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def read_ply_file(ply_path):
    """读取原始PLY点云文件"""
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError(f"点云文件 {ply_path} 不包含点数据")
    return pcd

def parse_instances_txt(instances_path, label_map=None):
    """解析实例信息TXT文件"""
    instances = []
    label_map = label_map or {}
    
    with open(instances_path, 'r') as f:
        # 跳过表头
        next(f)
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            class_id = int(parts[0])
            class_name = parts[1]
            center = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            dimensions = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            point_count = int(parts[8])
            
            # 使用预定义的标签映射或自动生成颜色
            color = label_map.get(class_id, None)
            instances.append({
                'class_id': class_id,
                'class_name': class_name,
                'center': center,
                'dimensions': dimensions,
                'point_count': point_count,
                'color': color
            })
    
    return instances

def generate_label_colors(instances):
    """为不同类别生成唯一颜色"""
    label_ids = np.unique([i['class_id'] for i in instances])
    color_map = {}
    
    # 使用 matplotlib 生成均匀分布的颜色
    for i, label_id in enumerate(label_ids):
        # 生成HSV颜色空间中均匀分布的颜色
        hue = i / len(label_ids)
        color = plt.cm.hsv(hue)[:3]
        color_map[label_id] = color
    
    # 为每个实例设置颜色
    for instance in instances:
        instance['color'] = color_map[instance['class_id']]
    
    return color_map

def create_annotation_geometries(instances):
    """创建边界框和文本标签几何对象"""
    geometries = []
    
    for instance in instances:
        class_id = instance['class_id']
        class_name = instance['class_name']
        center = instance['center']
        dimensions = instance['dimensions']
        color = instance['color']
        
        # 创建轴对齐边界框(AABB)
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=center - dimensions / 2,
            max_bound=center + dimensions / 2
        )
        aabb.color = color
        geometries.append(aabb)
        
        # 创建文本标签（使用BillboardText替代Text3D）
        text = f"{class_name} ({class_id})\nPoints: {instance['point_count']}"
        text_geo = create_billboard_text(text, center, dimensions, color)
        geometries.append(text_geo)
    
    return geometries

def create_billboard_text(text, position, dimensions, color):
    """创建广告牌文本（替代Text3D）"""
    # 创建一个平面作为文本背景
    height = dimensions[2] / 5
    width = len(text.split('\n')[0]) * height / 2
    
    # 创建矩形平面
    mesh = o3d.geometry.TriangleMesh.create_rectangle(width=width, height=height)
    mesh.compute_vertex_normals()
    
    # 移动平面到指定位置
    mesh.translate(position + np.array([0, 0, dimensions[2]/2 + height/2 + 0.5]))
    
    # 设置平面颜色
    mesh.paint_uniform_color(color)
    
    # 创建文本几何体（此部分需要PIL库）
    try:
        from PIL import Image, ImageDraw, ImageFont
        # 创建图像
        img_width, img_height = int(width * 100), int(height * 100)
        img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 选择字体
        font_size = int(height * 70)
        font = ImageFont.truetype("arial.ttf", font_size)
        
        # 计算文本位置
        text_width, text_height = draw.textsize(text, font=font)
        text_x = (img_width - text_width) // 2
        text_y = (img_height - text_height) // 2
        
        # 绘制文本
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
        
        # 将图像转换为纹理
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        texture = o3d.geometry.Image(np.array(img))
        mesh.textures = [texture]
        
    except ImportError:
        print("警告: PIL库未安装，文本将显示为纯色平面")
    
    return mesh

def visualize_annotated_pcd(pcd, instances, output_image_path=None):
    """可视化标注后的点云"""
    # 为点云着色（按类别）
    if instances and 'color' in instances[0]:
        colors = np.zeros((len(pcd.points), 3))
        # 这里假设原始点云没有颜色，需要根据实例信息着色
        # 如果原始点云已有颜色，可跳过此步骤
        for instance in instances:
            # 注意：此示例中没有点索引信息，实际应用中需要根据instance['point_indices']着色
            # 这里仅作演示，使用类别颜色为点云着色
            colors += np.array(instance['color']) / len(instances)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建标注几何对象
    annotation_geometries = create_annotation_geometries(instances)
    
    # 准备可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="标注点云",
        width=1280,
        height=720
    )
    
    # 添加几何对象
    vis.add_geometry(pcd)
    for geom in annotation_geometries:
        vis.add_geometry(geom)
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_front([-0.5, 0.5, -0.5])
    ctr.set_lookat(np.mean(np.asarray(pcd.points), axis=0))
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.8)
    
    # 添加交互选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 1.0
    
    # 运行可视化
    vis.run()
    
    # 保存结果（如果指定）
    if output_image_path:
        vis.capture_screen_image(output_image_path)
        print(f"标注点云图像已保存到: {output_image_path}")
    
    vis.destroy_window()

def main():
    # 检查是否安装了PIL库
    try:
        from PIL import Image
    except ImportError:
        print("警告: PIL库未安装，文本标注功能将受限")
    
    # 设置文件路径
    ply_path = input("请输入原始PLY文件路径: ")
    instances_path = input("请输入instances.txt文件路径: ")
    output_image_path = input("请输入输出图像路径(可选，留空则不保存): ") or None
    
    try:
        # 读取点云
        pcd = read_ply_file(ply_path)
        print(f"已读取点云，点数: {len(pcd.points)}")
        
        # 解析实例信息
        instances = parse_instances_txt(instances_path)
        print(f"已解析实例信息，实例数: {len(instances)}")
        
        if not instances:
            print("警告: 未找到实例信息，无法进行标注")
            return
        
        # 生成类别颜色
        color_map = generate_label_colors(instances)
        
        # 可视化标注点云
        visualize_annotated_pcd(pcd, instances, output_image_path)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()
