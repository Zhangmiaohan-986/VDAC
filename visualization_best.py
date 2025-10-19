import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm # 导入字体管理器

def visualize_plan(best_state, output_path="VDAC/map_test/collaborative_delivery_plan.png"):
    """
    可视化车辆-无人机协同配送方案。
    (V3: 修正图例字体错误)

    参数:
    best_state: 包含方案数据的状态对象，需要有:
                - best_state.customer_plan: defaultdict(list, {customer_id: [drone_id, launch_node, customer_node, retrieval_node, ...]})
                - best_state.vehicle_routes: list[list[int]], 车辆路线列表
                - best_state.node: dict-like, {node_id: object_with_position}
                - object_with_position.position: (x, y, z) 坐标
    output_path: 图像保存路径
    """
    
    print("开始生成可视化图形 (V3 - 修正图例)...")

    # --- 1. 设置中文字体 ---
    font_paths = [
        r'C:\Windows\Fonts\SimHei.ttf', # Windows 简体黑体
        r'/System/Library/Fonts/PingFang.ttc', # macOS 苹方
        r'/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', # Linux Noto Sans CJK
        r'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' # Linux 文泉驿微米黑
    ]
    chinese_font = None
    for fp in font_paths:
        if os.path.exists(fp):
            chinese_font = fm.FontProperties(fname=fp, size=14)
            break
    
    if chinese_font:
        plt.rcParams['font.family'] = chinese_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False 
        print(f"已设置中文字体: {chinese_font.get_name()}")
    else:
        print("警告: 未找到常见中文字体，图形中的中文可能无法正常显示。")
    # --- 字体设置结束 ---


    # 2. 提取数据
    try:
        customer_plan = best_state.customer_plan
        vehicle_routes = best_state.vehicle_routes
        nodes = best_state.node
    except AttributeError as e:
        print(f"错误: 'best_state' 对象缺少必要的属性。 {e}")
        return

    # 3. 辅助函数，获取2D坐标
    def get_pos(node_id):
        try:
            return (nodes[node_id].position[0], nodes[node_id].position[1])
        except (KeyError, IndexError, TypeError):
            print(f"警告: 无法获取节点 {node_id} 的坐标。将使用 (0,0)。")
            return (0, 0)

    # 4. 设置画布
    fig, ax = plt.subplots(figsize=(24, 24)) 
    ax.set_aspect('equal')
    ax.set_title('车辆-无人机协同配送方案 (Vehicle-Drone Collaborative Delivery Plan)', fontsize=20, pad=20, fontproperties=chinese_font)
    ax.set_xlabel('X 坐标', fontsize=14, fontproperties=chinese_font)
    ax.set_ylabel('Y 坐标', fontsize=14, fontproperties=chinese_font)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 5. 准备颜色和样式
    vehicle_linestyles = ['-', '--', ':', '-.']
    vehicle_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    try:
        all_drone_ids = sorted(list(set(v[0] for v in customer_plan.values())))
        drone_cmap = plt.cm.get_cmap('hsv', len(all_drone_ids) + 1)
        drone_color_map = {drone_id: drone_cmap(i) for i, drone_id in enumerate(all_drone_ids)}
    except (IndexError, TypeError):
        print("错误: 'customer_plan' 数据格式不正确。")
        drone_color_map = {}

    
    # 6. 绘制车辆路线
    legend_elements = [] 
    
    for i, route in enumerate(vehicle_routes):
        vehicle_id = i + 1 
        route_coords = [get_pos(node_id) for node_id in route]
        
        if not route_coords:
            continue
            
        x_coords = [c[0] for c in route_coords]
        y_coords = [c[1] for c in route_coords]
        
        color = vehicle_colors[i % len(vehicle_colors)]
        linestyle = vehicle_linestyles[i % len(vehicle_linestyles)]
        
        # 新增：为车辆路线添加顺序箭头（标识行驶方向）
        for j in range(len(route_coords) - 1):
            # 起点和终点坐标
            start = route_coords[j]
            end = route_coords[j + 1]
            # 绘制线段间的箭头
            ax.annotate('',
                        xy=end,  # 箭头指向终点
                        xytext=start,  # 箭头从起点出发
                        arrowprops=dict(
                            arrowstyle='->',
                            color=color,
                            lw=2.0,  # 与路线线宽一致
                            mutation_scale=12  # 车辆箭头大小（可根据需要调整）
                        ),
                        zorder=1)  # 与路线同层，避免遮挡
        
        # --- (!! 修正 !!) ---
        # 移除 fontproperties 参数
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, linestyle=linestyle, 
                                          label=f'车辆 {vehicle_id} 路线'))
        # --- (!! 修正结束 !!) ---

    # 7. 绘制无人机任务 (发射 -> 客户 -> 回收)
    drones_in_legend = set()
    
    for customer_id, mission_data in customer_plan.items():
        try:
            drone_id, launch_node_id, _, retrieval_node_id, _, _ = mission_data
        except ValueError:
            print(f"警告: 节点 {customer_id} 的任务数据格式不正确。跳过。")
            continue

        launch_pos = get_pos(launch_node_id)
        customer_pos = get_pos(customer_id)
        retrieval_pos = get_pos(retrieval_node_id)
        
        color = drone_color_map.get(drone_id, 'black') 

        ax.annotate('', 
                    xy=customer_pos, 
                    xytext=launch_pos,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0, ls='--',mutation_scale=15),
                    zorder=2)
        
        ax.annotate('', 
                    xy=retrieval_pos, 
                    xytext=customer_pos,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0, ls='--',mutation_scale=15),
                    zorder=2)
        
        if drone_id not in drones_in_legend and drone_id in drone_color_map:
            # --- (!! 修正 !!) ---
            # 移除 fontproperties 参数
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, linestyle='--', 
                                              label=f'无人机 {drone_id}'))
            # --- (!! 修正结束 !!) ---
            drones_in_legend.add(drone_id)

    # 8. 绘制节点 (方块、圆圈和三角形)
    
    customer_node_ids = set(customer_plan.keys())

    start_node_ids = set()
    other_vehicle_node_ids = set()
    
    for route in vehicle_routes:
        if route: 
            start_node_ids.add(route[0])
            other_vehicle_node_ids.update(route[1:])
            
    other_vehicle_node_ids.difference_update(start_node_ids)
    
    marker_size = 12 
    node_text_size = 9 
    text_bbox = dict(boxstyle="round,pad=0.3", fc="yellow", ec="darkgrey", lw=0.5, alpha=0.7) 

    start_node_marker_size = marker_size + 3 
    for node_id in start_node_ids:
        pos = get_pos(node_id)
        ax.plot(pos[0], pos[1], 
                marker='^', 
                markersize=start_node_marker_size, 
                mfc='white',
                mec='green',
                mew=2.0,
                zorder=3)
        ax.text(pos[0], pos[1], str(node_id), 
                fontsize=node_text_size, 
                ha='center', 
                va='center', 
                color='green',
                zorder=4,
                bbox=text_bbox,
                clip_on=True)

    for node_id in other_vehicle_node_ids: 
        pos = get_pos(node_id)
        ax.plot(pos[0], pos[1], 
                marker='s', 
                markersize=marker_size, 
                mfc='white',
                mec='blue',
                mew=1.5,
                zorder=3)
        ax.text(pos[0], pos[1], str(node_id), 
                fontsize=node_text_size, 
                ha='center', 
                va='center', 
                color='blue', 
                zorder=4,
                bbox=text_bbox,
                clip_on=True)

    for node_id in customer_node_ids:
        pos = get_pos(node_id)
        ax.plot(pos[0], pos[1], 
                marker='o', 
                markersize=marker_size, 
                mfc='white',
                mec='red',
                mew=1.5,
                zorder=5)
        ax.text(pos[0], pos[1], str(node_id), 
                fontsize=node_text_size, 
                ha='center', 
                va='center', 
                color='red', 
                zorder=6,
                bbox=text_bbox,
                clip_on=True)

    # 9. 创建图例
    
    # --- (!! 修正 !!) ---
    # 移除所有 fontproperties 参数
    
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='none',
                                      markerfacecolor='white', markeredgecolor='green', markersize=start_node_marker_size, 
                                      label='车辆出发节点 (三角形)'))
    
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='none',
                                      markerfacecolor='white', markeredgecolor='blue', markersize=marker_size, 
                                      label='车辆路线节点 (方块)'))
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='none',
                                      markerfacecolor='white', markeredgecolor='red', markersize=marker_size, 
                                      label='客户节点 (圆圈)'))
    # --- (!! 修正结束 !!) ---

    # --- (!! 这里是正确的 !!) ---
    # ax.legend() 使用 'prop' 参数来设置字体
    ax.legend(handles=legend_elements, 
              loc='upper left', 
              bbox_to_anchor=(1.02, 1), 
              fontsize='medium',
              title="图例", 
              title_fontsize='large',
              borderpad=1,
              prop=chinese_font # <--- 这是正确的字体设置方式
             )

    # 10. 调整布局并保存
    plt.show()
    # output_dir = os.path.dirname(output_path)
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #     print(f"已创建目录: {output_dir}")

    # plt.subplots_adjust(right=0.7) 
    
    # try:
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"可视化图形已成功保存到: {output_path}")
    # except Exception as e:
    #     print(f"保存文件时出错: {e}")
        
    # plt.close(fig)

# --- 模拟数据和示例用法 ---
# (这部分是为了让代码可以独立运行并展示效果)
# (您在实际使用时，请删除或注释掉这部分，并传入您真实的 'best_state' 对象)

# 1. 模拟 'best_state' 对象的结构
# class MockNode:
#     """模拟节点对象"""
#     def __init__(self, position):
#         self.position = position

# class MockState:
#     """模拟 'best_state' 对象"""
#     def __init__(self, customer_plan, vehicle_routes, node_dict):
#         self.customer_plan = customer_plan
#         self.vehicle_routes = vehicle_routes
#         self.node = node_dict

# # 2. 使用您提供的示例数据
# customer_plan_data = {
#     66: [9, 143, 66, 144, 2, 2], 67: [9, 119, 67, 145, 2, 2], 68: [7, 145, 68, 143, 1, 1], 
#     69: [10, 131, 69, 115, 2, 1], 70: [7, 142, 70, 134, 1, 1], 71: [7, 118, 71, 123, 1, 1], 
#     72: [11, 120, 72, 136, 3, 3], 73: [8, 145, 73, 143, 1, 1], 74: [9, 129, 74, 126, 2, 1], 
#     75: [11, 124, 75, 133, 3, 1], 76: [12, 120, 76, 131, 3, 2], 77: [9, 145, 77, 129, 1, 1], 
#     78: [12, 129, 78, 132, 2, 2], 79: [7, 133, 79, 121, 1, 1], 80: [7, 129, 80, 113, 1, 1], 
#     81: [12, 132, 81, 129, 2, 1], 82: [12, 143, 82, 144, 2, 2], 83: [7, 116, 83, 137, 1, 1], 
#     84: [8, 135, 84, 128, 1, 1], 85: [7, 122, 85, 135, 1, 1], 86: [8, 133, 86, 121, 1, 1], 
#     87: [9, 133, 87, 121, 1, 1], 88: [10, 133, 88, 121, 1, 1], 89: [7, 138, 89, 141, 1, 1], 
#     90: [12, 142, 90, 119, 2, 2], 91: [8, 134, 91, 126, 1, 1], 92: [7, 127, 92, 115, 1, 1], 
#     93: [10, 145, 93, 143, 1, 1], 94: [12, 126, 94, 145, 2, 2], 95: [8, 129, 95, 113, 1, 1], 
#     96: [10, 129, 96, 113, 1, 1], 97: [12, 129, 97, 120, 1, 2], 98: [10, 134, 98, 126, 1, 1]
# }
# # 转换为 defaultdict
# mock_customer_plan = defaultdict(list)
# mock_customer_plan.update(customer_plan_data)

# # 车辆路线数据 (使用了您提供的完整数据)
# mock_vehicle_routes = [
#     [112, 117, 138, 141, 122, 135, 118, 125, 123, 128, 127, 115, 116, 137, 142, 134, 126, 145, 143],
#     [112, 131, 142, 119, 130, 134, 126, 145, 143, 144, 129, 132, 113, 133, 139, 140, 120, 112],
#     [112, 120, 136, 140, 139, 121, 114, 124, 133, 112]
# ]

# # 3. 自动生成所有涉及的节点的坐标
# all_node_ids = set()
# for route in mock_vehicle_routes:
#     all_node_ids.update(route)
# all_node_ids.update(customer_plan_data.keys())
# for mission in customer_plan_data.values():
#     all_node_ids.add(mission[1]) # launch_node
#     all_node_ids.add(mission[3]) # retrieval_node

# # 为所有节点生成随机坐标
# random.seed(42) # 保证可复现
# mock_nodes = {}
# for node_id in all_node_ids:
#     mock_nodes[node_id] = MockNode(position=(random.uniform(0, 100), random.uniform(0, 100), 0))

# # 4. 创建模拟 'best_state' 实例
# mock_best_state = MockState(mock_customer_plan, mock_vehicle_routes, mock_nodes)

# # 5. 定义输出路径并执行函数
# output_file_path = "VDAC/map_test/collaborative_delivery_plan_v3_legend_fixed.png"
# visualize_plan(mock_best_state, output_file_path)