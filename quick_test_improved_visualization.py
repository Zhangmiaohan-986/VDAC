#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试改进后的可视化效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_visualization import RouteVisualizer

def create_simple_test_state():
    """创建简单的测试状态"""
    class TestState:
        def __init__(self):
            # 车辆路线
            self.vehicle_routes = [
                [1, 2, 3, 4, 5],  # 车辆1的路线
                [1, 6, 7, 8, 9],  # 车辆2的路线
            ]
            
            # 客户计划
            self.customer_plan = {
                2: (1, 2, 2, 3, 1, 1),    # 客户2：无人机1，同车任务
                3: (2, 3, 3, 4, 1, 1),    # 客户3：无人机2，同车任务
                6: (1, 6, 6, 7, 2, 2),    # 客户6：无人机1，同车任务
                7: (3, 7, 7, 8, 2, 2),    # 客户7：无人机3，同车任务
            }
    
    return TestState()

def main():
    """主函数"""
    print("🚁 快速测试改进后的可视化效果")
    print("="*50)
    
    # 创建测试状态
    state = create_simple_test_state()
    
    # 创建可视化器
    visualizer = RouteVisualizer()
    
    # 设置保存路径
    save_path = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\map_test\quick_test_improved.png"
    
    # 进行可视化
    print("🎨 开始可视化...")
    visualizer.visualize_routes(state, "改进后的清晰可视化", save_path)
    
    print(f"✅ 测试完成！图片已保存到: {save_path}")
    print("\n📋 改进内容:")
    print("1. 使用更大的画布 (20x16)")
    print("2. 无人机任务按ID分组显示")
    print("3. 右侧显示详细任务列表")
    print("4. 箭头上的标签更简洁")
    print("5. 避免标注重叠问题")

if __name__ == "__main__":
    main()
