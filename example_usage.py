#!/usr/bin/env python
"""
IncrementalALNS Helper使用示例
展示如何使用新的helper模块来组织destroy和repair功能
"""

from fast_alns_solver import IncrementalALNS, FastMfstspState
from incremental_alns_helper import IncrementalALNSHelper, create_incremental_alns_helper

def example_usage():
    """
    展示如何使用重构后的IncrementalALNS类
    """
    print("=== IncrementalALNS Helper 使用示例 ===\n")
    
    # 假设您已经有了必要的参数
    # 这里只是示例，实际使用时需要提供真实的参数
    node = {}  # 节点信息
    DEPOT_nodeID = 0
    V = 3  # 车辆数量
    T = 10  # 客户数量
    vehicle = {}  # 车辆信息
    uav_travel = {}  # 无人机旅行信息
    veh_distance = {}  # 车辆距离信息
    veh_travel = {}  # 车辆旅行信息
    N = 20  # 总节点数
    N_zero = 5  # 零节点数
    N_plus = 15  # 正节点数
    A_total = list(range(20))  # 所有节点
    A_cvtp = list(range(5, 15))  # 客户VTP节点
    A_vtp = list(range(5, 15))  # VTP节点
    A_aerial_relay_node = list(range(15, 20))  # 空中中继节点
    G_air = {}  # 空中图
    G_ground = {}  # 地面图
    air_matrix = {}  # 空中距离矩阵
    ground_matrix = {}  # 地面距离矩阵
    air_node_types = {}  # 空中节点类型
    ground_node_types = {}  # 地面节点类型
    A_c = list(range(5, 15))  # 客户节点
    xeee = {}  # 其他参数
    
    # 1. 创建IncrementalALNS实例（现在包含helper）
    alns_solver = IncrementalALNS(
        node, DEPOT_nodeID, V, T, vehicle, uav_travel, veh_distance, veh_travel,
        N, N_zero, N_plus, A_total, A_cvtp, A_vtp, A_aerial_relay_node, G_air, G_ground,
        air_matrix, ground_matrix, air_node_types, ground_node_types, A_c, xeee,
        max_iterations=100, max_runtime=60
    )
    
    print("1. IncrementalALNS实例创建成功，包含helper模块")
    
    # 2. 直接使用helper进行破坏操作
    print("\n2. 使用helper进行破坏操作:")
    
    # 假设您有一个初始状态
    # initial_state = FastMfstspState(...)
    
    # 使用不同的破坏策略
    destroy_types = ["random", "worst", "shaw", "vtp", "important"]
    
    for destroy_type in destroy_types:
        print(f"   - {destroy_type} 破坏: 使用 alns_solver.destroy_with_helper(state, '{destroy_type}')")
        # destroyed_state, destroyed_info = alns_solver.destroy_with_helper(initial_state, destroy_type)
    
    # 3. 直接使用helper进行修复操作
    print("\n3. 使用helper进行修复操作:")
    
    repair_types = ["greedy", "regret"]
    
    for repair_type in repair_types:
        print(f"   - {repair_type} 修复: 使用 alns_solver.repair_with_helper(state, '{repair_type}')")
        # repaired_state, schemes = alns_solver.repair_with_helper(destroyed_state, repair_type)
    
    # 4. 直接访问helper实例进行更精细的控制
    print("\n4. 直接访问helper实例:")
    print("   - 访问helper: alns_solver.helper")
    print("   - 直接调用方法: alns_solver.helper.destroy_random_removal(state)")
    print("   - 添加新的destroy方法: 在incremental_alns_helper.py中添加新方法")
    
    # 5. 添加新的destroy方法的示例
    print("\n5. 添加新的destroy方法示例:")
    print("""
    # 在incremental_alns_helper.py中添加新方法:
    def destroy_custom_removal(self, state, custom_param=None):
        \"\"\"
        自定义破坏算子
        \"\"\"
        # 实现您的自定义破坏逻辑
        pass
    
    # 在IncrementalALNS类中添加对应的包装方法:
    def destroy_with_helper(self, state, destroy_type="random", force_vtp_mode=None):
        # ... 现有代码 ...
        elif destroy_type == "custom":
            return self.helper.destroy_custom_removal(state, custom_param)
        # ... 其他代码 ...
    """)
    
    print("\n=== 使用示例完成 ===")

def demonstrate_helper_benefits():
    """
    展示使用helper模块的好处
    """
    print("\n=== Helper模块的优势 ===")
    print("1. 代码组织更清晰:")
    print("   - 破坏和修复逻辑分离到独立模块")
    print("   - fast_alns_solver.py文件更简洁")
    print("   - 便于维护和扩展")
    
    print("\n2. 功能模块化:")
    print("   - 每个destroy方法都是独立的")
    print("   - 可以单独测试和调试")
    print("   - 便于添加新的破坏策略")
    
    print("\n3. 代码复用:")
    print("   - helper类可以在其他地方复用")
    print("   - 避免代码重复")
    print("   - 统一的接口设计")
    
    print("\n4. 易于扩展:")
    print("   - 添加新的destroy方法只需在helper类中添加")
    print("   - 不需要修改主求解器类")
    print("   - 支持插件式架构")

if __name__ == "__main__":
    example_usage()
    demonstrate_helper_benefits()
