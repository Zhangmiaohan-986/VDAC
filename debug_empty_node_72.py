#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试节点72变成空值的问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_customer_plan_changes():
    """调试customer_plan的变化过程"""
    print("🔍 调试节点72变成空值的问题")
    print("="*60)
    
    # 模拟您的场景
    print("1. 模拟随机破坏前的状态...")
    
    # 假设这是破坏前的customer_plan
    original_customer_plan = {
        66: [9, 143, 66, 144, 2, 2],
        67: [9, 119, 67, 145, 2, 2], 
        68: [7, 145, 68, 143, 1, 1],
        69: [10, 131, 69, 115, 2, 1],
        70: [7, 142, 70, 134, 1, 1],
        71: [7, 118, 71, 123, 1, 1],
        72: [11, 120, 72, 136, 3, 3],  # 节点72
        73: [8, 145, 73, 143, 1, 1],
        74: [9, 129, 74, 126, 2, 1],
        75: [11, 124, 75, 133, 3, 1],
        76: [12, 120, 76, 131, 3, 2],
        77: [9, 145, 77, 129, 1, 1],
        78: [12, 129, 78, 132, 2, 2],
        79: [7, 133, 79, 121, 1, 1],
        80: [7, 129, 80, 113, 1, 1],
        81: [12, 132, 81, 129, 2, 1],
        82: [12, 143, 82, 144, 2, 2],
        83: [7, 116, 83, 137, 1, 1],
        84: [8, 135, 84, 128, 1, 1],
        85: [7, 122, 85, 135, 1, 1],
        86: [8, 133, 86, 121, 1, 1],
        87: [9, 133, 87, 121, 1, 1],
        88: [10, 133, 88, 121, 1, 1],
        89: [7, 138, 89, 141, 1, 1],
        90: [12, 142, 90, 119, 2, 2],
        91: [8, 134, 91, 126, 1, 1],
        92: [7, 127, 92, 115, 1, 1],
        93: [10, 145, 93, 143, 1, 1],
        94: [12, 126, 94, 145, 2, 2],
        95: [8, 129, 95, 113, 1, 1],
        96: [10, 129, 96, 113, 1, 1],
        97: [12, 129, 97, 120, 1, 2],
        98: [10, 134, 98, 126, 1, 1]
    }
    
    print(f"   原始customer_plan包含 {len(original_customer_plan)} 个节点")
    print(f"   节点72的值: {original_customer_plan.get(72, 'NOT_FOUND')}")
    
    # 模拟破坏操作
    print("\n2. 模拟随机破坏操作...")
    destroyed_customer_plan = original_customer_plan.copy()
    
    # 假设节点72被破坏
    if 72 in destroyed_customer_plan:
        destroyed_value = destroyed_customer_plan.pop(72)
        print(f"   节点72被破坏，原值: {destroyed_value}")
        print(f"   破坏后customer_plan包含 {len(destroyed_customer_plan)} 个节点")
        print(f"   节点72是否还存在: {72 in destroyed_customer_plan}")
    else:
        print("   节点72不在原始customer_plan中")
    
    # 模拟fast_copy操作
    print("\n3. 模拟fast_copy操作...")
    copied_customer_plan = destroyed_customer_plan.copy()
    print(f"   拷贝后customer_plan包含 {len(copied_customer_plan)} 个节点")
    print(f"   节点72是否在拷贝中: {72 in copied_customer_plan}")
    
    # 检查可能的问题点
    print("\n4. 检查可能的问题点...")
    
    # 问题点1：检查是否有地方会添加空值
    print("   问题点1: 检查是否有地方会添加空值")
    if 72 in copied_customer_plan:
        value_72 = copied_customer_plan[72]
        print(f"   节点72的值: {value_72}")
        print(f"   值是否为空: {value_72 is None or value_72 == [] or value_72 == ''}")
    else:
        print("   节点72不在拷贝中")
    
    # 问题点2：检查defaultdict行为
    print("\n   问题点2: 检查defaultdict行为")
    from collections import defaultdict
    
    # 模拟可能的defaultdict初始化
    test_dd = defaultdict(list)
    test_dd[72] = []  # 这可能会添加空列表
    print(f"   defaultdict[72] = {test_dd[72]}")
    
    # 问题点3：检查字典更新操作
    print("\n   问题点3: 检查字典更新操作")
    test_dict = {}
    test_dict.update({72: []})  # 这可能会添加空列表
    print(f"   dict.update后[72] = {test_dict[72]}")
    
    print("\n5. 建议的调试方法:")
    print("   - 在destroy_random_removal方法中添加调试输出")
    print("   - 在fast_copy方法中添加调试输出")
    print("   - 在repair_greedy_insertion方法中添加调试输出")
    print("   - 检查是否有地方使用了defaultdict(list)")

def add_debug_output_to_code():
    """提供在代码中添加调试输出的建议"""
    print("\n" + "="*60)
    print("🛠️ 建议在代码中添加的调试输出")
    print("="*60)
    
    debug_code = '''
# 在destroy_random_removal方法中添加：
print(f"DEBUG: 破坏前customer_plan包含节点: {list(new_state.customer_plan.keys())}")
print(f"DEBUG: 节点72在破坏前: {72 in new_state.customer_plan}")

# 在破坏操作后添加：
print(f"DEBUG: 破坏后customer_plan包含节点: {list(new_state.customer_plan.keys())}")
print(f"DEBUG: 节点72在破坏后: {72 in new_state.customer_plan}")

# 在fast_copy方法中添加：
print(f"DEBUG: fast_copy前customer_plan包含节点: {list(self.customer_plan.keys())}")
print(f"DEBUG: 节点72在fast_copy前: {72 in self.customer_plan}")

# 在fast_copy后添加：
print(f"DEBUG: fast_copy后customer_plan包含节点: {list(new_state.customer_plan.keys())}")
print(f"DEBUG: 节点72在fast_copy后: {72 in new_state.customer_plan}")

# 在repair_greedy_insertion方法开始时添加：
print(f"DEBUG: 修复前customer_plan包含节点: {list(state.customer_plan.keys())}")
print(f"DEBUG: 节点72在修复前: {72 in state.customer_plan}")

# 在修复操作后添加：
print(f"DEBUG: 修复后customer_plan包含节点: {list(repaired_state.customer_plan.keys())}")
print(f"DEBUG: 节点72在修复后: {72 in repaired_state.customer_plan}")
'''
    
    print(debug_code)

if __name__ == "__main__":
    debug_customer_plan_changes()
    add_debug_output_to_code()

