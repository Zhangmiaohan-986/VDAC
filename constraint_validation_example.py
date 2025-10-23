#!/usr/bin/env python
"""
约束验证使用示例
演示如何使用约束验证器检查FastMfstspState的约束条件
"""

from fast_alns_solver import FastMfstspState
from constraint_validator import validate_state_constraints, quick_validate

def example_usage():
    """使用示例"""
    print("🔍 约束验证器使用示例 - 按用户思路设计")
    print("="*60)
    print("核心思路：遍历每个车辆的每条路线，检查每个无人机的发射回收序列")
    print()
    
    # 假设你有一个state对象
    # state = your_fast_mfstsp_state_object
    
    # 方法1: 使用state对象的方法（推荐）
    print("方法1: 使用state对象的方法")
    print("-" * 30)
    print("# 详细验证 - 检查每个无人机的发射回收序列")
    print("validation_result = state.validate_constraints(verbose=True)")
    print("if validation_result['is_valid']:")
    print("    print('✅ 约束验证通过')")
    print("else:")
    print("    print('❌ 约束验证失败')")
    print("    for error in validation_result['errors']:")
    print("        print(f'  - {error}')")
    print()
    
    print("# 快速验证")
    print("if state.is_constraints_satisfied():")
    print("    print('✅ 约束满足')")
    print("else:")
    print("    print('❌ 约束违反')")
    print()
    
    # 方法2: 直接使用验证函数
    print("方法2: 直接使用验证函数")
    print("-" * 30)
    print("# 详细验证")
    print("validation_result = validate_state_constraints(state, verbose=True)")
    print()
    print("# 快速验证")
    print("is_valid = quick_validate(state)")
    print()
    
    # 方法3: 在ALNS算法中集成验证
    print("方法3: 在ALNS算法中集成验证")
    print("-" * 30)
    print("# 在修复操作后验证")
    print("repaired_state, _ = repair_op(destroyed_state, strategic_bonus, num_destroyed)")
    print("if not repaired_state.is_constraints_satisfied():")
    print("    print('修复后的状态违反约束，需要重新修复')")
    print("    # 可以尝试其他修复策略或回退")
    print()
    
    print("# 在算法迭代中定期验证")
    print("if iteration % 10 == 0:  # 每10次迭代验证一次")
    print("    validation_result = current_state.validate_constraints(verbose=False)")
    print("    if not validation_result['is_valid']:")
    print("        print(f'第{iteration}次迭代发现约束违反')")
    print("        # 记录或处理约束违反")
    print()
    
    # 方法4: 调试和诊断
    print("方法4: 调试和诊断")
    print("-" * 30)
    print("# 详细诊断")
    print("validation_result = state.validate_constraints(verbose=True)")
    print("print(f'验证结果: {validation_result[\"is_valid\"]}')")
    print("print(f'错误数量: {len(validation_result[\"errors\"])}')")
    print("print(f'警告数量: {len(validation_result[\"warnings\"])}')")
    print()
    print("# 检查特定类型的约束")
    print("if validation_result['errors']:")
    print("    for error in validation_result['errors']:")
    print("        if '时间约束' in error:")
    print("            print(f'时间约束错误: {error}')")
    print("        elif '车辆路线' in error:")
    print("            print(f'车辆路线错误: {error}')")
    print("        elif '无人机分配' in error:")
    print("            print(f'无人机分配错误: {error}')")
    print()
    
    print("="*50)
    print("✅ 约束验证器已准备就绪！")
    print("现在您可以在代码中使用这些方法来验证约束条件。")

def drone_sequence_validation_example():
    """无人机序列验证示例"""
    print("\n🚁 无人机序列验证示例")
    print("="*60)
    print("核心检查：每个无人机在车辆路线中的发射回收序列是否合理")
    print()
    
    print("""
# 验证逻辑说明
def explain_validation_logic():
    '''解释验证逻辑'''
    
    print("🔍 验证逻辑:")
    print("1. 遍历每个车辆的每条路线")
    print("2. 对每个无人机，检查其在路线中的操作序列")
    print("3. 检查以下约束:")
    print("   - 不能未发射就回收无人机")
    print("   - 不能连续两次发射同一无人机而未回收")
    print("   - 路线结束时，所有发射的无人机必须被回收")
    print()
    
    print("📋 错误类型示例:")
    print("❌ 车辆1在节点5未发射就回收无人机10 (客户15)")
    print("❌ 车辆2在节点8连续发射无人机11未回收 (客户20)")
    print("❌ 车辆1路线结束时，无人机12处于发射状态但未被回收")
    print()
    
    print("✅ 正确序列示例:")
    print("节点2: 发射无人机10 (客户5)")
    print("节点4: 回收无人机10 (客户5)")
    print("节点6: 发射无人机10 (客户8)")
    print("节点8: 回收无人机10 (客户8)")
    print()
    
    print("🔧 使用方式:")
    print("# 详细验证（推荐用于调试）")
    print("validation_result = state.validate_constraints(verbose=True)")
    print("if not validation_result['is_valid']:")
    print("    for error in validation_result['errors']:")
    print("        print(error)")
    print()
    print("# 快速验证（推荐用于算法中）")
    print("if not state.is_constraints_satisfied():")
    print("    print('约束违反，需要修复')")
    """)

def integration_example():
    """集成到ALNS算法中的示例"""
    print("\n🔧 ALNS算法集成示例")
    print("="*60)
    
    print("""
# 在IncrementalALNS类中集成约束验证
class IncrementalALNS:
    def solve(self, initial_state):
        current_state = initial_state.fast_copy()
        
        for iteration in range(self.max_iterations):
            # ... 破坏和修复操作 ...
            
            # 在关键点验证约束
            if not current_state.is_constraints_satisfied():
                print(f"第{iteration}次迭代后约束验证失败")
                validation_result = current_state.validate_constraints(verbose=True)
                # 处理约束违反...
                for error in validation_result['errors']:
                    print(f"  - {error}")
                
            # 定期详细验证
            if iteration % 50 == 0:
                validation_result = current_state.validate_constraints(verbose=False)
                if not validation_result['is_valid']:
                    print(f"发现{len(validation_result['errors'])}个约束违反")
                    # 显示前几个错误
                    for i, error in enumerate(validation_result['errors'][:5]):
                        print(f"  {i+1}. {error}")
                    if len(validation_result['errors']) > 5:
                        print(f"  ... 还有{len(validation_result['errors'])-5}个错误")
                    
        return current_state
    """)

if __name__ == "__main__":
    example_usage()
    drone_sequence_validation_example()
    integration_example()
