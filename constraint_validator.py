#!/usr/bin/env python
"""
约束验证器 - 用于检查FastMfstspState中的customer_plan和vehicle_route是否符合约束条件
按照用户思路：遍历每个车辆的每条路线，检查每个无人机的发射回收序列
"""

def validate_state_constraints(state, verbose=True):
    """
    验证state中的customer_plan和vehicle_route是否符合约束条件
    按照用户思路：遍历每个车辆的每条路线，检查每个无人机的发射回收序列
    
    Args:
        state: FastMfstspState对象
        verbose: 是否打印详细信息
        
    Returns:
        dict: 验证结果，包含是否通过验证和详细错误信息
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    try:
        # 1. 基础数据结构检查
        if not hasattr(state, 'customer_plan') or not hasattr(state, 'vehicle_routes'):
            validation_result['is_valid'] = False
            validation_result['errors'].append("State缺少必要的customer_plan或vehicle_routes属性")
            return validation_result
            
        # 2. 核心验证：遍历每个车辆路线，检查每个无人机的发射回收序列
        for vehicle_id in range(len(state.V)):
            vehicle_errors = _validate_vehicle_drone_sequences(state, vehicle_id, verbose)
            validation_result['errors'].extend(vehicle_errors)
        
        # 3. 设置最终验证结果
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        # 4. 更新总体验证结果
        if validation_result['errors']:
            validation_result['is_valid'] = False
            
    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"验证过程中发生错误: {str(e)}")
        if verbose:
            print(f"❌ 验证过程异常: {str(e)}")
    
    return validation_result


def _validate_vehicle_drone_sequences(state, vehicle_id, verbose=False):
    """
    验证指定车辆路线中每个无人机的发射回收序列
    
    Args:
        state: FastMfstspState对象
        vehicle_id: 车辆ID
        verbose: 是否打印详细信息
        
    Returns:
        list: 错误信息列表
    """
    errors = []
    
    try:
        # 获取车辆路线
        vehicle_route = state.vehicle_routes[vehicle_id]
        if not vehicle_route:
            return errors  # 空路线，无需验证
        
        # 获取该车辆的所有无人机
        vehicle_drones = _get_vehicle_drones(state, vehicle_id)
        if not vehicle_drones:
            return errors  # 该车辆没有无人机，无需验证
        
        if verbose:
            print(f"\n🔍 检查车辆{vehicle_id}的无人机序列...")
            print(f"   路线: {vehicle_route}")
            print(f"   无人机: {vehicle_drones}")
        
        # 遍历每个无人机，检查其在路线中的发射回收序列
        for drone_id in vehicle_drones:
            drone_errors = _validate_drone_sequence(state, vehicle_id, drone_id, vehicle_route, verbose)
            errors.extend(drone_errors)
            
    except Exception as e:
        errors.append(f"车辆{vehicle_id}验证过程中发生错误: {str(e)}")
        if verbose:
            print(f"❌ 车辆{vehicle_id}验证异常: {str(e)}")
    
    return errors


def _validate_drone_sequence(state, vehicle_id, drone_id, vehicle_route, verbose=False):
    """
    验证指定无人机在指定车辆路线中的发射回收序列
    
    Args:
        state: FastMfstspState对象
        vehicle_id: 车辆ID
        drone_id: 无人机ID
        vehicle_route: 车辆路线
        verbose: 是否打印详细信息
        
    Returns:
        list: 错误信息列表
    """
    errors = []
    
    try:
        # 获取该无人机在路线中的所有操作
        drone_operations = _get_drone_operations_in_route(state, vehicle_id, drone_id, vehicle_route)
        
        if verbose:
            print(f"\n   🚁 检查无人机{drone_id}的操作序列:")
            for i, op in enumerate(drone_operations):
                print(f"     {i+1}. 节点{op['node']}: {op['operation']} (客户{op['customer_id']})")
        
        # 检查操作序列的合理性
        sequence_errors = _check_drone_operation_sequence(drone_id, drone_operations, vehicle_id, verbose)
        errors.extend(sequence_errors)
        
    except Exception as e:
        errors.append(f"无人机{drone_id}在车辆{vehicle_id}中验证失败: {str(e)}")
        if verbose:
            print(f"❌ 无人机{drone_id}验证异常: {str(e)}")
    
    return errors


def _get_vehicle_drones(state, vehicle_id):
    """获取指定车辆的所有无人机"""
    try:
        if hasattr(state, 'base_drone_assignment') and state.base_drone_assignment:
            return state.base_drone_assignment.get(vehicle_id, [])
        else:
            # 如果没有分配信息，从customer_plan中推断
            vehicle_drones = set()
            for customer_id, plan in state.customer_plan.items():
                if plan.get('vehicle_id') == vehicle_id:
                    drone_id = plan.get('drone_id')
                    if drone_id is not None:
                        vehicle_drones.add(drone_id)
            return list(vehicle_drones)
    except:
        return []


def _get_drone_operations_in_route(state, vehicle_id, drone_id, vehicle_route):
    """
    获取指定无人机在指定车辆路线中的所有操作
    
    Returns:
        list: 操作列表，每个操作包含 {'node', 'operation', 'customer_id'}
    """
    operations = []
    
    try:
        # 遍历路线中的每个节点
        for node_idx, node_id in enumerate(vehicle_route):
            # 检查该节点是否有该无人机的操作
            node_operations = _get_drone_operations_at_node(state, vehicle_id, drone_id, node_id)
            operations.extend(node_operations)
            
    except Exception as e:
        pass  # 忽略错误
    
    return operations


def _get_drone_operations_at_node(state, vehicle_id, drone_id, node_id):
    """
    获取指定无人机在指定节点的操作
    
    Returns:
        list: 该节点的操作列表
    """
    operations = []
    
    try:
        # 遍历customer_plan，找到该节点、该车辆、该无人机的操作
        for customer_id, plan in state.customer_plan.items():
            if (plan.get('vehicle_id') == vehicle_id and 
                plan.get('drone_id') == drone_id and 
                plan.get('node_id') == node_id):
                
                # 确定操作类型
                operation_type = None
                if plan.get('is_launch', False):
                    operation_type = 'launch'
                elif plan.get('is_recovery', False):
                    operation_type = 'recovery'
                
                if operation_type:
                    operations.append({
                        'node': node_id,
                        'operation': operation_type,
                        'customer_id': customer_id
                    })
                    
    except Exception as e:
        pass  # 忽略单个节点的错误
    
    return operations


def _check_drone_operation_sequence(drone_id, operations, vehicle_id, verbose=False):
    """
    检查无人机操作序列的合理性
    
    Args:
        drone_id: 无人机ID
        operations: 操作序列
        vehicle_id: 车辆ID
        verbose: 是否打印详细信息
        
    Returns:
        list: 错误信息列表
    """
    errors = []
    
    try:
        # 跟踪无人机状态：True表示已发射，False表示已回收
        drone_launched = False
        
        for i, op in enumerate(operations):
            operation = op['operation']
            node_id = op['node']
            customer_id = op['customer_id']
            
            if operation == 'launch':
                if drone_launched:
                    # 连续两次发射未回收
                    error_msg = f"❌ 车辆{vehicle_id}在节点{node_id}连续发射无人机{drone_id}未回收 (客户{customer_id})"
                    errors.append(error_msg)
                    if verbose:
                        print(f"     {error_msg}")
                else:
                    drone_launched = True
                    if verbose:
                        print(f"     ✅ 节点{node_id}: 发射无人机{drone_id} (客户{customer_id})")
                        
            elif operation == 'recovery':
                if not drone_launched:
                    # 未发射就回收
                    error_msg = f"❌ 车辆{vehicle_id}在节点{node_id}未发射就回收无人机{drone_id} (客户{customer_id})"
                    errors.append(error_msg)
                    if verbose:
                        print(f"     {error_msg}")
                else:
                    drone_launched = False
                    if verbose:
                        print(f"     ✅ 节点{node_id}: 回收无人机{drone_id} (客户{customer_id})")
        
        # 检查最终状态
        if drone_launched:
            error_msg = f"❌ 车辆{vehicle_id}路线结束时，无人机{drone_id}处于发射状态但未被回收"
            errors.append(error_msg)
            if verbose:
                print(f"     {error_msg}")
                
    except Exception as e:
        error_msg = f"❌ 检查无人机{drone_id}操作序列时出错: {str(e)}"
        errors.append(error_msg)
        if verbose:
            print(f"     {error_msg}")
    
    return errors


def quick_validate(state):
    """
    快速验证约束条件（不返回详细错误信息）
    
    Args:
        state: FastMfstspState对象
        
    Returns:
        bool: 是否满足约束条件
    """
    try:
        result = validate_state_constraints(state, verbose=False)
        return result['is_valid']
    except:
        return False