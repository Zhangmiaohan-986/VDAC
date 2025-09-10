import pickle
import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime
def convert_for_saving(data):
    """
    转换数据结构，移除不可序列化的对象
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            try:
                # 测试是否可以序列化
                pickle.dumps(value)
                new_dict[key] = value
            except (pickle.PicklingError, AttributeError):
                # 如果是不可序列化的对象，转换为字符串描述
                if callable(value):
                    new_dict[key] = f"<function {value.__name__}>"
                else:
                    new_dict[key] = str(value)
        return new_dict
    return data

def save_solution_data(time_uav_task_dict, time_customer_plan, time_uav_plan, 
                      vehicle_plan_time, vehicle_task_data, save_dir='saved_solutions'):
    """
    保存规划结果数据，处理不可序列化的对象
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 深拷贝数据以避免修改原始数据
    save_data = {
        'time_uav_task_dict': deepcopy(time_uav_task_dict),
        'time_customer_plan': deepcopy(time_customer_plan),
        'time_uav_plan': deepcopy(time_uav_plan),
        'vehicle_plan_time': deepcopy(vehicle_plan_time),
        'vehicle_task_data': deepcopy(vehicle_task_data)
    }
    
    # 转换数据结构
    for key in save_data:
        save_data[key] = convert_for_saving(save_data[key])
    
    # 使用时间戳作为文件名
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'solution_{timestamp}.pkl'
    
    try:
        # 保存为pickle文件
        with open(f'{save_dir}/{filename}', 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 保存可读的JSON版本
        json_data = {
            'timestamp': timestamp,
            'summary': {
                'time_uav_task_count': len(time_uav_task_dict) if isinstance(time_uav_task_dict, dict) else 'N/A',
                'time_customer_plan_count': len(time_customer_plan) if isinstance(time_customer_plan, dict) else 'N/A',
                'vehicle_plan_time_count': len(vehicle_plan_time) if isinstance(vehicle_plan_time, dict) else 'N/A'
            }
        }
        
        with open(f'{save_dir}/{filename}.json', 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"解决方案已保存到: {save_dir}/{filename}")
        return filename
        
    except Exception as e:
        print(f"保存数据时发生错误: {e}")
        # 创建错误日志
        error_log = {
            'timestamp': timestamp,
            'error': str(e),
            'data_types': {
                'time_uav_task_dict': str(type(time_uav_task_dict)),
                'time_customer_plan': str(type(time_customer_plan)),
                'time_uav_plan': str(type(time_uav_plan)),
                'vehicle_plan_time': str(type(vehicle_plan_time)),
                'vehicle_task_data': str(type(vehicle_task_data))
            }
        }
        error_filename = f'error_log_{timestamp}.json'
        with open(f'{save_dir}/{error_filename}', 'w') as f:
            json.dump(error_log, f, indent=4)
        raise

def load_solution_data(filename, load_dir='saved_solutions'):
    """
    加载保存的规划结果数据
    """
    try:
        filepath = f'{load_dir}/{filename}'
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return (
            data['time_uav_task_dict'],
            data['time_customer_plan'],
            data['time_uav_plan'],
            data['vehicle_plan_time'],
            data['vehicle_task_data']
        )
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        print(f"尝试加载的文件: {filepath}")
        raise

def save_partial_solution(data, name, save_dir='saved_solutions'):
    """
    保存部分解决方案数据
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{name}_{timestamp}.pkl'
    
    try:
        converted_data = convert_for_saving(deepcopy(data))
        with open(f'{save_dir}/{filename}', 'wb') as f:
            pickle.dump(converted_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"部分数据已保存到: {save_dir}/{filename}")
        return filename
    except Exception as e:
        print(f"保存部分数据时发生错误: {e}")
        raise


import os
import pickle
import json
import datetime
import networkx as nx
from collections import defaultdict
import shutil
import numpy as np  # 确保导入numpy

def list_saved_files():
    """列出所有保存的数据文件"""
    save_dir = 'saved_solutions'
    if not os.path.exists(save_dir):
        print("没有找到saved_solutions目录")
        return []
    
    files = [f for f in os.listdir(save_dir) if f.startswith('input_data_') and f.endswith('.pkl')]
    if not files:
        print("没有找到保存的数据文件")
        return []
    
    print("\n当前保存的文件列表:")
    for i, f in enumerate(files, 1):
        file_path = os.path.join(save_dir, f)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
        file_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
        print(f"{i}. {f}")
        print(f"   创建时间: {file_time}")
        print(f"   文件大小: {file_size:.2f}MB")
    return files

def delete_saved_file(filename):
    """删除指定的数据文件"""
    save_dir = 'saved_solutions'
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    if not filename.startswith('input_data_'):
        filename = 'input_data_' + filename
        
    file_path = os.path.join(save_dir, filename)
    json_file = file_path.replace('.pkl', '.json')
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除文件: {filename}")
        if os.path.exists(json_file):
            os.remove(json_file)
            print(f"已删除对应的JSON文件")
        return True
    except Exception as e:
        print(f"删除文件时出错: {e}")
        return False

def backup_saved_file(filename, backup_name=None):
    """备份指定的数据文件"""
    save_dir = 'saved_solutions'
    backup_dir = os.path.join(save_dir, 'backups')
    
    # 确保文件名格式正确
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    if not filename.startswith('input_data_'):
        filename = 'input_data_' + filename
    
    # 创建备份目录
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    source_path = os.path.join(save_dir, filename)
    source_json = source_path.replace('.pkl', '.json')
    
    if not os.path.exists(source_path):
        print(f"未找到源文件: {filename}")
        return False
    
    try:
        # 如果没有指定备份名称，使用时间戳
        if backup_name is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}_{filename}"
        elif not backup_name.startswith('backup_'):
            backup_name = f"backup_{backup_name}"
        if not backup_name.endswith('.pkl'):
            backup_name += '.pkl'
            
        # 复制文件
        backup_path = os.path.join(backup_dir, backup_name)
        backup_json = backup_path.replace('.pkl', '.json')
        
        shutil.copy2(source_path, backup_path)
        if os.path.exists(source_json):
            shutil.copy2(source_json, backup_json)
            
        print(f"已创建备份: {backup_name}")
        return True
    except Exception as e:
        print(f"创建备份时出错: {e}")
        return False

def get_latest_saved_file():
    """获取最新保存的数据文件"""
    save_dir = 'saved_solutions'
    if not os.path.exists(save_dir):
        print("没有找到saved_solutions目录")
        return None
    
    files = [f for f in os.listdir(save_dir) if f.startswith('input_data_') and f.endswith('.pkl')]
    if not files:
        print("没有找到保存的数据文件")
        return None
        
    # 按文件名排序（因为文件名包含时间戳）
    latest_file = sorted(files)[-1]
    return latest_file.replace('.pkl', '')

def convert_defaultdict_to_dict(d):
    """将defaultdict转换为普通dict"""
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [convert_defaultdict_to_dict(v) for v in d]
    return d

def convert_str_to_tuple_key(s):
    """将字符串形式的元组键转回元组"""
    if s.startswith('(') and s.endswith(')'):
        try:
            # 移除括号并分割字符串
            items = s[1:-1].split(',')
            # 转换每个项
            items = [int(item.strip()) if item.strip().isdigit() else item.strip() for item in items]
            return tuple(items) if len(items) > 1 else items[0]
        except:
            return s
    return s

def convert_keys_back(obj):
    """将字典中的字符串键转回元组（如果可能）"""
    if isinstance(obj, dict):
        return {convert_str_to_tuple_key(k): convert_keys_back(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_back(v) for v in obj]
    return obj
# def convert_defaultdict_to_dict(obj):
#     """将defaultdict转换为普通dict"""
#     if isinstance(obj, defaultdict):
#         return {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
#     elif isinstance(obj, dict):
#         return {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_defaultdict_to_dict(x) for x in obj]
#     return obj

def create_object_from_dict(data_dict):
    """从字典创建对象"""
    if isinstance(data_dict, dict):
        # 检查是否是特殊类型的对象
        if '_class_name' in data_dict:
            try:
                if data_dict['_class_name'] == 'Node':
                    from initialize import Node
                    node = Node()
                    for attr_name, attr_value in data_dict['attributes'].items():
                        setattr(node, attr_name, create_object_from_dict(attr_value))
                    return node
                elif data_dict['_class_name'] == 'Vehicle':
                    from initialize import Vehicle
                    vehicle = Vehicle()
                    for attr_name, attr_value in data_dict['attributes'].items():
                        setattr(vehicle, attr_name, create_object_from_dict(attr_value))
                    return vehicle
                elif data_dict['_class_name'] == 'vehicle_task':
                    from task_data import vehicle_task
                    attrs = data_dict['attributes']
                    task = vehicle_task(
                        attrs.get('id'),
                        attrs.get('vehicleType'),
                        attrs.get('node_id'),
                        attrs.get('node')
                    )
                    for attr_name, attr_value in attrs.items():
                        if attr_name not in ['id', 'vehicleType', 'node_id', 'node']:
                            setattr(task, attr_name, create_object_from_dict(attr_value))
                    return task
            except ImportError as e:
                print(f"警告: 无法导入类 {data_dict['_class_name']}, 使用字典替代: {str(e)}")
                return data_dict['attributes']
        # 处理普通字典，保持键的类型
        return {convert_key_to_original(k): create_object_from_dict(v) 
                for k, v in data_dict.items()}
    elif isinstance(data_dict, list):
        return [create_object_from_dict(item) for item in data_dict]
    elif isinstance(data_dict, tuple):
        return tuple(create_object_from_dict(item) for item in data_dict)
    return data_dict

def convert_dict_to_defaultdict(data):
    """将普通dict转换为defaultdict"""
    if isinstance(data, dict):
        # 创建新的defaultdict
        dd = defaultdict(lambda: None)
        for k, v in data.items():
            if isinstance(v, dict):
                dd[k] = convert_dict_to_defaultdict(v)
            else:
                dd[k] = v
        return dd
    return data

def save_input_data(input_data):
    """保存输入数据"""
    # 创建保存目录
    save_dir = 'saved_solutions'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成时间戳文件名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'input_data_{timestamp}'
    
    # 预处理数据，转换defaultdict和其他特殊对象
    serializable_data = {}
    for key, value in input_data.items():
        if isinstance(value, defaultdict):
            # 将defaultdict转换为普通dict
            serializable_data[key] = {
                'type': 'defaultdict',
                'data': convert_defaultdict_to_dict(value)
            }
        elif isinstance(value, nx.Graph):
            serializable_data[key] = {
                'type': 'networkx_graph',
                'data': nx.node_link_data(value)
            }
        elif key in ['node', 'vehicle']:
            serializable_data[key] = {
                'type': 'object_dict',
                'data': {k: obj.__dict__ for k, obj in value.items()}
            }
        else:
            serializable_data[key] = {
                'type': 'regular',
                'data': value
            }
    
    # 保存为pickle格式
    pickle_filename = os.path.join(save_dir, f'{base_filename}.pkl')
    with open(pickle_filename, 'wb') as f:
        pickle.dump(serializable_data, f)
    
    return base_filename


def save_input_data_with_name(input_data, custom_name):
    """使用自定义名称保存数据"""
    if not custom_name.endswith('.pkl'):
        custom_name = custom_name + '.pkl'
    if not custom_name.startswith('input_data_'):
        custom_name = 'input_data_' + custom_name
    
    save_dir = 'saved_solutions'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, custom_name)
    
    if os.path.exists(file_path):
        print(f"文件 {custom_name} 已存在，创建备份...")
        backup_saved_file(custom_name)
    
    try:
        serializable_data = {}
        for key, value in input_data.items():
            if isinstance(value, defaultdict):
                converted_data = dict(value)  # 转换defaultdict为普通dict
            else:
                converted_data = value
            
            # 转换所有复杂对象为可序列化的字典
            converted_data = convert_object_to_dict(converted_data)
            
            serializable_data[key] = {
                'type': 'defaultdict' if isinstance(value, defaultdict) else 'regular',
                'data': converted_data
            }
        
        # 保存为pickle格式
        with open(file_path, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        print(f"数据已保存为: {custom_name}")
        return custom_name.replace('.pkl', '')
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")
        print("详细错误信息:")
        import traceback
        print(traceback.format_exc())
        return None

# def save_input_data_with_name(input_data, custom_name):
#     """使用自定义名称保存数据"""
#     if not custom_name.endswith('.pkl'):
#         custom_name = custom_name + '.pkl'
#     if not custom_name.startswith('input_data_'):
#         custom_name = 'input_data_' + custom_name
    
#     save_dir = 'saved_solutions'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     file_path = os.path.join(save_dir, custom_name)
    
#     if os.path.exists(file_path):
#         print(f"文件 {custom_name} 已存在，创建备份...")
#         backup_saved_file(custom_name)
    
#     try:
#         serializable_data = {}
#         for key, value in input_data.items():
#             if isinstance(value, defaultdict):
#                 converted_data = convert_defaultdict_to_dict(value)
#                 converted_data = convert_key_to_str(converted_data)
#                 converted_data = convert_numpy_types(converted_data)
#                 converted_data = convert_vehicle_task(converted_data)
#                 converted_data = convert_node_object(converted_data)
#                 serializable_data[key] = {
#                     'type': 'defaultdict',
#                     'data': converted_data
#                 }
#             elif isinstance(value, nx.Graph):
#                 graph_data = nx.node_link_data(value)
#                 graph_data = convert_numpy_types(graph_data)
#                 serializable_data[key] = {
#                     'type': 'networkx_graph',
#                     'data': graph_data
#                 }
#             elif key in ['node', 'vehicle']:
#                 # 特别处理node和vehicle对象
#                 obj_dict = {}
#                 for k, obj in value.items():
#                     if key == 'node':
#                         obj_dict[k] = convert_node_object(obj)
#                     else:
#                         obj_dict[k] = convert_vehicle_task(obj)
#                 obj_dict = convert_numpy_types(obj_dict)
#                 serializable_data[key] = {
#                     'type': 'object_dict',
#                     'data': obj_dict
#                 }
#             else:
#                 converted_value = convert_key_to_str(value)
#                 converted_value = convert_numpy_types(converted_value)
#                 converted_value = convert_vehicle_task(converted_value)
#                 converted_value = convert_node_object(converted_value)
#                 serializable_data[key] = {
#                     'type': 'regular',
#                     'data': converted_value
#                 }
        
#         # 只保存pickle格式，因为JSON可能还有其他序列化问题
#         with open(file_path, 'wb') as f:
#             pickle.dump(serializable_data, f)
        
#         print(f"数据已保存为: {custom_name}")
#         return custom_name.replace('.pkl', '')
#     except Exception as e:
#         print(f"保存数据时出错: {str(e)}")
#         print("详细错误信息:")
#         import traceback
#         print(traceback.format_exc())
#         return None


def reconstruct_vehicle_task(data):
    """从字典重建vehicle_task对象"""
    if isinstance(data, dict):
        if data.get('_class_name') == 'vehicle_task':
            # 导入vehicle_task类
            from task_data import vehicle_task
            # 创建新的vehicle_task实例（需要提供必要的参数）
            # 注意：这里需要根据你的vehicle_task类的__init__方法进行调整
            task = vehicle_task(
                data['attributes'].get('id'),
                data['attributes'].get('vehicleType'),
                data['attributes'].get('node_id'),
                data['attributes'].get('node')
            )
            # 设置其他属性
            for attr_name, attr_value in data['attributes'].items():
                if attr_name not in ['id', 'vehicleType', 'node_id', 'node']:  # 跳过已经在初始化时设置的属性
                    if isinstance(attr_value, dict):
                        setattr(task, attr_name, reconstruct_vehicle_task(attr_value))
                    elif isinstance(attr_value, list):
                        setattr(task, attr_name, [reconstruct_vehicle_task(item) for item in attr_value])
                    else:
                        setattr(task, attr_name, attr_value)
            return task
        else:
            return {k: reconstruct_vehicle_task(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [reconstruct_vehicle_task(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(reconstruct_vehicle_task(item) for item in data)
    return data

def convert_vehicle_task(obj):
    """转换vehicle_task对象为可序列化的字典"""
    if isinstance(obj, dict):
        return {k: convert_vehicle_task(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_vehicle_task(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_vehicle_task(item) for item in obj)
    elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'vehicle_task':
        # 将vehicle_task对象转换为字典
        task_dict = {
            '_class_name': 'vehicle_task',  # 保存类名以便后续重建
            'attributes': {}
        }
        for attr_name, attr_value in obj.__dict__.items():
            # 递归处理嵌套的对象
            if isinstance(attr_value, (list, tuple)):
                task_dict['attributes'][attr_name] = [convert_vehicle_task(item) for item in attr_value]
            elif isinstance(attr_value, dict):
                task_dict['attributes'][attr_name] = {k: convert_vehicle_task(v) for k, v in attr_value.items()}
            elif hasattr(attr_value, '__dict__'):
                task_dict['attributes'][attr_name] = convert_vehicle_task(attr_value)
            else:
                task_dict['attributes'][attr_name] = attr_value
        return task_dict
    return obj

def convert_node_object(obj):
    """转换make_node对象为可序列化的字典"""
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'make_node':
        # 获取对象的所有属性并转换为字典
        node_dict = {
            '_class_name': 'make_node',  # 保存类名以便后续重建
            'attributes': {}
        }
        for attr_name, attr_value in obj.__dict__.items():
            # 递归处理嵌套的对象
            if isinstance(attr_value, (list, tuple)):
                node_dict['attributes'][attr_name] = [convert_node_object(item) for item in attr_value]
            elif isinstance(attr_value, dict):
                node_dict['attributes'][attr_name] = {k: convert_node_object(v) for k, v in attr_value.items()}
            elif hasattr(attr_value, '__dict__'):
                node_dict['attributes'][attr_name] = convert_node_object(attr_value)
            else:
                node_dict['attributes'][attr_name] = attr_value
        return node_dict
    return obj

# def save_input_data_with_name(input_data, custom_name):
#     """使用自定义名称保存数据"""
#     # 确保文件名格式正确
#     if not custom_name.endswith('.pkl'):
#         custom_name = custom_name + '.pkl'
#     if not custom_name.startswith('input_data_'):
#         custom_name = 'input_data_' + custom_name
    
#     save_dir = 'saved_solutions'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     file_path = os.path.join(save_dir, custom_name)
    
#     # 如果文件已存在，先备份
#     if os.path.exists(file_path):
#         print(f"文件 {custom_name} 已存在，创建备份...")
#         backup_saved_file(custom_name)
    
#     try:
#         # 预处理数据，转换defaultdict和其他特殊对象
#         serializable_data = {}
#         for key, value in input_data.items():
#             if isinstance(value, defaultdict):
#                 # 将defaultdict转换为普通dict并处理numpy类型
#                 converted_data = convert_defaultdict_to_dict(value)
#                 converted_data = convert_key_to_str(converted_data)
#                 converted_data = convert_numpy_types(converted_data)
#                 serializable_data[key] = {
#                     'type': 'defaultdict',
#                     'data': converted_data
#                 }
#             elif isinstance(value, nx.Graph):
#                 graph_data = nx.node_link_data(value)
#                 graph_data = convert_numpy_types(graph_data)
#                 serializable_data[key] = {
#                     'type': 'networkx_graph',
#                     'data': graph_data
#                 }
#             elif key in ['node', 'vehicle']:
#                 obj_dict = {k: obj.__dict__ for k, obj in value.items()}
#                 obj_dict = convert_numpy_types(obj_dict)
#                 serializable_data[key] = {
#                     'type': 'object_dict',
#                     'data': obj_dict
#                 }
#             else:
#                 # 对其他数据也进行类型转换
#                 converted_value = convert_key_to_str(value)
#                 converted_value = convert_numpy_types(converted_value)
#                 serializable_data[key] = {
#                     'type': 'regular',
#                     'data': converted_value
#                 }
        
#         # 保存为pickle格式
#         with open(file_path, 'wb') as f:
#             pickle.dump(serializable_data, f)
        
#         # 保存为json格式（用于查看）
#         json_path = file_path.replace('.pkl', '.json')
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        
#         print(f"数据已保存为: {custom_name}")
#         return custom_name.replace('.pkl', '')
#     except Exception as e:
#         print(f"保存数据时出错: {str(e)}")
#         print("详细错误信息:")
#         import traceback
#         print(traceback.format_exc())
#         return None

def convert_numpy_types(obj):
    """转换numpy数据类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    return obj

# def convert_key_to_str(obj):
#     """将字典中的元组键转换为字符串"""
#     if isinstance(obj, dict):
#         return {str(k): convert_key_to_str(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_key_to_str(v) for v in obj]
#     elif isinstance(obj, defaultdict):
#         return {str(k): convert_key_to_str(v) for k, v in obj.items()}
#     return obj

def convert_key_to_original(key):
    """将键转换为原始类型"""
    try:
        # 尝试转换为整数
        return int(key)
    except (ValueError, TypeError):
        return key

def manage_saved_data():
    """管理保存的数据文件"""
    while True:
        print("\n=== 数据文件管理 ===")
        print("1. 列出所有保存的文件")
        print("2. 删除指定文件")
        print("3. 备份指定文件")
        print("4. 保存新数据")
        print("5. 退出")
        
        choice = input("请选择操作 (1-5): ")
        
        if choice == '1':
            list_saved_files()
        
        elif choice == '2':
            files = list_saved_files()
            if files:
                filename = input("请输入要删除的文件名: ")
                delete_saved_file(filename)
        
        elif choice == '3':
            files = list_saved_files()
            if files:
                filename = input("请输入要备份的文件名: ")
                backup_name = input("请输入备份名称（直接回车使用时间戳）: ")
                if not backup_name:
                    backup_name = None
                backup_saved_file(filename, backup_name)
        
        elif choice == '4':
            custom_name = input("请输入新数据文件名: ")
            # 这里需要你提供input_data
            # save_input_data_with_name(input_data, custom_name)
            print("请在代码中提供input_data后使用此功能")
        
        elif choice == '5':
            break
        
        else:
            print("无效的选择，请重试")


def load_input_data(filename):
    """加载保存的数据"""
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    if not filename.startswith('input_data_'):
        filename = 'input_data_' + filename
    
    file_path = os.path.join('saved_solutions', filename)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            restored_data = {}
            for key, value in data.items():
                if value['type'] == 'defaultdict':
                    restored_data[key] = defaultdict(list, create_object_from_dict(value['data']))
                else:
                    restored_data[key] = create_object_from_dict(value['data'])
            return restored_data
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        print("详细错误信息:")
        import traceback
        print(traceback.format_exc())
        return None
    
# def load_input_data(filename):
#     """加载输入数据"""
#     save_dir = 'saved_solutions'
#     pickle_path = os.path.join(save_dir, f'{filename}.pkl')
    
#     with open(pickle_path, 'rb') as f:
#         data = pickle.load(f)
    
#     # 还原数据结构
#     restored_data = {}
#     for key, value in data.items():
#         if value['type'] == 'defaultdict':
#             # 将普通dict转回defaultdict
#             dd = defaultdict(lambda: None)
#             dd.update(value['data'])
#             restored_data[key] = dd
#         elif value['type'] == 'networkx_graph':
#             restored_data[key] = nx.node_link_graph(value['data'])
#         elif value['type'] == 'object_dict':
#             restored_data[key] = {k: create_object_from_dict(v, key) for k, v in value['data'].items()}
#         else:
#             restored_data[key] = value['data']
    
#     return restored_data

# def create_object_from_dict(data_dict, obj_type):
#     """从字典创建对象"""
#     class DummyObject:
#         def __init__(self, **kwargs):
#             for key, value in kwargs.items():
#                 setattr(self, key, value)
    
#     return DummyObject(**data_dict)

# def create_object_from_dict(data_dict, class_type=None):
#     """从字典创建对象"""
#     if isinstance(data_dict, dict):
#         # 检查是否是特殊类型的对象
#         if '_class_name' in data_dict:
#             if data_dict['_class_name'] == 'make_node':
#                 from initialize import make_node
#                 node = make_node()
#                 for attr_name, attr_value in data_dict['attributes'].items():
#                     setattr(node, attr_name, create_object_from_dict(attr_value))
#                 return node
#             elif data_dict['_class_name'] == 'vehicle_task':
#                 from task_data import vehicle_task
#                 # 根据vehicle_task的初始化参数创建对象
#                 attrs = data_dict['attributes']
#                 task = vehicle_task(
#                     attrs.get('id'),
#                     attrs.get('vehicleType'),
#                     attrs.get('node_id'),
#                     attrs.get('node')
#                 )
#                 for attr_name, attr_value in attrs.items():
#                     if attr_name not in ['id', 'vehicleType', 'node_id', 'node']:
#                         setattr(task, attr_name, create_object_from_dict(attr_value))
#                 return task
#             elif data_dict['_class_name'] == 'make_vehicle':
#                 from initialize import make_vehicle
#                 vehicle = make_vehicle()
#                 for attr_name, attr_value in data_dict['attributes'].items():
#                     setattr(vehicle, attr_name, create_object_from_dict(attr_value))
#                 return vehicle
#         else:
#             # 处理普通字典
#             return {k: create_object_from_dict(v) for k, v in data_dict.items()}
#     elif isinstance(data_dict, list):
#         return [create_object_from_dict(item) for item in data_dict]
#     elif isinstance(data_dict, tuple):
#         return tuple(create_object_from_dict(item) for item in data_dict)
#     return data_dict

# def create_object_from_dict(data_dict, class_type=None):
#     """从字典创建对象"""
#     if isinstance(data_dict, dict):
#         # 检查是否是特殊类型的对象
#         if '_class_name' in data_dict:
#             try:
#                 if data_dict['_class_name'] == 'Node':  # 修改类名
#                     from initialize import Node  # 修改导入的类名
#                     node = Node()
#                     for attr_name, attr_value in data_dict['attributes'].items():
#                         setattr(node, attr_name, create_object_from_dict(attr_value))
#                     return node
#                 elif data_dict['_class_name'] == 'vehicle_task':
#                     from task_data import vehicle_task
#                     attrs = data_dict['attributes']
#                     task = vehicle_task(
#                         attrs.get('id'),
#                         attrs.get('vehicleType'),
#                         attrs.get('node_id'),
#                         attrs.get('node')
#                     )
#                     for attr_name, attr_value in attrs.items():
#                         if attr_name not in ['id', 'vehicleType', 'node_id', 'node']:
#                             setattr(task, attr_name, create_object_from_dict(attr_value))
#                     return task
#                 elif data_dict['_class_name'] == 'Vehicle':  # 修改类名
#                     from initialize import Vehicle  # 修改导入的类名
#                     vehicle = Vehicle()
#                     for attr_name, attr_value in data_dict['attributes'].items():
#                         setattr(vehicle, attr_name, create_object_from_dict(attr_value))
#                     return vehicle
#             except ImportError as e:
#                 print(f"警告: 无法导入类 {data_dict['_class_name']}, 使用字典替代: {str(e)}")
#                 return data_dict['attributes']
#         # 处理普通字典
#         return {k: create_object_from_dict(v) for k, v in data_dict.items()}
#     elif isinstance(data_dict, list):
#         return [create_object_from_dict(item) for item in data_dict]
#     elif isinstance(data_dict, tuple):
#         return tuple(create_object_from_dict(item) for item in data_dict)
#     return data_dict

def convert_object_to_dict(obj):
    """将对象转换为可序列化的字典"""
    if hasattr(obj, '__class__'):
        class_name = obj.__class__.__name__
        if class_name in ['Node', 'Vehicle', 'vehicle_task']:
            return {
                '_class_name': class_name,
                'attributes': {
                    attr_name: convert_object_to_dict(attr_value)
                    for attr_name, attr_value in obj.__dict__.items()
                }
            }
    if isinstance(obj, dict):
        # 保持键的原始类型
        return {convert_key_to_original(k): convert_object_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_object_to_dict(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# def convert_object_to_dict(obj):
#     """将对象转换为可序列化的字典"""
#     if hasattr(obj, '__class__'):
#         class_name = obj.__class__.__name__
#         if class_name in ['Node', 'Vehicle', 'vehicle_task']:  # 修改类名
#             return {
#                 '_class_name': class_name,
#                 'attributes': {
#                     attr_name: convert_object_to_dict(attr_value)
#                     for attr_name, attr_value in obj.__dict__.items()
#                 }
#             }
#     if isinstance(obj, dict):
#         return {k: convert_object_to_dict(v) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return [convert_object_to_dict(item) for item in obj]
#     elif isinstance(obj, (int, float, str, bool, type(None))):
#         return obj
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj

# def convert_object_to_dict(obj):
#     """将对象转换为可序列化的字典"""
#     if hasattr(obj, '__class__'):
#         class_name = obj.__class__.__name__
#         if class_name in ['make_node', 'vehicle_task', 'make_vehicle']:
#             return {
#                 '_class_name': class_name,
#                 'attributes': {
#                     attr_name: convert_object_to_dict(attr_value)
#                     for attr_name, attr_value in obj.__dict__.items()
#                 }
#             }
#     if isinstance(obj, dict):
#         return {k: convert_object_to_dict(v) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return [convert_object_to_dict(item) for item in obj]
#     elif isinstance(obj, (int, float, str, bool, type(None))):
#         return obj
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj

def run_low_update_time_from_saved(input_filename):
    """从保存的数据运行low_update_time"""
    input_data = load_input_data(input_filename)
    if input_data is None:
        print("错误: 无法加载输入数据")
        return None
    
    try:
        return (input_data['uav_task_dict'],
                input_data['best_customer_plan'],
                input_data['best_uav_plan'],
                input_data['best_vehicle_route'],
                input_data['vehicle_task_data'],
                input_data['vehicle_arrival_time'],
                input_data['node'],
                input_data['DEPOT_nodeID'],
                input_data['V'],
                input_data['T'],
                input_data['vehicle'],
                input_data['uav_travel'],
                input_data['veh_distance'],
                input_data['veh_travel'],
                input_data['N'],
                input_data['N_zero'],
                input_data['N_plus'],
                input_data['A_total'],
                input_data['A_cvtp'],
                input_data['A_vtp'],
                input_data['A_aerial_relay_node'],
                input_data['G_air'],
                input_data['G_ground'],
                input_data['air_matrix'],
                input_data['ground_matrix'],
                input_data['air_node_types'],
                input_data['ground_node_types'],
                input_data['A_c'],
                input_data['xeee'])
    except KeyError as e:
        print(f"错误: 缺少必要的数据键 {str(e)}")
        return None
    except Exception as e:
        print(f"错误: 处理数据时出错: {str(e)}")
        return None