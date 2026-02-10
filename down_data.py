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


# ✅ MODIFIED: 固定保存目录（你指定的那个）
# SAVE_DIR_ABS = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"
SAVE_DIR_ABS = r"D:\NKU\VDAC_PAP\VDAC\saved_solutions"

# ✅ MODIFIED: 这些工具函数如果你原来就有，就保留你的；没有就用我这份最小实现
def _convert_numpy_types(obj):
    """把 numpy 标量/数组转成 python 可pickle/可json的类型（pickle其实不强制，但保持一致）"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return { _convert_numpy_types(k): _convert_numpy_types(v) for k, v in obj.items() }
    if isinstance(obj, (list, tuple)):
        t = [_convert_numpy_types(x) for x in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj


def _convert_defaultdict_to_dict(dd):
    """defaultdict -> dict（递归）"""
    if isinstance(dd, defaultdict):
        dd = dict(dd)
    if isinstance(dd, dict):
        return {k: _convert_defaultdict_to_dict(v) for k, v in dd.items()}
    if isinstance(dd, list):
        return [_convert_defaultdict_to_dict(x) for x in dd]
    if isinstance(dd, tuple):
        return tuple(_convert_defaultdict_to_dict(x) for x in dd)
    return dd


def _convert_key_to_str(obj):
    """某些 key 不是基础类型时，转成 str（避免某些序列化/跨版本问题）"""
    if isinstance(obj, dict):
        return {str(k): _convert_key_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_key_to_str(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_key_to_str(x) for x in obj)
    return obj


# ✅ MODIFIED: 专门识别 vehicle_task / node 等对象：不要把它们变 dict（否则 fast_copy 没了）
def _is_vehicle_task(obj):
    return hasattr(obj, "__class__") and obj.__class__.__name__ == "vehicle_task"


def _is_make_node(obj):
    return hasattr(obj, "__class__") and obj.__class__.__name__ in ("make_node", "Node", "node")  # 你项目里实际类名可能是 make_node


def _safe_for_regular(obj):
    """
    ✅ MODIFIED:
    regular 情况下尽量把 numpy 转掉、defaultdict 转掉、key转str；
    但遇到 vehicle_task / make_node 这类“需要保持对象”的，就原样返回（pickle能存对象）。
    """
    if _is_vehicle_task(obj) or _is_make_node(obj):
        return obj
    if isinstance(obj, nx.Graph):
        # Graph 不走 regular，外层会单独处理
        return obj
    obj = _convert_defaultdict_to_dict(obj)
    obj = _convert_key_to_str(obj)
    obj = _convert_numpy_types(obj)
    return obj


SAVE_DIR_ABS = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"

# 假设这是你的绝对路径
SAVE_DIR_ABS = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"

def save_input_data_with_name(input_data, custom_name):
    """
    ✅ 最终融合版保存函数：
    1. 彻底解决 pickle lambda 报错 (递归清洗 defaultdict)。
    2. 集成 numpy 类型转换，确保数据纯净。
    3. 保留 vehicle_task/node 等核心对象为实例(Object)，确保方法可用。
    """

    # -------- 内部核心：数据清洗函数 --------
    def _sanitize_data(obj):
        """
        递归处理数据：
        1. defaultdict -> dict (移除 lambda)
        2. numpy -> int/float/list (兼容性)
        3. list/dict -> 递归处理
        4. 其他对象 (VehicleTask等) -> 保持原样
        """
        # 1. 处理 Numpy 类型 (引用了你提供的逻辑)
        if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool): # 注意：bool是int的子类
             return int(obj)
        elif isinstance(obj, (np.floating, float)):
             return float(obj)
        elif isinstance(obj, np.ndarray):
             return [_sanitize_data(x) for x in obj.tolist()]

        # 2. 处理 Defaultdict (这是解决报错的关键)
        elif isinstance(obj, defaultdict):
            return {k: _sanitize_data(v) for k, v in obj.items()}

        # 3. 处理普通 Dict
        elif isinstance(obj, dict):
            return {k: _sanitize_data(v) for k, v in obj.items()}

        # 4. 处理 List / Tuple
        elif isinstance(obj, list):
            return [_sanitize_data(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(_sanitize_data(v) for v in obj)

        # 5. 其他自定义对象 (VehicleTask, Node 等)
        # 直接返回对象本身！Pickle 会自动保存它的类结构和属性。
        else:
            return obj

    # -------- 文件名规范化 --------
    if not custom_name.endswith(".pkl"):
        custom_name += ".pkl"
    if not custom_name.startswith("input_data_"):
        custom_name = "input_data_" + custom_name

    save_dir = SAVE_DIR_ABS
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, custom_name)

    # -------- 执行保存 --------
    try:
        print(f"[SAVE] 正在预处理数据 (清洗 lambda 和 numpy)...")
        serializable_data = {}

        for key, value in input_data.items():
            # 特殊处理 NetworkX (它不能简单的递归清洗，用它自带的方法)
            if isinstance(value, nx.Graph):
                # node_link_data 返回的是 dict，我们也清洗一下里面的 numpy
                graph_data = nx.node_link_data(value)
                serializable_data[key] = {
                    'type': 'networkx_graph',
                    'data': _sanitize_data(graph_data)
                }
            
            # 特殊标记 defaultdict (方便加载时看日志，其实内容已经是 dict 了)
            elif isinstance(value, defaultdict):
                serializable_data[key] = {
                    'type': 'defaultdict',
                    'data': _sanitize_data(value)
                }
            
            # 关键对象字典 (vehicle, node) - 标记为 object_dict
            elif key in ['node', 'vehicle']:
                serializable_data[key] = {
                    'type': 'object_dict',
                    'data': _sanitize_data(value) # 递归进去清洗可能存在的 numpy，但对象本身保留
                }
            
            # 你的任务字典
            elif key in ['vehicle_task_data', 'uav_task_dict']:
                serializable_data[key] = {
                    'type': 'task_data', # 标记一下
                    'data': _sanitize_data(value) # 这里的 _sanitize 会把 defaultdict 剥离
                }

            # 普通数据
            else:
                serializable_data[key] = {
                    'type': 'regular',
                    'data': _sanitize_data(value)
                }

        print(f"[SAVE] 正在写入文件: {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(serializable_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ 数据保存成功: {custom_name}")
        return custom_name.replace(".pkl", "")

    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")
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


# def load_input_data(filename, save_dir=r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"):
#     """加载保存的数据（支持 base_name / 带前缀名 / 带后缀名 / 直接完整路径）"""

#     # ✅ MODIFIED: 统一绝对目录
#     save_dir = os.path.abspath(save_dir)

#     # ✅ MODIFIED: 如果传入的是完整路径，直接使用
#     if isinstance(filename, str) and os.path.exists(filename):
#         file_path = filename
#     else:
#         name = filename

#         # ✅ MODIFIED: 去掉后缀（如果用户传了 .pkl）
#         if name.endswith(".pkl"):
#             name = name[:-4]

#         # ✅ MODIFIED: 正确加前缀（绝对不能写成 filename='filename'）
#         if not name.startswith("input_data_"):
#             name = "input_data_" + name

#         file_path = os.path.join(save_dir, name + ".pkl")

#     # ✅ MODIFIED: 明确提示文件到底找没找到
#     if not os.path.exists(file_path):
#         print(f"[LOAD] 文件不存在: {file_path}")
#         return None

#     try:
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)

#         restored_data = {}
#         for key, value in data.items():
#             if value.get('type') == 'defaultdict':
#                 restored_data[key] = defaultdict(list, create_object_from_dict(value['data']))
#             else:
#                 restored_data[key] = create_object_from_dict(value['data'])

#         print(f"[LOAD] 成功加载: {file_path}")
#         return restored_data

#     except Exception as e:
#         print(f"加载数据时出错: {str(e)}")
#         print("详细错误信息:")
#         import traceback
#         print(traceback.format_exc())
#         return None

# def load_input_data(filename):  # 不行版本
#     """加载输入数据"""
#     # 你的保存目录
#     save_dir = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"
    
#     # === [修改开始] ===
#     # 你的实际文件名前面有 "input_data_"，但传入的 filename 参数(instance_name)没有
#     # 所以这里需要手动加上 "input_data_" 前缀
    
#     # 逻辑判断：为了防止 filename 变量里已经包含了 input_data_ 导致重复，做一个判断
#     if filename.startswith("input_data_"):
#         real_filename = f'{filename}.pkl'
#     else:
#         real_filename = f'input_data_{filename}.pkl'
        
#     pickle_path = os.path.join(save_dir, real_filename)
#     # === [修改结束] ===

#     # 为了方便调试，建议打印一下最终尝试读取的路径
#     print(f"正在尝试加载文件: {pickle_path}")

#     # 检查文件是否存在，如果不存在提前报错并给出提示
#     if not os.path.exists(pickle_path):
#         raise FileNotFoundError(f"无法找到文件: {pickle_path}，请检查 saved_solutions 目录下是否有该文件。")

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
#         # elif value['type'] == 'object_dict':
#         #     restored_data[key] = {k: create_object_from_dict(v, key) for k, v in value['data'].items()}
#         else:
#             # 注意：如果 value['type'] == 'object_dict' 逻辑是你自己写的，请保留
#             # 这里为了通用性，我处理了 else 情况
#             restored_data[key] = value['data']
    
#     return restored_data

def load_input_data(filename):
    """
    ✅ 修复版加载函数：
    1. 自动处理文件名后缀和前缀
    2. 仅对 defaultdict 和 nx.Graph 做特殊还原
    3. 对于 node/vehicle/task，直接返回 pickle 还原的对象（保留了方法）
    """
    
    # save_dir = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"
    save_dir = r"D:\NKU\VDAC_PAP\VDAC\saved_solutions"

    # -------- 1. 路径与文件名解析 --------
    # 如果传入的是完整路径且存在，直接用
    if os.path.isabs(filename) and os.path.exists(filename):
        pickle_path = filename
    else:
        # 清理文件名
        name = filename
        if name.endswith(".pkl"):
            name = name[:-4]
        if not name.startswith("input_data_"):
            name = "input_data_" + name
            
        pickle_path = os.path.join(save_dir, name + ".pkl")

    print(f"[LOAD] 正在读取文件: {pickle_path}")

    if not os.path.exists(pickle_path):
        print(f"[LOAD][ERROR] 文件不存在: {pickle_path}")
        return None

    # -------- 2. 读取与结构还原 --------
    try:
        with open(pickle_path, "rb") as f:
            loaded_wrapper = pickle.load(f)

        restored_data = {}

        for key, wrapper in loaded_wrapper.items():
            # 兼容性检查：确保是我们封装的 {'type':..., 'data':...} 格式
            if isinstance(wrapper, dict) and 'type' in wrapper and 'data' in wrapper:
                data_type = wrapper['type']
                raw_data = wrapper['data']
                
                if data_type == 'defaultdict':
                    # 还原 defaultdict
                    dd = defaultdict(lambda: None)
                    dd.update(raw_data)
                    restored_data[key] = dd
                    
                elif data_type == 'networkx_graph':
                    # 还原 NetworkX 图
                    restored_data[key] = nx.node_link_graph(raw_data)
                    
                elif data_type == 'object_dict':
                    # ✅ 关键修正：
                    # 因为保存时直接存了对象，pickle 自动还原了它们。
                    # 这里不需要 create_object_from_dict，直接用 raw_data 即可。
                    restored_data[key] = raw_data
                    
                else:
                    # regular 或其他类型，直接使用
                    restored_data[key] = raw_data
            else:
                # 假如读到了旧版数据（没有 type/data 包装），直接赋值
                restored_data[key] = wrapper

        print(f"[LOAD] ✅ 成功加载数据，对象方法已保留。")
        return restored_data

    except Exception as e:
        print(f"[LOAD][ERROR] 加载失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# def load_input_data(filename):
#     """加载输入数据"""
#     # save_dir = 'saved_solutions'
#     save_dir = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions"
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