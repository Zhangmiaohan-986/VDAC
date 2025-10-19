# VDAC可视化工具包 - 快速使用指南

## 🚀 一键可视化 - 最简单的使用方法

在您的代码中只需要**一行代码**就能看到可视化结果：

```python
from debug_visualize import debug_viz
debug_viz(your_state)  # 就这么简单！
```

## 📁 文件说明

| 文件名 | 用途 | 推荐使用场景 |
|--------|------|-------------|
| `debug_visualize.py` | **一键可视化** | 日常调试，最简单 |
| `quick_visualize.py` | 快速可视化 | 需要更多功能时 |
| `visualize_from_state.py` | 状态对象可视化 | 从状态对象直接可视化 |
| `visualization.py` | 核心可视化类 | 高级用户自定义 |
| `example_usage.py` | 使用示例 | 学习如何使用 |

## 🎯 核心函数

### 1. `debug_viz(state, title="调试可视化")`
**最推荐使用！** 一行代码搞定可视化。

```python
from debug_visualize import debug_viz

# 在您的代码中
debug_viz(my_state, "当前解决方案")
```

### 2. `debug_compare(state1, state2, title1, title2)`
比较两个解决方案。

```python
from debug_visualize import debug_compare

debug_compare(state_before, state_after, "优化前", "优化后")
```

### 3. `debug_info(state, label="状态信息")`
只显示文本信息，不显示图形。

```python
from debug_visualize import debug_info

debug_info(my_state, "当前状态分析")
```

## 📊 数据格式

您的数据需要符合以下格式：

### 方法1：使用FastMfstspState对象（推荐）
```python
# 直接传入您的状态对象
debug_viz(your_fast_mfstsp_state)
```

### 方法2：使用字典格式
```python
my_data = {
    'vehicle_routes': [
        [1, 2, 3, 4, 5],  # 车辆1的路线
        [1, 6, 7, 8, 9],  # 车辆2的路线
    ],
    'customer_plan': {
        customer_id: (drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle)
    },
    'uav_assignments': {
        drone_id: [(drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle), ...]
    }
}

debug_viz(my_data)
```

## 🎨 可视化元素说明

- **🔴 红色方形** - 仓库（起始点）
- **🔵 蓝色三角形** - VTP节点（发射/回收点）
- **🟢 绿色圆形** - 客户点
- **实线箭头** - 车辆地面路线
- **虚线箭头** - 无人机空中路线

## 💡 使用场景

### 1. 日常调试
```python
# 在您的求解循环中
for iteration in range(max_iterations):
    # ... 您的优化代码 ...
    
    if iteration % 100 == 0:
        debug_viz(current_state, f"第{iteration}次迭代")
```

### 2. 结果检查
```python
# 检查最终结果
debug_viz(final_solution, "最终解决方案")
```

### 3. 方案比较
```python
# 比较不同算法结果
debug_compare(greedy_solution, alns_solution, "贪心算法", "ALNS算法")
```

### 4. 快速信息查看
```python
# 只查看统计信息，不显示图形
debug_info(current_state, "当前状态统计")
```

## 🔧 安装依赖

```bash
pip install matplotlib numpy networkx
```

## 🚀 快速开始

1. **导入工具**：
   ```python
   from debug_visualize import debug_viz
   ```

2. **在您的代码中调用**：
   ```python
   debug_viz(your_state)
   ```

3. **查看结果**：图形会自动显示，控制台会输出详细信息

## 📝 完整示例

```python
#!/usr/bin/env python3
from debug_visualize import debug_viz, debug_compare, debug_info

# 您的VDAC求解代码
def solve_vdac():
    # ... 您的求解逻辑 ...
    
    # 检查初始状态
    debug_viz(initial_state, "初始状态")
    
    # 优化过程
    for iteration in range(max_iterations):
        # ... 优化代码 ...
        
        if iteration % 100 == 0:
            debug_info(current_state, f"第{iteration}次迭代")
    
    # 检查最终结果
    debug_viz(final_state, "最终解决方案")
    
    # 比较优化前后
    debug_compare(initial_state, final_state, "优化前", "优化后")

if __name__ == "__main__":
    solve_vdac()
```

## ❓ 常见问题

### Q: 图形不显示怎么办？
A: 确保安装了matplotlib：`pip install matplotlib`

### Q: 中文显示乱码怎么办？
A: 系统会自动处理中文字体，如果还有问题可以修改代码中的字体设置

### Q: 数据格式不对怎么办？
A: 检查您的数据是否符合要求的格式，或者直接使用FastMfstspState对象

### Q: 如何自定义图形大小？
A: 使用`quick_visualize`函数，它支持`figsize`参数

## 🎉 总结

这个可视化工具包让您能够：
- **一行代码**完成可视化
- **快速检查**解决方案的正确性
- **比较不同**的解决方案
- **调试优化**过程

只需要记住一个函数：`debug_viz(your_state)`，就能解决大部分可视化需求！

