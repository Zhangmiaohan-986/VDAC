#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import traceback
import datetime
import main
from main import missionControl
import sys
import datetime
import time
import math
from collections import defaultdict
import pandas as pd

from parseCSV import *
from parseCSVstring import *

from gurobipy import *
import os
import os.path
from subprocess import call		# allow calling an external command in python.  See http://stackoverflow.com/questions/89228/calling-an-external-command-in-python
from task_data import *
from cost_y import *
from call_function import *
from initialize import *
from cbs_plan import *
# from insert_plan import *
from down_data import *
from solve_mfstsp_heuristic import *

import distance_functions

from generate_test_problems import *
# =============================================================
startTime 		= time.time()

METERS_PER_MILE = 1609.34


UAVSpeedTypeString = {1: 'variable', 2: 'maximum', 3: 'maximum-range'}


NODE_TYPE_DEPOT	= 0
NODE_TYPE_CUST	= 1

TYPE_TRUCK 		= 1
TYPE_UAV 		= 2

# NUM_POINTS = 50
# NUM_POINTS = 100
SEED = 6
Z_COORD = 0.05  # 规划无人机空中高度情况
UAV_DISTANCE = 15

# ==============================================================================
# 【核心修复】解决 AttributeError: Can't get attribute 'make_node' on <module '__main__'>
# pickle 加载时会在当前运行的脚本(__main__)中寻找这些类定义。
# 我们必须把 main.py 里的类赋值给当前脚本的全局变量。
# ==============================================================================
make_node = main.make_node
make_vehicle = main.make_vehicle
make_travel = main.make_travel
make_dict = main.make_dict
make_assignments = main.make_assignments
make_packages = main.make_packages

# 同时确保 missionControl 也能被正常使用
from main import missionControl
import warnings

# ✅ 只屏蔽 sklearn KMeans 的 FutureWarning
warnings.filterwarnings(
    "ignore",
    message="The default value of `n_init` will change*",
    category=FutureWarning,
)

# ✅ 只屏蔽 KMeans 的 MKL memory leak 提示
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL*",
    category=UserWarning,
)

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*adjacency_matrix will return a scipy\.sparse array instead of a matrix.*"
)
# ==========================================================
# ✅ MODIFIED: 查找 saved_solutions 里是否已有对应结果（前缀匹配，兼容时间戳后缀）
# ==========================================================
def find_saved_solution_file(save_dir: str, base_name: str):
    patterns = [
        os.path.join(save_dir, base_name + ".*"),
        os.path.join(save_dir, base_name + "_*.*"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# ==========================================================
# ✅ MODIFIED: 统一生成实验列表（Grid → experiments）
# 你只改这里的“旋钮集合”，就能批量跑不同规模/类型/范围/航程
# ==========================================================
def build_experiments():
    # =========================
    # ✅ MODIFIED: 你想测试的“配对方案”
    # =========================
    dataset_types = ["R201"]           # 可以是单个（会自动广播到所有实验）
    num_points_list = [100, 100, 100]
    truck_list = [1, 3, 5]
    uav_list = [3, 6, 10]
    iter_list = [200, 300, 500]
    seeds = [6, 6, 6]
    loop_iter_list = [20, 20, 20]

    # 这些也可以单个或按组给
    target_ranges = [None, None, None]
    coord_scales = [1.0, 1.0, 1.0]
    Z_coords = [0.05, 0.05, 0.05]
    uav_distance_ratios = [None, None, None]
    uav_distances = [15, 15, 15]

    # =========================
    # ✅ MODIFIED: 对齐/广播工具
    # =========================
    def _pick(lst, i, L, name):
        if len(lst) == 1:
            return lst[0]
        if len(lst) == L:
            return lst[i]
        raise ValueError(f"[build_experiments] '{name}' 长度必须为 1 或 {L}，但现在是 {len(lst)}")

    L = max(
        len(num_points_list), len(truck_list), len(uav_list), len(iter_list), len(seeds), len(loop_iter_list),
        len(dataset_types), len(target_ranges), len(coord_scales), len(Z_coords),
        len(uav_distance_ratios), len(uav_distances)
    )

    experiments = []
    for i in range(L):
        ds   = _pick(dataset_types, i, L, "dataset_types")
        n    = _pick(num_points_list, i, L, "num_points_list")
        nt   = _pick(truck_list, i, L, "truck_list")
        nu   = _pick(uav_list, i, L, "uav_list")
        iters= _pick(iter_list, i, L, "iter_list")
        seed = _pick(seeds, i, L, "seeds")
        loop_iters = _pick(loop_iter_list, i, L, "loop_iter_list")
        tr   = _pick(target_ranges, i, L, "target_ranges")
        cs   = _pick(coord_scales, i, L, "coord_scales")
        zdr  = _pick(Z_coords, i, L, "Z_coords")
        ratio= _pick(uav_distance_ratios, i, L, "uav_distance_ratios")
        ud   = _pick(uav_distances, i, L, "uav_distances")

        # ✅ MODIFIED: 自动命名（只为这一组生成）
        tag_tr = "raw" if tr is None else f"rng{tr[0]}x{tr[1]}"
        tag_ratio = "ratioNA" if ratio is None else f"ratio{ratio:.3f}"
        save_name = (
            f"{ds}_N{n}_T{nt}_U{nu}_I{iters}_L{loop_iters}_S{seed}_"
            f"{tag_tr}_scale{cs}_{tag_ratio}_Z{zdr}"
        )

        cfg = {
            "problem_name": f"case_{nu}UAV_{nt}Truck",
            "save_name": save_name,
            "iterations": iters,
            "loop_iterations": loop_iters,

            "num_trucks": nt,
            "num_uavs": nu,
            "max_drones": 10,
            "per_uav_cost": 1,
            "per_vehicle_cost": 2,

            "early_arrival_cost": [5, 0.083],
            "late_arrival_cost": [20, 0.333],

            "num_points": n,
            "seed": seed,

            "dataset_type": ds,
            "target_range": tr,
            "coord_scale": cs,

            "Z_coord": zdr,
            "uav_distance": ud,
            "uav_distance_ratio": ratio,

            "split_ratio": (1/3, 1/3, 1/3),

            "resume_if_exists": True,
        }

        experiments.append(cfg)

    return experiments


def run_batch_experiments():
    save_dir = r"VDAC\saved_solutions"
    os.makedirs(save_dir, exist_ok=True)

    experiments = build_experiments()
    total_exp = len(experiments)

    print("==========================================")
    print(f"开始批量实验，共计 {total_exp} 个任务")
    print("==========================================\n")

    start_time_all = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, config in enumerate(experiments, start=1):
        save_name = config["save_name"]
        print("------------------------------------------")
        print(f"任务 [{idx}/{total_exp}]: {save_name}")
        print(f"数据={config['dataset_type']}  点数={config['num_points']}  卡车={config['num_trucks']}  无人机={config['num_uavs']}  迭代={config['iterations']}")
        print(f"range={config['target_range']}  scale={config['coord_scale']}  Z={config['Z_coord']}  uav_dist={config['uav_distance']}  ratio={config['uav_distance_ratio']}")
        print("------------------------------------------")

        # ✅ MODIFIED: 如果启用复用，且已存在结果文件 → 直接跳过（或者你也可以选择“加载后继续跑”）
        if config.get("resume_if_exists", True):
            hit = find_saved_solution_file(save_dir, save_name)
            if hit is not None:
                print(f"[SKIP] 已存在结果文件：{hit}")
                skip_count += 1
                continue

        # ✅ MODIFIED: 每个实验单独日志文件（失败也能定位）
        log_path = os.path.join(save_dir, f"{save_name}.log")

        # try:
        t0 = time.time()
        app = missionControl(config=config)
        t1 = time.time()

        msg = f"[OK] {save_name} finished in {t1 - t0:.2f}s\n"
        print(msg)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {msg}")

        success_count += 1

        # except Exception as e:
        #     fail_count += 1
        #     print(f"[FAIL] {save_name}")
        #     print(f"错误信息: {str(e)}")
        #     traceback.print_exc()

        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write(f"{datetime.datetime.now().isoformat()} [FAIL] {save_name}\n")
        #         f.write(str(e) + "\n")
        #         f.write(traceback.format_exc() + "\n")

        #     # 继续下一个
        #     continue

    duration = time.time() - start_time_all

    print("==========================================")
    print("批量实验结束。")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"成功: {success_count}, 跳过(已存在): {skip_count}, 失败: {fail_count}")
    print(f"结果目录: {save_dir}")
    print("==========================================")


if __name__ == "__main__":
    run_batch_experiments()
