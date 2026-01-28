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
    # dataset_types = ["R201"]           # 可以是单个（会自动广播到所有实验）
    dataset_types = ["RC1_4_1"]           # 可以是单个（会自动广播到所有实验）

    # num_points_list = [100, 100, 100, 100, 100, 100, 100]
    # truck_list = [2, 1, 3, 4, 5, 3, 2]
    # uav_list = [4, 3, 6, 8, 10, 3, 8]
    # iter_list = [400, 200, 300, 400, 400, 400, 400]
    # seeds = [6, 6, 6, 6, 6, 6, 6]
    # loop_iter_list = [20, 20, 20, 10, 10, 20, 20]
    # num_points_list = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
    # num_points_list = [105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105]
    # truck_list = [1, 1, 1, 1, 1, 2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
    # truck_list = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    num_points_list = [160,160,160,160,160,160,160,160]
    truck_list = [2,6,2,4,4,4,6,6]
    uav_list = [6,6,8,8,8,12,12,24]
    iter_list = [500,500,500,500,500,500,500,500]
    seeds = [6,6,6,6,6,6,6,6]
    loop_iter_list = [10,10,10,10,10,10,10,3]
    target_ranges = [None,None,None,None,None,None,None,None]
    coord_scales = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    Z_coords = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
    uav_distance_ratios = [None, None,None,None,None,None,None,None]
    uav_distances = [20, 20,20,20,20,20,20,20]
    # uav_list = [1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 3, 6, 9, 12, 15, 4, 8, 12, 16, 20, 5, 10, 15, 20, 25]
    # uav_list = [1, 2, 3, 4, 5, 2, 4, 6, 8, 3, 6, 9, 12, 15, 4, 8, 12, 16, 20, 5, 10, 15, 20, 25]
    # iter_list = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    # seeds = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    # loop_iter_list = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    # 这些也可以单个或按组给
    # target_ranges = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    # coord_scales = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Z_coords = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    # uav_distance_ratios = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    # uav_distances = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]

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

            # "split_ratio": (15, 54, 30),  # 分别对应空中air，地面节点以及客户节点数量
            "split_ratio": (25, 84, 50),  # 分别对应空中air，地面节点以及客户节点数量

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

import hashlib
import random
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

def _set_all_seeds(seed: int):
    """锁死常用随机源，保证不同rep/算法不同seed且可复现"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def _stable_seed(base_seed: int, save_name: str, alg: str, rep_id: int) -> int:
    """稳定派生seed：同一(配置,算法,rep)永远得到同一seed"""
    key = f"{base_seed}|{save_name}|{alg}|{rep_id}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit

def _limit_threads(n: int = 1):
    """并行时强烈建议限制BLAS/OMP线程，避免CPU超卖导致更慢"""
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

def _try_limit_gurobi_threads(n: int = 1):
    """可选：限制Gurobi线程，避免进程并行×线程叠加"""
    try:
        from gurobipy import setParam
        setParam("Threads", n)
    except Exception:
        pass

def _expected_total_xlsx(output_root: str, algorithm: str, problem_name: str, run_tag: str, seed: int) -> str:
    """
    对齐 solve_mfstsp_heuristic.py 里的 export_results_to_excel 命名规则：
    problemName = f"{algorithm}__{problemName}__{run_tag}__seed{seed}"
    xlsx_path  = f"{problemName}_data_total.xlsx"
    """
    fn = f"{algorithm}__{problem_name}__{run_tag}__seed{seed}_data_total.xlsx"
    return os.path.join(output_root, "data_total", fn)

def run_one_task(base_config: dict, rep_id: int, alg: str, output_root: str):
    """
    单任务：一个配置 × 一个算法 × 一次rep
    - 强制 loop_iterations=1（因为重复已外提）
    - 生成唯一 save_name/run_tag
    - 生成唯一seed并锁随机源
    """
    cfg = deepcopy(base_config)

    base_save_name = cfg["save_name"]
    base_seed = int(cfg.get("seed", 0))
    seed = _stable_seed(base_seed, base_save_name, alg, rep_id)

    # 并行建议：限制线程
    _limit_threads(1)
    _try_limit_gurobi_threads(1)

    # 锁随机源（同时也会影响你在 main.py 里用到的 random / numpy）
    _set_all_seeds(seed)

    # 这些字段要传给 missionControl/solver
    cfg["seed"] = seed
    cfg["algorithm"] = alg
    cfg["output_root"] = output_root

    # 关键：外提重复 → 单任务只跑一次
    cfg["loop_iterations"] = 1

    # 关键：唯一 run_tag（保存隔离的核心）
    run_tag = f"{base_save_name}__ALG-{alg}__R{rep_id:02d}__S{seed}"
    cfg["save_name"] = run_tag

    # 并行跳过逻辑：如果总表xlsx已存在则跳过（最稳）
    xlsx_path = _expected_total_xlsx(output_root, alg, cfg["problem_name"], run_tag, seed)
    if cfg.get("resume_if_exists", True) and os.path.exists(xlsx_path):
        return {"status": "skip", "run_tag": run_tag, "alg": alg, "rep": rep_id, "seed": seed, "xlsx": xlsx_path}

    # 日志
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_tag}.log")

    t0 = time.time()
    try:
        _ = missionControl(config=cfg)
        dur = time.time() - t0
        msg = f"[OK] {run_tag} finished in {dur:.2f}s\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {msg}")
        return {"status": "ok", "run_tag": run_tag, "alg": alg, "rep": rep_id, "seed": seed, "sec": dur, "xlsx": xlsx_path}
    except Exception as e:
        dur = time.time() - t0
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} [FAIL] {run_tag}\n")
            f.write(str(e) + "\n")
            f.write(traceback.format_exc() + "\n")
        return {"status": "fail", "run_tag": run_tag, "alg": alg, "rep": rep_id, "seed": seed, "sec": dur, "err": str(e)}

def run_batch_experiments_parallel(n_jobs: int = 8, force_repeats: int | None = None, algorithms=None):
    """
    并行批跑：
    - n_jobs: 并行进程数（通道数）
    - force_repeats: 论文固定跑5次就传5（覆盖config里的loop_iterations）
    - algorithms: 多算法对比列表，例如 ["H_ALNS","T_ALNS"]
    """
    # 统一输出根目录（建议用绝对路径）
    output_root = os.path.abspath(r"VDAC\saved_solutions")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "data_total"), exist_ok=True)

    experiments = build_experiments()
    total_exp = len(experiments)

    if algorithms is None:
        algorithms = ["H_ALNS"]  # 默认只跑H_ALNS，你可以改成 ["H_ALNS","T_ALNS"]

    tasks = []
    for cfg in experiments:
        repeats = int(cfg.get("loop_iterations", 1))
        if force_repeats is not None:
            repeats = int(force_repeats)

        for alg in algorithms:
            for rep_id in range(repeats):
                tasks.append((cfg, rep_id, alg))

    print("==========================================")
    print(f"开始并行批量实验：配置数={total_exp} 算法数={len(algorithms)} 总任务数={len(tasks)} 并行 n_jobs={n_jobs}")
    print(f"输出根目录: {output_root}")
    print("==========================================\n")

    start_all = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_one_task)(cfg, rep_id, alg, output_root)
        for (cfg, rep_id, alg) in tasks
    )

    dur_all = time.time() - start_all
    ok = sum(r["status"] == "ok" for r in results)
    sk = sum(r["status"] == "skip" for r in results)
    fl = sum(r["status"] == "fail" for r in results)

    print("\n==========================================")
    print("并行批量实验结束。")
    print(f"总耗时: {dur_all:.2f} 秒")
    print(f"OK: {ok}  SKIP: {sk}  FAIL: {fl}")
    print(f"输出根目录: {output_root}")
    print("==========================================")

    return results


# if __name__ == "__main__":
#     run_batch_experiments()
if __name__ == "__main__":
    # 1) 用config里的 loop_iterations（你现在是10）
    run_batch_experiments_parallel(n_jobs=10)

    # 2) 论文要求每个配置固定跑5次，就用这一行（把上面那行注释掉）
    # run_batch_experiments_parallel(n_jobs=8, force_repeats=5)

    # 3) 多算法对比（示例）
    # run_batch_experiments_parallel(n_jobs=8, force_repeats=5, algorithms=["H_ALNS","T_ALNS"])
