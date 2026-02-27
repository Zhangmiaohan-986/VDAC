#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
串行单次实验脚本（不并行）
用途：
1) 用单进程顺序运行各对比算法；
2) 每个配置只运行 1 次（用于快速验证算法逻辑）；
3) 生成轻量 .done / .error 标记文件，便于追踪运行状态。
"""

import os
import time
import traceback
import datetime
import warnings

# 引入你的自定义模块
import main
from main import missionControl
from task_data import *
from initialize import *


# =============================================================
# 全局配置
# =============================================================
# 单次运行（固定每个任务只跑一次）
ALGO_SEED_BASE = 30000
ENABLE_MARKER_SKIP = False  # True: 若存在.done则跳过；False: 总是重跑

# 标记文件目录
# MARKER_BASE_DIR = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions\markers_single"
MARKER_BASE_DIR = r"/Users/zhangmiaohan/猫咪存储文件/maomi_github/VDAC/saved_solutions"


# 可选：仅测试一个算法（设为 None 则按列表全跑）
# ONLY_ALGORITHM = None
# ONLY_ALGORITHM = "DAI_ALNS"  # 例如只测试 DA_I_ALNS
# ONLY_ALGORITHM = "TA_I_ALNS"  # 例如只测试 DA_I_ALNS
# ONLY_ALGORITHM = "A_I_ALNS"  # 例如只测试 DA_I_ALNS
# ONLY_ALGORITHM = "H_ALNS"  # 例如只测试 DA_I_ALNS
ONLY_ALGORITHM = "T_ALNS"  # 例如只测试 DA_I_ALNS
# ONLY_ALGORITHM = "T_I_ALNS"  # 例如只测试 DA_I_ALNS




# =============================================================
# 对比算法列表
# =============================================================
ALGORITHMS_TO_COMPARE = [
    "H_ALNS",
    "T_ALNS",
    "T_I_ALNS",
    "TA_I_ALNS",
    "A_I_ALNS",
    "DAI_ALNS",
]

ALG_ABBR = {
    "H_ALNS": "HA",
    "T_ALNS": "TA",
    "T_I_ALNS": "TI",
    "TA_I_ALNS": "TAI",
    "A_I_ALNS": "AI",
    "DAI_ALNS": "DAI",
}


# =============================================================
# 忽略警告
# =============================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================================
# 工具函数
# ==========================================================
def _pick(lst, i, L, name):
    if len(lst) == 1:
        return lst[0]
    if len(lst) == L:
        return lst[i]
    raise ValueError(f"[build_experiments] '{name}' 长度必须为 1 或 {L}")


def get_marker_path(cfg):
    """
    根据配置生成唯一标记文件路径
    文件名包含: 算法名_车辆_无人机_随机种子.done
    """
    os.makedirs(MARKER_BASE_DIR, exist_ok=True)

    alg = cfg.get("algorithm_name", "UnknownAlg")
    nt = cfg.get("num_trucks")
    nu = cfg.get("num_uavs")
    run_tag = cfg.get("run_tag", "single")

    alg_short = ALG_ABBR.get(alg, alg)
    filename = f"{alg_short}_T{nt}_U{nu}_{run_tag}.done"
    return os.path.join(MARKER_BASE_DIR, filename)


# ==========================================================
# 构建实验列表
# ==========================================================
def build_experiments():
    dataset_types = ["RC1_4_1"]
    # dataset_types = ["RC101"]

    # 客户节点为15的情况下配置配比------------------------------
    # num_points_list = [60]
    # truck_list = [3]
    # uav_list = [6]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [60]
    # truck_list = [1]
    # uav_list = [4]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [60]
    # truck_list = [3]
    # uav_list = [6]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [60]
    # truck_list = [4]
    # uav_list = [4]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 客户节点为30的情况下配置配比--------------------------------------
    # 对比实验分割线-----------------------------
    # num_points_list = [100]
    # truck_list = [2]
    # uav_list = [6]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [100]
    # truck_list = [4]
    # uav_list = [8]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [100]
    # truck_list = [6]
    # uav_list = [6]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 客户节点为50的情况下配置配比---------------------------------
    # 对比实验分割线-----------------------------
    # num_points_list = [165]
    # truck_list = [2]
    # uav_list = [8]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [165]
    # truck_list = [5]
    # uav_list = [10]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [165]
    # truck_list = [8]
    # uav_list = [8]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 客户节点为100的情况下配置配比---------------------------------------
    # 对比实验分割线-----------------------------
    # num_points_list = [335]
    # truck_list = [8]
    # uav_list = [16]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [335]
    # truck_list = [7]
    # uav_list = [14]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    # 对比实验分割线-----------------------------
    # num_points_list = [335]
    # truck_list = [10]
    # uav_list = [15]
    # iter_list = [500]
    # seeds = [6]
    # loop_iter_list = [1]
    num_points_list = [335]
    truck_list = [5]
    uav_list = [20]
    iter_list = [500]
    seeds = [6]
    loop_iter_list = [1]

    L = max(len(num_points_list), len(truck_list), len(uav_list), len(iter_list), len(seeds))

    experiments = []
    for i in range(L):
        ds = _pick(dataset_types, i, L, "dataset_types")
        n = _pick(num_points_list, i, L, "num_points_list")
        nt = _pick(truck_list, i, L, "truck_list")
        nu = _pick(uav_list, i, L, "uav_list")
        iters = _pick(iter_list, i, L, "iter_list")
        seed = _pick(seeds, i, L, "seeds")
        loop_iters = _pick(loop_iter_list, i, L, "loop_iter_list")

        base_save_name = f"N{n}_T{nt}_U{nu}"

        cfg = {
            "num_trucks": nt,
            "num_uavs": nu,
            "num_points": n,
            "iterations": iters,
            "loop_iterations": loop_iters,
            "seed": seed,
            "dataset_type": ds,
            "target_range": None,
            "coord_scale": 1.0,
            "Z_coord": 0.05,
            "uav_distance": 20,
            "uav_distance_ratio": None,
            # "split_ratio": (10, 24, 15),  # 处理15个节点的情况
            # "split_ratio": (15, 54, 30), # 分别对应空中air，地面节点以及客户节点数量。 对应30客户节点情况下，100nodes
            # "split_ratio": (25, 89, 50), # 分别对应空中air，地面节点以及客户节点数量。对应50客户节点情况下，150nodes
            # "split_ratio": (10, 34, 15),  # 分别对应空中air，地面节点以及客户节点数量。对应15客户节点情况下，50nodes
            "split_ratio": (35, 199, 100), # 分别对应空中air，地面节点以及客户节点数量。对应100客户节点情况下， 300nodes

            "max_drones": 10,
            "per_uav_cost": 1,
            "per_vehicle_cost": 2,
            "early_arrival_cost": [5, 0.083],
            "late_arrival_cost": [20, 0.333],
            "resume_if_exists": False,
        }

        for algo_name in ALGORITHMS_TO_COMPARE:
            if ONLY_ALGORITHM and algo_name != ONLY_ALGORITHM:
                continue

            cfg2 = dict(cfg)
            cfg2["algorithm_name"] = algo_name

            alg_short = ALG_ABBR.get(algo_name, algo_name)
            save_name_with_alg = f"{base_save_name}_{alg_short}"

            cfg2["save_name"] = save_name_with_alg
            cfg2["problem_name"] = f"Prob_{n}C_{nt}T_{nu}U_{alg_short}"

            experiments.append(cfg2)

    return experiments


def _make_run_config(base_cfg, exp_idx):
    """
    单次运行配置（无重复rep）
    """
    cfg = dict(base_cfg)
    algo_seed = ALGO_SEED_BASE + exp_idx * 100
    cfg["algo_seed"] = algo_seed
    run_tag = f"e{exp_idx}_r0_a{algo_seed}"
    cfg["run_tag"] = run_tag
    cfg["save_name"] = f"{base_cfg['save_name']}_{run_tag}"
    cfg["problem_name"] = f"{base_cfg['problem_name']}_{run_tag}"
    return cfg


# ==========================================================
# 串行执行（核心）
# ==========================================================
def run_single_task(cfg):
    """
    单任务执行：
    - 调 missionControl 运行算法
    - 写 .done / .error 标记文件
    """
    marker_path = get_marker_path(cfg)
    save_name = cfg.get("save_name", "unknown")

    # try:
    start = time.time()
    results_all = missionControl(config=cfg)
    elapsed = time.time() - start
    print(f"  -> Elapsed: {elapsed:.2f} seconds")

    #     if not results_all:
    #         print(f"❌ [Fail] {save_name} 结果为空")
    #         error_marker = marker_path.replace(".done", ".error")
    #         with open(error_marker, "w", encoding="utf-8") as f:
    #             f.write("Result is empty.\n")
    #         return False

    #     with open(marker_path, "w", encoding="utf-8") as f:
    #         f.write(f"Task Completed at: {datetime.datetime.now().isoformat()}\n")
    #         f.write(f"Algorithm: {cfg['algorithm_name']}\n")
    #         f.write(f"Trucks: {cfg['num_trucks']}, UAVs: {cfg['num_uavs']}\n")
    #         f.write(f"Seed: {cfg['algo_seed']}\n")
    #         f.write(f"Elapsed: {elapsed:.2f}s\n")
    #         f.write("-" * 20 + "\n")
    #         f.write("Run Successful.\n")

    #     return True

    # except Exception as e:
    #     print(f"🔥 [Error] {save_name}")
    #     error_marker = marker_path.replace(".done", ".error")
    #     with open(error_marker, "w", encoding="utf-8") as f:
    #         f.write(str(e))
    #         f.write("\n")
    #         f.write(traceback.format_exc())
    #     traceback.print_exc()
    #     return False


def run_serial_experiments():
    os.makedirs(MARKER_BASE_DIR, exist_ok=True)
    experiments = build_experiments()

    print("==========================================")
    print("开始串行单次实验 (No Parallel)")
    print(f"任务总数: {len(experiments)}")
    print(f"标记目录: {MARKER_BASE_DIR}")
    print(f"跳过开关: {ENABLE_MARKER_SKIP}")
    if ONLY_ALGORITHM:
        print(f"仅运行算法: {ONLY_ALGORITHM}")
    print("==========================================\n")

    start_all = time.time()
    run_count = 0
    skip_count = 0
    ok_count = 0
    fail_count = 0

    for idx, base_cfg in enumerate(experiments, start=1):
        cfg = _make_run_config(base_cfg, idx - 1)
        marker_path = get_marker_path(cfg)

        print(f"[{idx}/{len(experiments)}] {cfg['algorithm_name']} | {cfg['num_trucks']}T-{cfg['num_uavs']}U | {cfg['run_tag']}")

        if ENABLE_MARKER_SKIP and os.path.exists(marker_path):
            print("  -> Skip (done marker exists)")
            skip_count += 1
            continue

        run_count += 1
        ok = run_single_task(cfg)
        if ok:
            ok_count += 1
            print("  -> Done")
        else:
            fail_count += 1
            print("  -> Failed")

    elapsed_all = time.time() - start_all
    print("\n==========================================")
    print("所有串行任务结束")
    print(f"实际运行: {run_count}")
    print(f"成功: {ok_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {skip_count}")
    print(f"总耗时: {elapsed_all:.2f} 秒")
    print("==========================================")


if __name__ == "__main__":
    run_serial_experiments()

