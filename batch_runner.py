#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import traceback
import datetime
import sys
import math
from collections import defaultdict
import pandas as pd
from multiprocessing import Process, Queue
import warnings

# 引入你的自定义模块
import main
from main import missionControl
from task_data import *
# from call_function import export_results_to_excel #如果不保存结果，这行其实可以注释掉
from initialize import *

# =============================================================
# 全局配置
# =============================================================
startTime = time.time()

# 并行配置
REPEAT_PER_TASK = 10  # 每个算法配置跑多少次取平均
MAX_PARALLEL = 5      # 最大并行进程数
ALGO_SEED_BASE = 20000 

# ✅ 定义保存标记文件的根目录
# 这个目录只存轻量级的 .done 文件，方便快速检查
MARKER_BASE_DIR = r"D:\Zhangmiaohan_Palace\VDAC_基于空中走廊的配送任务研究\VDAC\saved_solutions\markers"

# =============================================================
# 对比算法列表
# =============================================================
ALGORITHMS_TO_COMPARE = [
    "H_ALNS", 
    "T_ALNS",
    "T_I_ALNS",
    "TA_I_ALNS",
    "A_I_ALNS",
    "DA_I_ALNS",
]

ALG_ABBR = {
    "H_ALNS": "HA",
    "T_ALNS": "TA",
    "T_I_ALNS": "TI",
    "TA_I_ALNS": "TAI",
    "A_I_ALNS": "AI",
    "DA_I_ALNS": "DAI",
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
    根据配置生成唯一的标记文件路径
    文件名包含: 算法名_车辆_无人机_随机种子.done
    """
    # 确保目录存在
    os.makedirs(MARKER_BASE_DIR, exist_ok=True)
    
    # 获取关键参数
    alg = cfg.get('algorithm_name', 'UnknownAlg')
    nt = cfg.get('num_trucks')
    nu = cfg.get('num_uavs')
    run_tag = cfg.get('run_tag', 'unknown_tag') # 包含了 rep 和 seed 信息
    
    # 生成唯一文件名，例如: HALNS_T3_U6_e0_r0_a20000.done
    alg_short = ALG_ABBR.get(alg, alg)
    filename = f"{alg_short}_T{nt}_U{nu}_{run_tag}.done"
    
    return os.path.join(MARKER_BASE_DIR, filename)

# ==========================================================
# 构建实验列表 (逻辑不变)
# ==========================================================
def build_experiments():
    dataset_types = ["RC1_4_1"] 
    num_points_list = [50]      
    truck_list = [3]            
    uav_list = [6]              
    iter_list = [500]           
    seeds = [6]
    loop_iter_list = [1]        
    target_ranges = [None]
    coord_scales = [1.0]
    Z_coords = [0.05]
    uav_distance_ratios = [None]
    uav_distances = [20]

    L = max(len(num_points_list), len(truck_list), len(uav_list), len(iter_list), len(seeds))

    experiments = []
    for i in range(L):
        ds   = _pick(dataset_types, i, L, "dataset_types")
        n    = _pick(num_points_list, i, L, "num_points_list")
        nt   = _pick(truck_list, i, L, "truck_list")
        nu   = _pick(uav_list, i, L, "uav_list")
        iters= _pick(iter_list, i, L, "iter_list")
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
            "split_ratio": (25, 89, 50),
            "max_drones": 10,
            "per_uav_cost": 1,
            "per_vehicle_cost": 2,
            "early_arrival_cost": [5, 0.083],
            "late_arrival_cost": [20, 0.333],
            "resume_if_exists": False, 
        }

        for algo_name in ALGORITHMS_TO_COMPARE:
            cfg2 = dict(cfg)
            cfg2["algorithm_name"] = algo_name 
            
            alg_short = ALG_ABBR.get(algo_name, algo_name)
            save_name_with_alg = f"{base_save_name}_{alg_short}"
            
            cfg2["save_name"] = save_name_with_alg
            cfg2["problem_name"] = f"Prob_{n}C_{nt}T_{nu}U_{alg_short}"
            
            experiments.append(cfg2)

    return experiments

def _make_run_config(base_cfg, exp_idx, rep_idx):
    cfg = dict(base_cfg)
    algo_seed = ALGO_SEED_BASE + exp_idx * 100 + rep_idx
    cfg["algo_seed"] = algo_seed
    run_tag = f"e{exp_idx}_r{rep_idx}_a{algo_seed}"
    cfg["run_tag"] = run_tag
    cfg["save_name"] = f"{base_cfg['save_name']}_{run_tag}"
    cfg["problem_name"] = f"{base_cfg['problem_name']}_{run_tag}"
    return cfg

# ==========================================================
# ✅ 核心修改：Worker 生成 Log 标记文件，不存 Excel
# ==========================================================
def _worker(cfg):
    save_name = cfg.get('save_name', 'unknown')
    marker_path = get_marker_path(cfg) # 获取标记文件路径
    
    try:
        # print(f"--> [Worker Start] PID:{os.getpid()} 处理: {os.path.basename(marker_path)}")
        
        # 1. 执行任务 (耗时操作)
        # 即使不保存结果，也要跑一遍算法来验证流程或测试性能
        results_all = missionControl(config=cfg)
        
        # 2. 验证运行是否成功
        if not results_all:
            print(f"❌ [Fail] PID:{os.getpid()} {save_name} 结果为空")
            return

        # 3. ✅ 【关键】生成 Log/标记文件 (代替 export_results_to_excel)
        # 内容可以包含简要的性能指标，方便后续查看，而不需要打开 Excel
        
        # 尝试获取一些标量信息写入 log (如果 results_all 里有的话)
        # 这里假设 results_all 是个字典，里面可能有 'best_objective' 等
        # 如果结构复杂，就只写简单的完成时间
        
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Task Completed at: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Algorithm: {cfg['algorithm_name']}\n")
            f.write(f"Trucks: {cfg['num_trucks']}, UAVs: {cfg['num_uavs']}\n")
            f.write(f"Seed: {cfg['algo_seed']}\n")
            f.write("-" * 20 + "\n")
            f.write("Run Successful.\n")
            # f.write(f"Best Obj: {results_all.get('best_objective', 'N/A')}\n") # 示例
            
        # print(f"✅ [Done] PID:{os.getpid()} 标记已生成: {os.path.basename(marker_path)}")

    except Exception as e:
        print(f"🔥 [Error] PID:{os.getpid()} {save_name}")
        # 如果出错，可以写一个 .error 文件
        error_marker = marker_path.replace(".done", ".error")
        with open(error_marker, "w", encoding="utf-8") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        traceback.print_exc()

# ==========================================================
# 主程序
# ==========================================================
def run_batch_experiments():
    # 确保标记目录存在
    os.makedirs(MARKER_BASE_DIR, exist_ok=True)
    
    experiments = build_experiments()
    total_exp = len(experiments)

    print("==========================================")
    print(f"开始并行实验 (Log标记模式)")
    print(f"标记文件存储于: {MARKER_BASE_DIR}")
    print("==========================================\n")

    start_time_all = time.time()
    
    skip_count = 0
    run_count = 0
    
    for idx, config in enumerate(experiments, start=1):
        print(f"--- 组 [{idx}/{total_exp}]: {config['algorithm_name']} {config['num_trucks']}T {config['num_uavs']}U ---")
        
        procs = []
        for rep_idx in range(REPEAT_PER_TASK):
            # 1. 生成具体配置
            cfg = _make_run_config(config, idx-1, rep_idx)
            
            # 2. ✅ 【关键】检查标记文件是否存在 (Skip 逻辑)
            marker_path = get_marker_path(cfg)
            if os.path.exists(marker_path):
                # print(f"   [Skip] 已存在标记: {os.path.basename(marker_path)}")
                skip_count += 1
                continue # 跳过当前 rep_idx
            
            # 3. 如果不存在，启动子进程
            p = Process(target=_worker, args=(cfg,))
            p.start()
            procs.append(p)
            run_count += 1
            
            # 4. 进程池流控
            if len(procs) >= MAX_PARALLEL:
                for p in procs:
                    p.join()
                procs.clear()
                sys.stdout.flush() # 刷新打印

        # 等待该组剩余进程
        for p in procs:
            p.join()
        
    duration = time.time() - start_time_all
    print("\n==========================================")
    print(f"所有任务结束。")
    print(f"实际运行: {run_count}")
    print(f"跳过任务: {skip_count} (基于 .done 文件)")
    print(f"总耗时: {duration:.2f} 秒")
    print("==========================================")

if __name__ == "__main__":
    run_batch_experiments()