#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import traceback
import datetime

from main import missionControl


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
    num_points_list = [100, 200]
    truck_list = [3, 5]
    uav_list = [6, 10]
    iter_list = [500, 500]
    seeds = [6, 6]

    # 这些也可以单个或按组给
    target_ranges = [None, None]
    coord_scales = [1.0, 1.0]
    Z_coords = [0.05, 0.05]
    uav_distance_ratios = [None, None]
    uav_distances = [15, 15]

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
        len(num_points_list), len(truck_list), len(uav_list), len(iter_list), len(seeds),
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

        tr   = _pick(target_ranges, i, L, "target_ranges")
        cs   = _pick(coord_scales, i, L, "coord_scales")
        zdr  = _pick(Z_coords, i, L, "Z_coords")
        ratio= _pick(uav_distance_ratios, i, L, "uav_distance_ratios")
        ud   = _pick(uav_distances, i, L, "uav_distances")

        # ✅ MODIFIED: 自动命名（只为这一组生成）
        tag_tr = "raw" if tr is None else f"rng{tr[0]}x{tr[1]}"
        tag_ratio = "ratioNA" if ratio is None else f"ratio{ratio:.3f}"
        save_name = (
            f"{ds}_N{n}_T{nt}_U{nu}_I{iters}_S{seed}_"
            f"{tag_tr}_scale{cs}_{tag_ratio}_Z{zdr}"
        )

        cfg = {
            "problem_name": f"case_{nu}UAV_{nt}Truck",
            "save_name": save_name,
            "iterations": iters,

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

        try:
            t0 = time.time()
            app = missionControl(config=config)
            t1 = time.time()

            msg = f"[OK] {save_name} finished in {t1 - t0:.2f}s\n"
            print(msg)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().isoformat()} {msg}")

            success_count += 1

        except Exception as e:
            fail_count += 1
            print(f"[FAIL] {save_name}")
            print(f"错误信息: {str(e)}")
            traceback.print_exc()

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().isoformat()} [FAIL] {save_name}\n")
                f.write(str(e) + "\n")
                f.write(traceback.format_exc() + "\n")

            # 继续下一个
            continue

    duration = time.time() - start_time_all

    print("==========================================")
    print("批量实验结束。")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"成功: {success_count}, 跳过(已存在): {skip_count}, 失败: {fail_count}")
    print(f"结果目录: {save_dir}")
    print("==========================================")


if __name__ == "__main__":
    run_batch_experiments()
