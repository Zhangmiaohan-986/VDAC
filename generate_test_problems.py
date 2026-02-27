import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math

# 基准测试函数数据选择
def generate_points(num_points,seed):
    # file_path_R201 = r'E:\TRE-双层路网协同巡检调度优化\coding\tits_CMDVRP-CT\tits_CMDVRP-CT\map\R201.txt'
    # file_path_RC101 = r'E:\TRE-双层路网协同巡检调度优化\coding\tits_CMDVRP-CT\tits_CMDVRP-CT\map\RC101.txt'

    # file_path_R201 = r'/Volumes/Zhang/TRE-双层路网协同巡检调度优化/coding/tits_CMDVRP-CT/tits_CMDVRP-CT/map/R201.txt'
    # file_path_RC101 = r'/Volumes/Zhang/TRE-双层路网协同巡检调度优化/coding/tits_CMDVRP-CT/tits_CMDVRP-CT/map/RC101.txt'
    # file_path_R201 = r'VDAC\map_test\R201.txt'
    # file_path_RC101 = r'VDAC\map_test\RC101.txt'
    file_path_R201 = r'/Users/zhangmiaohan/猫咪存储文件/maomi_github/VDAC/map_test/RC1_4_1.TXT'
    # file_path_R201 = r'map_test\RC1_4_1.TXT'

    np.random.seed(seed)
    with open(file_path_R201, 'r') as file:
        data0_R201 = file.read()
    # 将数据转换为DataFrame
    data_R201 = []
    for line in data0_R201.strip().split('\n'):
        data_R201.append(line.split())
    columns = ["CUST", "XCOORD.", 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R201 = pd.DataFrame(data_R201[1:], columns=columns)
    # 将字符型列转换为数字
    numeric_cols = ['CUST', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R201[numeric_cols] = df_R201[numeric_cols].apply(pd.to_numeric, errors='coerce')
    start_pos = (float(df_R201.at[0, 'XCOORD.']), float(df_R201.at[0, 'YCOORD.']))
    # if num_points <= 100:
    print("客户节点数量:", num_points)
    if num_points <= 500:
        # position_points_sample = df_R201.sample(n=num_points,random_state=seed)
        # position_points_sample = position_points_sample.sort_index()
        # return position_points_sample, start_pos
        # 单独取出 depot 行（index=0）
        depot_row = df_R201.iloc[[0]]
        # 剩下的全是客户
        customer_rows = df_R201.iloc[1:]

        if num_points <= 1:
            # 只要 1 个点，那就只返回 depot
            position_points_sample = depot_row.copy()
        else:
            sample_size = min(num_points - 1, len(customer_rows))
            sampled_customers = _sample_customers_diameter_and_min_neighbors(
                customer_rows=customer_rows,
                k=sample_size,
                seed=seed,
                max_pair_dist=100.0,  # 任意两点<=100km
                near_dist=40.0,       # 至少2个邻居<=40km
                min_nb=2              # ✅ 想要更紧就改成3
            )
            # ✅ ================== 改动：先筛后抽（结束） ==================
            position_points_sample = pd.concat([depot_row, sampled_customers], axis=0)
            # sampled_customers = customer_rows.sample(
            #     n=sample_size,
            #     random_state=seed
            # )
            # position_points_sample = pd.concat(
            #     [depot_row, sampled_customers],
            #     axis=0
            # )

        # 用原始 index 排序，这样 depot 一定在最前面（index=0）
        position_points_sample = position_points_sample.sort_index()
        return position_points_sample, start_pos
    else:
        # with open(file_path_RC101, 'r') as file:
        #     data0_RC101 = file.read()
        # data_RC101 = []
        # for line in data0_RC101.strip().split('\n'):
        #     data_RC101.append(line.split())
        # df_RC101 = pd.DataFrame(data_RC101[1:], columns=columns)
        # df_RC101[numeric_cols] = df_RC101[numeric_cols].apply(pd.to_numeric, errors='coerce')
        # df_combine = pd.concat([df_R201, df_RC101], ignore_index=True)
        # position_points_sample = df_combine.sample(n=num_points,random_state=seed)
        # position_points_sample = position_points_sample.sort_index()
        with open(file_path_RC101, 'r') as file:
            data0_RC101 = file.read()

        data_RC101 = []
        for line in data0_RC101.strip().split('\n'):
            data_RC101.append(line.split())

        df_RC101 = pd.DataFrame(data_RC101[1:], columns=columns)
        df_RC101[numeric_cols] = df_RC101[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # depot 仍然只用 R201 的第一个点
        depot_row = df_R201.iloc[[0]]
        r201_customers = df_R201.iloc[1:]

        # 合并：R201 客户 + RC101 全部
        df_combine_no_depot = pd.concat(
            [r201_customers, df_RC101],
            ignore_index=True
        )

        if num_points <= 1:
            position_points_sample = depot_row.copy()
        else:
            sample_size = min(num_points - 1, len(df_combine_no_depot))
            sampled_customers = df_combine_no_depot.sample(
                n=sample_size,
                random_state=seed
            )
            position_points_sample = pd.concat(
                [depot_row, sampled_customers],
                axis=0
            )

        position_points_sample = position_points_sample.sort_index()
        return position_points_sample, start_pos

# def generates_points(num_points,seed):
#     file_path_R1 = '/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/homberger_400_customer_instances/R1_4_1.TXT'
#     np.random.seed(seed)
#     with open(file_path_R1, 'r') as file:
#         data0_R1 = file.read()
#     # 将数据转换为DataFrame
#     data_R1 = []
#     for line in data0_R1.strip().split('\n'):
#         data_R1.append(line.split())
#     columns = ["CUST", "XCOORD.", 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
#     df_R1 = pd.DataFrame(data_R1[1:], columns=columns) 
#     # 将字符型列转换为数字
#     numeric_cols = ['CUST', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
#     df_R1[numeric_cols] = df_R1[numeric_cols].apply(pd.to_numeric, errors='coerce')
#     position_points_sample = df_R1.sample(n=num_points,random_state=seed)
#     position_points_sample = position_points_sample.sort_index()
#     return position_points_sample  # 输出pd格式的

def generates_points(num_points, seed):
    file_path_R1 = '/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/homberger_400_customer_instances/R1_4_1.TXT'
    np.random.seed(seed)

    with open(file_path_R1, 'r') as file:
        data0_R1 = file.read()

    data_R1 = []
    for line in data0_R1.strip().split('\n'):
        data_R1.append(line.split())

    columns = ["CUST", "XCOORD.", 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R1 = pd.DataFrame(data_R1[1:], columns=columns)

    numeric_cols = ['CUST', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R1[numeric_cols] = df_R1[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Homberger 的 depot：同样假设在第一行（CUST=1）
    depot_row = df_R1.iloc[[0]]
    customer_rows = df_R1.iloc[1:]

    if num_points <= 1:
        # 只要 depot
        position_points_sample = depot_row.copy()
    else:
        sample_size = min(num_points - 1, len(customer_rows))
        sampled_customers = customer_rows.sample(
            n=sample_size,
            random_state=seed
        )
        position_points_sample = pd.concat(
            [depot_row, sampled_customers],
            axis=0
        )

    # 排序保证 depot 在最前
    position_points_sample = position_points_sample.sort_index()

    return position_points_sample

def generate_graph(position_points, seed, uav_distance=None, prob=0.75):
    if uav_distance is None:
        uav_distance = 50  # 设置您的阈值,无人机最大飞行距离
    random.seed(seed)
    coordinates = list(zip(position_points['XCOORD.'], position_points['YCOORD.']))
    # 空地节点数目
    num_air = math.floor(len(coordinates) * 1 / 2)  # 空地节点数目
    num_ground = int(len(coordinates) - num_air)  # 地面节点数目
    # 空地节点坐标/随机生成
    coordinates_air = random.sample(coordinates, num_air)  # 空地节点坐标
    coordinates_air_matrix = compute_distance_matrix(coordinates_air)  # 空地节点距离矩阵
    # 获得地面节点坐标
    coordinates_ground = list(set(coordinates).difference(coordinates_air))
    coordinates_ground = list(coordinates_ground)  # 转换为列表
    coordinates_ground_matrix = compute_distance_matrix(coordinates_ground)
    # 创建空中无向图
    G_air = nx.Graph()
    # 添加空中节点，节点ID从0开始
    for i, coord in enumerate(coordinates_air):
        G_air.add_node(i, pos=coord)
    # 添加空中边的连接
    G_air = add_random_tree(G_air, coordinates_air_matrix, coordinates_air)
    # 分割空中图的长边
    G_air, air_positions, max_air_id, new_air_nodes = split_long_edges(G_air, uav_distance,
                                                                       starting_node_id=len(coordinates_air))

    # 创建地面无向图，节点ID从0开始
    G_ground = nx.Graph()
    for i, coord in enumerate(coordinates_ground):
        G_ground.add_node(i, pos=coord)
    # 添加地面边的连接
    G_ground = add_random_tree(G_ground, coordinates_ground_matrix, coordinates_ground, max_connect_num=4)
    # 分割地面图的长边
    # G_ground, ground_positions, max_ground_id, new_ground_nodes = split_long_edges(G_ground, uav_distance,
    #                                                                                starting_node_id=len(
    #                                                                                    coordinates_ground))
    ground_positions = nx.get_node_attributes(G_ground, 'pos')

    # 为了避免节点ID冲突，需要调整地面图的节点ID
    # 将地面图的节点ID偏移
    node_id_offset = max_air_id + 1
    mapping = {node: node + node_id_offset for node in G_ground.nodes()}
    G_ground = nx.relabel_nodes(G_ground, mapping)
    # 更新地面图的positions
    ground_positions = {node + node_id_offset: pos for node, pos in ground_positions.items()}

    # 获取空地距离矩阵及邻接矩阵
    air_adj_matrix = np.array(nx.adjacency_matrix(G_air).todense())
    ground_adj_matrix = np.array(nx.adjacency_matrix(G_ground).todense())

    return G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions

# ✅ ================== 约束抽样器（新增） ==================
def _sample_with_neighbor_bounds(customer_rows, k, seed,
                                 radius=40.0, min_nb=2, max_nb=3,
                                 trials=120):
    """
    从 customer_rows 中抽 k 个点，使得在抽样子集内：
    每个点的“近邻数”(distance < radius) ∈ [min_nb, max_nb]

    返回：满足约束的 DataFrame（尽量满k；如果极端情况下凑不齐，会返回最接近的一次）
    """
    if k <= 0:
        return customer_rows.iloc[0:0].copy()
    if len(customer_rows) <= k:
        # 点都不够，直接返回（后续你可以选择放宽约束）
        return customer_rows.copy()

    coords = customer_rows[['XCOORD.', 'YCOORD.']].to_numpy(dtype=float)
    n = len(coords)

    # 距离矩阵（n*n）
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)

    # 邻接矩阵：近邻(<radius)
    near = (d < radius)

    # 先做一次“必要条件”预筛：在全体里近邻数 < min_nb 的点必然无解
    deg_full = near.sum(axis=1)
    feasible_mask = (deg_full >= min_nb)
    feasible_idx = np.where(feasible_mask)[0]
    if len(feasible_idx) < k:
        # 兜底：可行点都不够，退一步——只保证“至少1个近邻”或直接返回可行点
        # 你要是绝对必须 2–3，那这里就只能接受“凑不齐”
        return customer_rows.iloc[feasible_idx].copy()

    rng = np.random.RandomState(seed)

    best_S = None
    best_score = -1

    # --- 多次尝试构造满足度约束的子集 ---
    for t in range(trials):
        # 从可行点里挑一个起点：偏向“近邻数中等”的点更容易做到≤3
        start = int(rng.choice(feasible_idx))
        S = [start]
        inS = np.zeros(n, dtype=bool); inS[start] = True

        # 子集内度数
        deg_in = np.zeros(n, dtype=int)

        # 构造到k个
        while len(S) < k:
            candidates = np.where((~inS) & feasible_mask)[0]
            if len(candidates) == 0:
                break

            # 计算候选加入后的可行性与得分
            best_c = None
            best_c_score = -1e18

            # 当前“缺邻居”的点（希望新点靠近它们）
            need_more = (inS & (deg_in < min_nb))

            for c in candidates:
                neigh_to_S = near[c] & inS
                add_to = np.where(neigh_to_S)[0]

                # 如果加进来，会把某些已有点度数推到 > max_nb，直接不允许
                if np.any(deg_in[add_to] + 1 > max_nb):
                    continue

                # 新点本身加入后在S里的近邻数
                new_deg = int(neigh_to_S.sum())
                if new_deg > max_nb:
                    continue
                # 注意：new_deg 可能 < min_nb，允许暂时不足，后面靠再加点补齐
                # 但是如果它在“全体可行池里”近邻都很少，也不太可能补齐
                # 用 deg_full[c] 给它一点惩罚/奖励

                # 得分：优先帮助当前缺邻居的点，同时希望新点本身也能更容易达到min_nb
                help_need = int((neigh_to_S & need_more).sum())
                score = (
                    3.0 * help_need +          # ✅ 帮助“缺邻居”的老点
                    0.4 * new_deg +            # ✅ 新点连上越多越好（但不能>3）
                    0.05 * deg_full[c]         # ✅ 新点在全体里邻居多一点，后续可补齐概率更大
                )

                # 加一点随机扰动，避免陷入局部最优
                score += rng.uniform(-0.15, 0.15)

                if score > best_c_score:
                    best_c_score = score
                    best_c = c

            if best_c is None:
                break

            # 真正加入 best_c
            inS[best_c] = True
            S.append(best_c)

            # 更新度数
            connected = np.where(near[best_c] & inS)[0]
            # connected 包含自身吗？near对角是False，所以不包含
            deg_in[connected] += 1
            deg_in[best_c] = int((near[best_c] & inS).sum())

        # 评价这次构造
        S_idx = np.array(S, dtype=int)
        degS = np.array([int((near[i] & inS).sum()) for i in S_idx], dtype=int)

        # 满足度约束的数量
        ok = np.sum((degS >= min_nb) & (degS <= max_nb))
        # 同时偏好“接近满k”
        score_total = ok * 10 + len(S_idx)

        if score_total > best_score:
            best_score = score_total
            best_S = S_idx

        # 如果已经满k且全部满足，直接收工
        if len(S_idx) == k and ok == k:
            break

    # 输出最好的一次
    if best_S is None:
        # 理论上不会到这里
        best_S = rng.choice(feasible_idx, size=k, replace=False)

    # 如果最好的一次超过k（一般不会），截断
    if len(best_S) > k:
        best_S = best_S[:k]

    return customer_rows.iloc[best_S].copy()
# ✅ ================== 约束抽样器（新增结束） ==================

# ✅ ================== 先筛后抽：直径<=100 + 每点近邻>=min_nb（新增） ==================
def _sample_customers_diameter_and_min_neighbors(
    customer_rows: pd.DataFrame,
    k: int,
    seed: int,
    max_pair_dist: float = 100.0,   # 任意两点<=100
    near_dist: float = 40.0,        # 近邻阈值<=40
    min_nb: int = 2,                # 每点至少2个近邻（要>=3就改成3）
    center_trials: int = 80,        # 尝试多少个中心
    draw_trials: int = 120          # 每个中心尝试抽样次数
):
    """
    返回 sampled_customers（DataFrame，结构与 customer_rows 完全一致）
    目标：
      (1) 任意两点距离 <= max_pair_dist
      (2) 每个点在子集内的近邻数（<=near_dist） >= min_nb
    """
    if k <= 0:
        return customer_rows.iloc[0:0].copy()
    if len(customer_rows) <= k:
        return customer_rows.copy()

    rng = np.random.RandomState(seed)
    coords = customer_rows[['XCOORD.', 'YCOORD.']].to_numpy(dtype=float)
    n = len(coords)

    # 全局距离矩阵
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)

    # ✅ 用“中心半径R=max_pair_dist/2”的充分条件保证直径<=max_pair_dist
    R = max_pair_dist / 2.0

    center_ids = rng.choice(n, size=min(center_trials, n), replace=False)

    best_idx = None
    best_score = (-1, -1)  # (满足近邻下界的点数, 子集规模)

    for c in center_ids:
        center = coords[c]
        pool = np.where(np.linalg.norm(coords - center, axis=1) <= R + 1e-9)[0]
        if len(pool) < k:
            continue

        # pool 内近邻关系（<=near_dist）
        near_pool = (D[np.ix_(pool, pool)] <= near_dist + 1e-9)
        np.fill_diagonal(near_pool, False)
        deg_pool = near_pool.sum(axis=1)

        # 必要条件：在pool内度数>=min_nb才可能
        feasible_mask = (deg_pool >= min_nb)
        feasible_pool = pool[feasible_mask]
        if len(feasible_pool) < k:
            continue

        # 在 feasible_pool 内做多次随机抽样，找满足“每点>=min_nb近邻”的子集
        for t in range(draw_trials):
            idx = rng.choice(feasible_pool, size=k, replace=False)

            # 子集内近邻度数
            subD = D[np.ix_(idx, idx)]
            subNear = (subD <= near_dist + 1e-9)
            np.fill_diagonal(subNear, False)
            subDeg = subNear.sum(axis=1)

            satisfied = int(np.sum(subDeg >= min_nb))
            score = (satisfied, k)

            if score > best_score:
                best_score = score
                best_idx = idx

            if satisfied == k:
                # 完全满足，直接返回
                return customer_rows.iloc[idx].copy()

    # 兜底：如果没找到完全满足的，返回“最接近”的那次；再不行就随机k个
    if best_idx is not None:
        return customer_rows.iloc[best_idx].copy()

    idx = rng.choice(n, size=k, replace=False)
    return customer_rows.iloc[idx].copy()
# ✅ ================== 新增结束 ==================


# 分割长边的函数
def split_long_edges(graph, threshold, starting_node_id):
    positions = nx.get_node_attributes(graph, 'pos')
    max_node_id = starting_node_id - 1  # 初始值为起始节点ID减1
    edges_to_remove = []
    new_nodes = {}

    for u, v in list(graph.edges()):
        pos_u = positions[u]
        pos_v = positions[v]
        distance = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
        if distance <= threshold:
            continue
        else:
            # 需要分割这条长边
            num_segments = int(np.ceil(distance / threshold))
            # 计算中间节点的位置
            positions_list = []
            for i in range(1, num_segments):
                t = i / num_segments
                x_new = (1 - t) * pos_u[0] + t * pos_v[0]
                y_new = (1 - t) * pos_u[1] + t * pos_v[1]
                positions_list.append((x_new, y_new))
            # 添加新节点到图中
            new_node_ids = []
            for pos in positions_list:
                max_node_id += 1
                new_node_id = max_node_id
                positions[new_node_id] = pos
                graph.add_node(new_node_id, pos=pos)
                new_node_ids.append(new_node_id)
                new_nodes[new_node_id] = pos
            # 移除原来的长边
            edges_to_remove.append((u, v))
            # 添加新的边，连接原节点和新节点
            nodes_in_chain = [u] + new_node_ids + [v]
            for n1, n2 in zip(nodes_in_chain[:-1], nodes_in_chain[1:]):
                pos1 = positions[n1]
                pos2 = positions[n2]
                new_distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                graph.add_edge(n1, n2, weight=new_distance)
    # 移除所有需要移除的边
    graph.remove_edges_from(edges_to_remove)
    return graph, positions, max_node_id, new_nodes

# 基于所得到的测试坐标图例，设计得到无向图
# def generate_graph(position_points, seed, prob=0.75, uav_distance=None):
#     if uav_distance is None:
#         uav_distance = 50
#     random.seed(seed)
#     coordinates = list(zip(position_points['XCOORD.'], position_points['YCOORD.']))
#     # 空地节点数目
#     num_air = math.floor(len(coordinates) * 3/5)
#     num_ground = int(len(coordinates) - num_air)
#     # 空地节点坐标/随机生成
#     coordinates_air = random.sample(coordinates,num_air)
#     coordinates_air_matrix = compute_distance_matrix(coordinates_air)
#     # 获得地面节点坐标
#     coordinates_ground = list(set(coordinates).difference(coordinates_air))
#     coordinates_ground_matrix = compute_distance_matrix(coordinates_ground)
#     max_distance_air = np.max(coordinates_air_matrix)
#     max_distance_ground = np.max(coordinates_ground_matrix)
#     # 创建空中无向图
#     G_air = nx.Graph()
#     # 添加空中节点
#     for i, coord in enumerate(coordinates_air):
#         G_air.add_node(i, pos=coord)
#     # 添加空中边的连接
#     G_air = add_random_tree(G_air,coordinates_air_matrix,coordinates_air)
#     # 创建地面节点
#     G_ground = nx.Graph()
#     for i, coord in enumerate(coordinates_ground):
#         G_ground.add_node(i, pos=coord)
#     G_ground = add_random_tree(G_ground,coordinates_ground_matrix,coordinates_ground,4)
#     # 获取空地距离矩阵及邻接矩阵
#     air_adj_matrix = np.array(nx.adjacency_matrix(G_air).todense())
#     air_pos = nx.get_node_attributes(G_air, 'pos')
#     ground_adj_matrix = np.array(nx.adjacency_matrix(G_ground).todense())
#     ground_pos = nx.get_node_attributes(G_ground, 'pos')
#
#     return G_air,G_ground,air_adj_matrix,air_pos,ground_adj_matrix,ground_pos

def add_random_tree(G, distance_matrix, coordinates, max_connect_num=None):
    if max_connect_num is None:
        max_connect_num = 3
    num_nodes = len(distance_matrix)
    visited = np.full((num_nodes,num_nodes),False)
    coordinates_origin = (0,0,0)
    distance_origin = compute_distance_matrix_3D(coordinates, coordinates_origin)
    sorted_indices_origin = np.argsort(distance_origin)[0,0]
    near_path = find_nearest_path(distance_matrix,sorted_indices_origin)
    connect_count = {i: 0 for i in range(num_nodes)}
    for i in range(len(near_path)-1):
        current = near_path[i]
        next_current = near_path[i+1]
        current_prob = connect_prob(connect_count[current])
        if random.random() < current_prob:
            G.add_edge(current,next_current,weight=distance_matrix[current,next_current])
            connect_count[current] += 1
            connect_count[next_current] += 1
            visited[current,next_current] = True
            visited[next_current,current] = True
    # 继续遍历未连接的节点，按 50% 概率连接
    for current in near_path:
        # 获取当前节点的距离排序
        sorted_neighbors = np.argsort(distance_matrix[current])
        for neighbor in sorted_neighbors:
            # 检查连接是否已经存在，且是否当前节点和邻居节点不是同一节点
            if not visited[current, neighbor] and current != neighbor:
                # 仅当两个节点的连接次数都少于 3 时才考虑连接
                if connect_count[current] < max_connect_num and connect_count[neighbor] < 10:
                    # 50% 概率连接
                    # 连接最近的节点（确保 neighbor 也是 current 最近的）
                    # if sorted_neighbors[1] == neighbor or sorted_neighbors[0] == neighbor:
                    if random.random() < connect_prob(connect_count[current]):
                        # 添加边并更新连接计数
                        G.add_edge(current, neighbor, weight=distance_matrix[current, neighbor])
                        visited[current, neighbor] = True
                        visited[neighbor, current] = True
                        connect_count[current] += 1
                        connect_count[neighbor] += 1
    return G
# 生成随机树的函数
# def add_random_tree(G, distance_matrix, coordinates, max_connect_num=None):
#     if max_connect_num is None:
#         max_connect_num = 3
#     num_nodes = len(distance_matrix)

#     # 先用 MST 保证全连通，再做原本的随机加边
#     visited = np.full((num_nodes, num_nodes), False)
#     connect_count = {i: 0 for i in range(num_nodes)}

#     # 构造完全图并取最小生成树，避免分断
#     complete = nx.Graph()
#     for i in range(num_nodes):
#         complete.add_node(i)
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             w = distance_matrix[i, j]
#             complete.add_edge(i, j, weight=w)

#     mst = nx.minimum_spanning_tree(complete)
#     for u, v, data in mst.edges(data=True):
#         G.add_edge(u, v, weight=data["weight"])
#         visited[u, v] = visited[v, u] = True
#         connect_count[u] += 1
#         connect_count[v] += 1

#     # 保留原有“最近路径+随机加边”的逻辑以丰富度数
#     coordinates_origin = (0, 0, 0)
#     distance_origin = compute_distance_matrix_3D(coordinates, coordinates_origin)
#     sorted_indices_origin = np.argsort(distance_origin)[0, 0]
#     near_path = find_nearest_path(distance_matrix, sorted_indices_origin)
#     for i in range(len(near_path)-1):
#         current = near_path[i]
#         next_current = near_path[i+1]
#         current_prob = connect_prob(connect_count[current])
#         if random.random() < current_prob:
#             G.add_edge(current,next_current,weight=distance_matrix[current,next_current])
#             connect_count[current] += 1
#             connect_count[next_current] += 1
#             visited[current,next_current] = True
#             visited[next_current,current] = True
#     # 继续遍历未连接的节点，按 50% 概率连接
#     for current in near_path:
#         # 获取当前节点的距离排序
#         sorted_neighbors = np.argsort(distance_matrix[current])
#         for neighbor in sorted_neighbors:
#             # 检查连接是否已经存在，且是否当前节点和邻居节点不是同一节点
#             if not visited[current, neighbor] and current != neighbor:
#                 # 仅当两个节点的连接次数都少于 3 时才考虑连接
#                 if connect_count[current] < max_connect_num and connect_count[neighbor] < 10:
#                     # 50% 概率连接
#                     # 连接最近的节点（确保 neighbor 也是 current 最近的）
#                     # if sorted_neighbors[1] == neighbor or sorted_neighbors[0] == neighbor:
#                     if random.random() < connect_prob(connect_count[current]):
#                         # 添加边并更新连接计数
#                         G.add_edge(current, neighbor, weight=distance_matrix[current, neighbor])
#                         visited[current, neighbor] = True
#                         visited[neighbor, current] = True
#                         connect_count[current] += 1
#                         connect_count[neighbor] += 1
#     return G

def find_nearest_path(distance_matrix, start_index):
    num_nodes = len(distance_matrix)
    visited = [False]*num_nodes
    path = []
    current_node = start_index
    while len(path)<num_nodes:
        visited[current_node] = True
        path.append(current_node)
        distance = distance_matrix[current_node]
        sorted_indices = np.argsort(distance)
        found_next = False
        for neighbor in sorted_indices:
            if not visited[neighbor] and neighbor != current_node:
                current_node = neighbor
                found_next = True
                break
        if not found_next:
            break  # 没有找到未访问的邻居节点，结束循环
    return path

# 建立塔杆节点连接概率函数
def connect_prob(connections):
    # 计算连接概率
    if connections == 0:
        return 1.0
    return max(0,1.0-0.2*connections)

# 生成获得的距离矩阵
def compute_distance_matrix(coordinates, origin=None):
    num_coordinates = len(coordinates)
    if origin is None:
        distance_matrix = np.zeros((num_coordinates, num_coordinates))
        for i in range(num_coordinates):
            for j in range(i + 1, num_coordinates):
                distance_matrix[i,j] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
                distance_matrix[j,i] = distance_matrix[i,j]
        return distance_matrix
    else:
        # 计算所有节点到原点的距离
        distance_matrix = np.linalg.norm(np.array(coordinates) - np.array(origin), axis=1).reshape(1, -1)
        return distance_matrix

# 生成3D距离矩阵
# 修改后的3D距离矩阵计算函数
# 修改后的3D距离矩阵计算函数
def compute_distance_matrix_3D(coordinates, origin=None):
    """
    计算3D坐标点之间的欧几里得距离矩阵
    
    参数:
    coordinates: 包含3D坐标的列表 [(x1,y1,z1), (x2,y2,z2), ...]
    origin: 可选的原点坐标 (x0,y0,z0)
    
    返回:
    距离矩阵或到原点的距离向量
    """
    # 确保坐标是数组
    coords_array = np.array(coordinates)
    
    if origin is None:
        num_coordinates = len(coordinates)
        distance_matrix = np.zeros((num_coordinates, num_coordinates))
        
        for i in range(num_coordinates):
            for j in range(i + 1, num_coordinates):
                # 计算3D欧几里得距离
                distance_matrix[i, j] = np.linalg.norm(coords_array[i] - coords_array[j])
                distance_matrix[j, i] = distance_matrix[i, j]  # 对称性
                
        return distance_matrix
    else:
        # 计算所有节点到原点的3D距离
        origin_array = np.array(origin)
        distance_vector = np.linalg.norm(coords_array - origin_array, axis=1).reshape(1, -1)
        return distance_vector

def split_long_edges_with_type(graph, threshold, starting_node_id, edge_type):
    """
    分割长边并保留节点类型信息
    
    参数:
    graph: 需要处理的图
    threshold: 边长度阈值
    starting_node_id: 新节点编号的起始值
    
    返回:
    更新后的图、节点位置字典、最大节点ID、新节点字典
    """
    positions = nx.get_node_attributes(graph, 'pos')
    node_types = nx.get_node_attributes(graph, 'type')
    max_node_id = starting_node_id - 1
    edges_to_remove = []
    new_nodes = {}

    for u, v in list(graph.edges()):
        pos_u = np.array(positions[u])
        pos_v = np.array(positions[v])
        distance = np.linalg.norm(pos_u - pos_v)
        
        if distance <= threshold:
            continue
        else:
            # 需要分割这条长边
            num_segments = int(np.ceil(distance / threshold))
            
            # 计算中间节点的位置（3D）
            positions_list = []
            for i in range(1, num_segments):
                t = i / num_segments
                new_pos = (1 - t) * pos_u + t * pos_v
                positions_list.append(tuple(new_pos))
            
            # 添加新节点到图中
            new_node_ids = []
            for pos in positions_list:
                max_node_id += 1
                new_node_id = max_node_id
                positions[new_node_id] = pos
                # 计算节点类型 - 使用原边两端节点类型的组合或保持一致
                # edge_type = f"RELAY_{node_types.get(u, 'AIR')}_{node_types.get(v, 'AIR')}"
                new_edge_type = edge_type
                graph.add_node(new_node_id, pos=pos, type=new_edge_type)
                new_node_ids.append(new_node_id)
                new_nodes[new_node_id] = pos
            
            # 移除原来的长边
            edges_to_remove.append((u, v))
            
            # 添加新的边，连接原节点和新节点
            nodes_in_chain = [u] + new_node_ids + [v]
            for n1, n2 in zip(nodes_in_chain[:-1], nodes_in_chain[1:]):
                pos1 = np.array(positions[n1])
                pos2 = np.array(positions[n2])
                new_distance = np.linalg.norm(pos1 - pos2)
                graph.add_edge(n1, n2, weight=new_distance)
    
    # 移除所有需要移除的边
    graph.remove_edges_from(edges_to_remove)
    
    return graph, positions, max_node_id, new_nodes

from mpl_toolkits.mplot3d import Axes3D
def visualize_tower_connections(G_air, G_ground):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    G_air_pos = nx.get_node_attributes(G_air, 'pos')
    G_air_pos_3d = {node: (x, y, 45/1000) for node, (x, y) in G_air_pos.items()}

    air_x = [coord[0] for coord in G_air_pos_3d.values()]
    air_y = [coord[1] for coord in G_air_pos_3d.values()]
    air_z = [coord[2] for coord in G_air_pos_3d.values()]

    ax.scatter(air_x, air_y, air_z, c='red', label='Air Nodes')
    # # 添加空中节点的标签
    # for node, (x, y, z) in G_air_pos_3d.items():
    #     ax.text(x, y, z, f'{node}', fontsize=10, color='black')  # 标签显示节点编号


    for edge in G_air.edges():
        node1, node2 = edge
        x_values = [G_air_pos_3d[node1][0], G_air_pos_3d[node2][0]]
        y_values = [G_air_pos_3d[node1][1], G_air_pos_3d[node2][1]]
        z_values = [G_air_pos_3d[node1][2], G_air_pos_3d[node2][2]]
        ax.plot(x_values, y_values, z_values, color='blue', linestyle='dashed', linewidth=2, alpha=0.6)

    # 地面节点绘制
    G_ground_pos = nx.get_node_attributes(G_ground, 'pos')
    G_ground_pos_3d = {node: (x, y, 0) for node, (x, y) in G_ground_pos.items()}


    ground_x = [coord[0] for coord in G_ground_pos_3d.values()]
    ground_y = [coord[1] for coord in G_ground_pos_3d.values()]
    ground_z = [coord[2] for coord in G_ground_pos_3d.values()]

    ax.scatter(ground_x, ground_y, ground_z, c='b', marker='o', label='Ground Nodes')
    # 添加地面节点的标签
    # for node, (x, y, z) in G_ground_pos_3d.items():
    #     ax.text(x, y, z, f'{node}', fontsize=10, color='black')  # 标签显示节点编号

    for edge in G_ground.edges():
        node1, node2 = edge
        x_values_ground = [G_ground_pos_3d[node1][0], G_ground_pos_3d[node2][0]]
        y_values_ground = [G_ground_pos_3d[node1][1], G_ground_pos_3d[node2][1]]
        z_values_ground = [G_ground_pos_3d[node1][2], G_ground_pos_3d[node2][2]]
        ax.plot(x_values_ground, y_values_ground, z_values_ground, color='yellow',  linewidth=2, alpha=0.6)

    ax.set_xlim(min(air_x) - 10, max(air_x) + 10)
    ax.set_ylim(min(air_y) - 10, max(air_y) + 10)
    ax.set_zlim(0, 25)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=12)

    plt.title("Benchmark Graph of a 50-Node Dual-Layer Road Network. test_points:6. generate_graph:1", fontsize=16)
    # # 使用plt.savefig()并指定参数
    # plt.savefig('/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/env_map/100节点示意图1.1.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()  # 本地显示图形

def generate_complex_network(num_points, seed, Z_coord, uav_distance, air_node_num, ground_node_num, customer_node_num):
    # 0. 生成原始数据点
    position_points, start_pos = generate_points(num_points, seed)
    # 新增有关时间窗的考量
    time_windows_h = {}
    for _, row in position_points.iterrows():
        cust_id = int(row["CUST"])
        ready_h   = row["READY"]   / 60.0  # 分钟 → 小时
        due_h     = row["DUE"]     / 60.0
        service_h = row["SERVICE"] / 60.0
        time_windows_h[cust_id] = {
            "ready_h":   ready_h,
            "due_h":     due_h,
            "service_h": service_h,
        }
    # 结束添加
    depot = position_points.iloc[0:1].copy()
    depot['Z_COORD'] = 0  # 仓库节点高度
    depot['NODE_TYPE'] = 'DEPOT'  # 定义仓库节点
    # 1. 生成原始数据点
    remaining_points = position_points.iloc[1:].copy()
    # 计算每类节点的数量
    total_remaining = len(remaining_points)
    # num_air_nodes = total_remaining // 3  # 空中中继集结点
    # num_vtp_nodes = total_remaining // 3  # 地面中继集结点
    # num_customer_nodes = total_remaining - num_air_nodes - num_vtp_nodes  # 客户点
    num_air_nodes = air_node_num  # 空中中继集结点
    num_vtp_nodes = ground_node_num  # 地面中继集结点，地面中继点少选一个，在把仓库加入，数量则正好
    num_customer_nodes = customer_node_num  # 客户点
    # 随机抽样划分
    air_nodes = remaining_points.sample(n=num_air_nodes, random_state=seed)
    air_nodes['NODE_TYPE'] = 'AIR'  # 定义空中中继节点集合
    air_nodes['Z_COORD'] = Z_coord  # 定义空中中继节点高度
    # 删除已选中的空中节点
    remaining_after_air = remaining_points.drop(air_nodes.index)
    vtp_nodes = remaining_after_air.sample(n=num_vtp_nodes, random_state=seed+1)
    vtp_nodes['NODE_TYPE'] = 'VTP'  # 地面VTP节点
    vtp_nodes['Z_COORD'] = 0  # 地面VTP节点高度
    air_vtp_nodes = vtp_nodes.copy()
    air_vtp_nodes['NODE_TYPE'] = 'AIR'
    air_vtp_nodes['Z_COORD'] = Z_coord
    # 删除已选中的地面节点,生成客户节点
    customer_nodes = remaining_after_air.drop(vtp_nodes.index)
    customer_nodes['NODE_TYPE'] = 'CUSTOMER'  # 客户点
    customer_nodes['Z_COORD'] = 0  # 客户点高度
    # 客户点降落的空中节点生成
    air_customer_nodes = customer_nodes.copy()
    air_customer_nodes['NODE_TYPE'] = 'AIR'
    air_customer_nodes['Z_COORD'] = Z_coord

    # 合并所有节点数据
    all_nodes = pd.concat([depot, air_nodes, vtp_nodes, customer_nodes, air_vtp_nodes, air_customer_nodes])

    return generate_complex_graph(all_nodes, depot, air_nodes, vtp_nodes, customer_nodes, air_vtp_nodes, air_customer_nodes, uav_distance, seed)

def generate_complex_graph(all_nodes, depot, air_nodes, vtp_nodes, customer_nodes, air_vtp_nodes, air_customer_nodes, uav_distance, seed):
    # 生成空中图
    random.seed(seed)
     # 1. 提取坐标信息
    air_coordinates = list(zip(air_nodes['XCOORD.'], air_nodes['YCOORD.'], air_nodes['Z_COORD']))
    customer_coordinates = list(zip(customer_nodes['XCOORD.'], customer_nodes['YCOORD.'], customer_nodes['Z_COORD']))
    vtp_coordinates = list(zip(vtp_nodes['XCOORD.'], vtp_nodes['YCOORD.'], vtp_nodes['Z_COORD']))
    depot_coordinate = (depot['XCOORD.'].values[0], depot['YCOORD.'].values[0], depot['Z_COORD'].values[0])
    air_vtp_coordinates = list(zip(air_vtp_nodes['XCOORD.'], air_vtp_nodes['YCOORD.'], air_vtp_nodes['Z_COORD']))
    air_customer_coordinates = list(zip(air_customer_nodes['XCOORD.'], air_customer_nodes['YCOORD.'], air_customer_nodes['Z_COORD']))
    # 2. 创建空中网络
    # 合并空中节点和客户节点，因为客户也要加入空中线路（包含客户点，空中中继节点，和vtp节点）
    air_network_coords = air_coordinates + air_vtp_coordinates + air_customer_coordinates
    air_distance_matrix = compute_distance_matrix_3D(air_network_coords)
    G_air = nx.Graph()
    # 添加空中节点并保留类型信息
    air_node_types = {}  # 用于存储所有节点类型
    # 添加中继空中节点
    for i, coord in enumerate(air_coordinates):
        G_air.add_node(i, pos=coord, type='Aerial Relay Node')
        air_node_types[i] = 'Aerial Relay Node'
    # 添加VTP对应的空中节点
    vtp_offset = len(air_coordinates)

    for i, coord in enumerate(air_vtp_coordinates):
        node_id = i + vtp_offset
        G_air.add_node(node_id, pos=coord, type='VTP Takeoff/Landing Node')
        air_node_types[node_id] = 'VTP Takeoff/Landing Node'

    # 添加客户对应的空中节点
    customer_offset = vtp_offset + len(air_vtp_coordinates)
    customer_positions = {}
    for i, coord in enumerate(air_customer_coordinates):
        node_id = i + customer_offset
        G_air.add_node(node_id, pos=coord, type='CUSTOMER Takeoff/Landing Node')
        air_node_types[node_id] = 'CUSTOMER Takeoff/Landing Node'
        customer_positions[node_id] = coord
    
    # 使用现有函数添加空中边连接
    G_air = add_random_tree(G_air, air_distance_matrix, air_network_coords)
    air_node_type = 'Aerial Relay Node'
    # 分割空中网络长边，并保留节点类型
    G_air, air_positions, max_air_id, new_air_nodes = split_long_edges_with_type(
        G_air, uav_distance, starting_node_id=len(air_network_coords), edge_type=air_node_type
    )
    # 新增代码：更新 air_node_types 字典，添加新中继节点的类型信息
    for new_node_id in new_air_nodes.keys():
        air_node_types[new_node_id] = air_node_type  # 使用统一的中继节点类型
    # 新增代码：为分割后的网络创建新的连接
    if new_air_nodes:  # 如果有新节点生成
        # 获取所有节点的坐标
        all_air_positions = nx.get_node_attributes(G_air, 'pos')
        all_air_coords = list(all_air_positions.values())
        
        # 计算新的距离矩阵，包含所有节点
        new_air_distance_matrix = compute_distance_matrix_3D(all_air_coords)
        
        # 创建新的临时图，包含所有节点但不包含边
        G_air_temp = nx.Graph()
        for node_id, pos in all_air_positions.items():
            G_air_temp.add_node(node_id, pos=pos, type=air_node_types.get(node_id, air_node_type))
        
        # 先复制原有的边到新图
        for u, v in G_air.edges():
            G_air_temp.add_edge(u, v, weight=G_air[u][v].get('weight', 1.0))
        
        # 只为新节点和它们的邻近节点添加新的连接
        new_node_ids = list(new_air_nodes.keys())
        # 为每个新节点寻找可能的新连接
        for node_id in new_node_ids:
            node_pos = all_air_positions[node_id]
            # 计算与其他节点的距离
            for other_id, other_pos in all_air_positions.items():
                # 跳过已经连接的节点和自身
                if other_id == node_id or G_air_temp.has_edge(node_id, other_id):
                    continue
                
                # 计算距离
                dist = np.linalg.norm(np.array(node_pos) - np.array(other_pos))
                
                # 如果距离小于阈值且随机概率满足，则添加连接
                if dist <= uav_distance * 0.8 and random.random() < 0.45:  # 0.8倍阈值，30%概率添加
                    G_air_temp.add_edge(node_id, other_id, weight=dist)
        
        # 将优化后的图赋值给G_air
        G_air = G_air_temp
    # 地面节点
    # 3. 创建地面网络
    G_ground = nx.Graph()
    ground_node_types = {}  # 用于存储所有地面节点类型
    # 添加仓库节点
    G_ground.add_node(0, pos=depot_coordinate, type='DEPOT')
    ground_node_types[0] = 'DEPOT'
    
    # 添加VTP地面节点
    for i, coord in enumerate(vtp_coordinates):
        node_id = i + 1
        G_ground.add_node(node_id, pos=coord, type='VTP')
        ground_node_types[node_id] = 'VTP'
     # 添加客户地面节点
    vtp_count = len(vtp_coordinates)
    for i, coord in enumerate(customer_coordinates):
        node_id = i + vtp_count + 1
        G_ground.add_node(node_id, pos=coord, type='CUSTOMER')
        ground_node_types[node_id] = 'CUSTOMER'
    # 4. 获取图的邻接矩阵和位置
    # air_adj_matrix = np.array(nx.adjacency_matrix(G_air).todense())
    # ground_adj_matrix = np.array(nx.adjacency_matrix(G_ground).todense())
    ground_all_nodes = list(G_ground.nodes())
    for i in range(len(ground_all_nodes)):
        for j in range(i + 1, len(ground_all_nodes)):  # 避免重复添加边 (i,j) 和 (j,i)
            node_i = ground_all_nodes[i]    
            node_j = ground_all_nodes[j]
            # 计算两节点间的欧几里得距离
            dist = np.linalg.norm(np.array(G_ground.nodes[node_i]['pos']) - np.array(G_ground.nodes[node_j]['pos']))
            # 添加边并设置权重为距离
            G_ground.add_edge(node_i, node_j, weight=dist)
    air_positions = nx.get_node_attributes(G_air, 'pos')  # 获取节点位置
    ground_positions = nx.get_node_attributes(G_ground, 'pos')
    # customer_positions = {i: coord for i, coord in enumerate(customer_coordinates)}
    air_adj_matrix = np.array(nx.adjacency_matrix(G_air).todense())
    ground_adj_matrix = np.array(nx.adjacency_matrix(G_ground).todense())
    # 5. 整理原始格式数据，包含所有节点信息
    original_format_data = all_nodes.copy()
    
    # 为分割产生的新节点添加数据
    relay_data = []
    for node_id, pos in new_air_nodes.items():
        relay_row = {
            'CUST': f'RELAY_{node_id}', 
            'XCOORD.': pos[0], 
            'YCOORD.': pos[1],
            'Z_COORD': pos[2],
            'DEMAND': 0,
            'READY': 0,
            'DUE': 9999,
            'SERVICE': 0,
            'NODE_TYPE': 'Aerial Relay Node New'
        }
        relay_data.append(relay_row)
    
    # 添加中继点数据
    if relay_data:
        relay_df = pd.DataFrame(relay_data)
        original_format_data = pd.concat([original_format_data, relay_df])

    # ========= 新增：为每个 ground 节点生成“时间窗（小时）”字典 =========
    customer_time_windows_h = {}

    for node_id, coord in customer_positions.items():
        x, y, z = coord  # z = Z_coord

        # 在 original_format_data 里找到对应的地面客户行：
        # XCOORD, YCOORD 相同，NODE_TYPE 是 'CUSTOMER'，Z_COORD=0
        mask = (
            np.isclose(original_format_data['XCOORD.'], x) &
            np.isclose(original_format_data['YCOORD.'], y) &
            (original_format_data['NODE_TYPE'] == 'CUSTOMER')
        )
        matched = original_format_data[mask]

        if matched.empty:
            # 如果找不到，可以选择 raise 或者略过，这里先打印一下方便你调试
            # print(f"[WARN] No matching CUSTOMER row for air customer node {node_id} at ({x},{y})")
            continue

        row = matched.iloc[0]
        cust_id   = int(row['CUST'])
        ready_h   = row['READY']   / 60.0  # 分钟 → 小时
        due_h     = row['DUE']     / 60.0
        service_h = row['SERVICE'] / 60.0

        customer_time_windows_h[node_id] = {
            "cust":      cust_id,
            "ready_h":   ready_h,
            "due_h":     due_h,
            "service_h": service_h,
        }
    # =================================================================

    return (
        G_air,
        G_ground,
        air_adj_matrix,
        air_positions,
        ground_adj_matrix,
        ground_positions,
        original_format_data,
        air_node_types,
        ground_node_types,
        customer_time_windows_h   # <== 多返回了这个 dict
    )
    # ============================================================

    # return G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions, original_format_data, air_node_types, ground_node_types, ground_time_windows_h


def visualize_tower_connections_3D(G_air, G_ground, all_data, air_node_types, ground_node_types, fig_size=(10, 8), dpi=150, save_path=None):
    """
    创建更美观的3D双层网络可视化，统一显示和保存的效果
    """
    # 设置字体和样式 - 增大所有元素
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,        
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,   
        'axes.titlesize': 16,   
        'xtick.labelsize': 12,  
        'ytick.labelsize': 12,
        'legend.fontsize': 10,  
        'legend.frameon': True,
        'legend.framealpha': 0.7,
        'legend.edgecolor': 'k'
    })
    
    # 创建图形和3D坐标系 - 固定DPI以确保一致性
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取节点位置
    air_positions = nx.get_node_attributes(G_air, 'pos')
    ground_positions = nx.get_node_attributes(G_ground, 'pos')
    
    # 修正的配色方案 - 确保键名与实际节点类型匹配
    node_colors = {
        'DEPOT': '#E63946',       # 鲜红色
        'VTP': '#457B9D',         # 蓝绿色
        'CUSTOMER': '#2A9D8F',    # 绿松石色
        'Aerial Relay Node': '#F3722C',   # 橙色
        'VTP Takeoff/Landing Node': '#8338EC',     # 紫色
        'CUSTOMER Takeoff/Landing Node': '#FF66CC',# 粉色 - 修正键名
        'Aerial Relay Node New': '#FFD60A'# 明亮金色 - 修正大小写
    }
    
    # 修正的标记方案 - 确保键名与实际节点类型匹配
    node_markers = {
        'DEPOT': '*',     # 星形
        'VTP': 's',       # 方形
        'CUSTOMER': 'o',  # 圆形
        'Aerial Relay Node': '^', # 三角形
        'VTP Takeoff/Landing Node': 'D',   # 菱形
        'CUSTOMER Takeoff/Landing Node': 'h', # 六边形 - 修正键名
        'Aerial Relay Node New': 'X'  # X形 - 修正大小写
    }
    
    # 修正的节点尺寸 - 确保键名与实际节点类型匹配
    node_sizes = {
        'DEPOT': 150,      # 显著的仓库节点
        'VTP': 80,         # 增大VTP节点
        'CUSTOMER': 60,    # 增大客户节点
        'Aerial Relay Node': 80,   # 增大空中中继节点
        'VTP Takeoff/Landing Node': 80,     # 增大VTP对应空中节点
        'CUSTOMER Takeoff/Landing Node': 60,# 增大客户对应空中节点 - 修正键名
        'Aerial Relay Node New': 50# 增大新生成的空中中继节点 - 修正大小写
    }
    
    # 创建节点ID与原始数据的映射
    node_to_data = {}
    for _, row in all_data.iterrows():
        if 'CUST' in row and 'NODE_TYPE' in row:
            node_to_data[row['CUST']] = row
    
    # 处理中继节点
    for _, row in all_data.iterrows():
        if 'CUST' in row and str(row['CUST']).startswith('RELAY_'):  # 修正前缀 CUST_ -> RELAY_
            try:
                node_id = int(str(row['CUST']).replace('RELAY_', ''))
                node_to_data[node_id] = row
            except:
                continue
    
    # 整合所有节点类型
    all_node_types = {}
    all_node_types.update(air_node_types)
    all_node_types.update(ground_node_types)
    
    # 存储绘制的节点，用于避免在图例中重复
    plotted_types = set()
    
    # 设置背景颜色，增强3D效果
    ax.set_facecolor('#f8f9fa')
    
    # 绘制空中网络节点，添加阴影效果以增强立体感
    for node in G_air.nodes():
        node_type = air_node_types.get(node, 'Aerial Relay Node')
        pos = air_positions[node]
        
        if node_type not in plotted_types:
            scatter = ax.scatter(
                pos[0], pos[1], pos[2],
                s=node_sizes.get(node_type, 60),
                c=node_colors.get(node_type, 'gray'),
                marker=node_markers.get(node_type, 'o'),
                edgecolors='black',
                linewidths=0.8,  # 增粗边框
                alpha=0.9,
                label=node_type,
                zorder=10  # 确保节点在线条上方
            )
            plotted_types.add(node_type)
        else:
            scatter = ax.scatter(
                pos[0], pos[1], pos[2],
                s=node_sizes.get(node_type, 60),
                c=node_colors.get(node_type, 'gray'),
                marker=node_markers.get(node_type, 'o'),
                edgecolors='black',
                linewidths=0.8,
                alpha=0.9,
                zorder=10
            )
    
    # 绘制空中网络连接 - 使用虚线表示空中走廊
    for edge in G_air.edges():
        node1, node2 = edge
        if node1 in air_positions and node2 in air_positions:
            pos1 = air_positions[node1]
            pos2 = air_positions[node2]
            
            # 所有空中连接使用虚线，增粗线条
            ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                [pos1[2], pos2[2]],
                color='#666666',
                linestyle=(0, (3, 3)),  # 虚线：3点线，3点空白
                linewidth=1.5,  # 增粗线宽
                alpha=0.6,      # 调整透明度
                zorder=5        # 确保线在节点下方
            )
    
    # 绘制地面网络节点
    for node in G_ground.nodes():
        node_type = ground_node_types.get(node, 'UNKNOWN')
        pos = ground_positions[node]
        
        if node_type not in plotted_types:
            ax.scatter(
                pos[0], pos[1], pos[2],
                s=node_sizes.get(node_type, 60),
                c=node_colors.get(node_type, 'gray'),
                marker=node_markers.get(node_type, 'o'),
                edgecolors='black',
                linewidths=0.8,
                alpha=0.9,
                label=node_type,
                zorder=10
            )
            plotted_types.add(node_type)
        else:
            ax.scatter(
                pos[0], pos[1], pos[2],
                s=node_sizes.get(node_type, 60),
                c=node_colors.get(node_type, 'gray'),
                marker=node_markers.get(node_type, 'o'),
                edgecolors='black',
                linewidths=0.8,
                alpha=0.9,
                zorder=10
            )
    
    # 添加VTP地面节点到空中节点的垂直连接线 - 修正空中节点类型名称
    vtp_ground_nodes = [n for n in G_ground.nodes() if ground_node_types.get(n) == 'VTP']
    for g_node in vtp_ground_nodes:
        g_pos = ground_positions[g_node]
        
        matching_air_nodes = []
        for a_node in G_air.nodes():
            if air_node_types.get(a_node) == 'VTP Takeoff/Landing Node':  # 修正节点类型名称
                a_pos = air_positions[a_node]
                if abs(a_pos[0] - g_pos[0]) < 0.01 and abs(a_pos[1] - g_pos[1]) < 0.01:
                    matching_air_nodes.append(a_node)
        
        for a_node in matching_air_nodes:
            a_pos = air_positions[a_node]
            ax.plot(
                [g_pos[0], a_pos[0]],
                [g_pos[1], a_pos[1]],
                [g_pos[2], a_pos[2]],
                color='#457B9D',
                linestyle=(0, (2, 2)),  # 虚线(2, 2)
                linewidth=1.2,  # 增粗线宽
                alpha=0.7,
                zorder=1
            )
    
    # 添加客户地面节点到空中节点的虚线连接 - 修正空中节点类型名称
    customer_ground_nodes = [n for n in G_ground.nodes() if ground_node_types.get(n) == 'CUSTOMER']
    for g_node in customer_ground_nodes:
        g_pos = ground_positions[g_node]
        
        matching_air_nodes = []
        for a_node in G_air.nodes():
            if air_node_types.get(a_node) == 'CUSTOMER Takeoff/Landing Node':  # 修正节点类型名称
                a_pos = air_positions[a_node]
                if abs(a_pos[0] - g_pos[0]) < 0.01 and abs(a_pos[1] - g_pos[1]) < 0.01:
                    matching_air_nodes.append(a_node)
        
        for a_node in matching_air_nodes:
            a_pos = air_positions[a_node]
            ax.plot(
                [g_pos[0], a_pos[0]],
                [g_pos[1], a_pos[1]],
                [g_pos[2], a_pos[2]],
                color='#2A9D8F',
                linestyle=(0, (2, 2)),
                linewidth=1.2,  # 增粗线宽
                alpha=0.7,
                zorder=1
            )
    
    # 设置坐标轴范围和标签
    x_values = [pos[0] for pos in air_positions.values()] + [pos[0] for pos in ground_positions.values()]
    y_values = [pos[1] for pos in air_positions.values()] + [pos[1] for pos in ground_positions.values()]
    z_values = [pos[2] for pos in air_positions.values()] + [pos[2] for pos in ground_positions.values()]
    
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    z_min, z_max = min(z_values), max(z_values)
    
    # 增加一些边距
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    z_margin = max((z_max - z_min) * 0.1, 1)  # 确保至少有一些高度
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min, z_max + z_margin * 2)  # 增加顶部空间展示立体感
    
    # 设置坐标轴标签 - 增大字体
    ax.set_xlabel('X (Operation Range, km)', fontweight='bold', labelpad=10, fontsize=12)
    ax.set_ylabel('Y (Operation Range, km)', fontweight='bold', labelpad=10, fontsize=12)
    ax.set_zlabel('Z (Altitude, m)', fontweight='bold', labelpad=10, fontsize=12)
    
    # 网格线 - 保持轻盈但可见
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    
    # 设置刻度标记的格式和数量
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    
    # 减少刻度数量
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    
    # 创建自定义图例 - 增大图例
    from matplotlib.lines import Line2D
    
    # 节点类型
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # 连接类型 - 增粗线条
    custom_lines = [
        Line2D([0], [0], color='#666666', lw=1.5, linestyle=(0, (3, 3))),
        Line2D([0], [0], color='#457B9D', lw=1.5, linestyle=(0, (2, 2))),
        Line2D([0], [0], color='#2A9D8F', lw=1.5, linestyle=(0, (2, 2)))
    ]
    
    # 整合所有图例元素
    all_handles = list(by_label.values()) + custom_lines
    all_labels = list(by_label.keys()) + [
        'Urban Aerial Corridor',     # 城市空中走廊
        'VTP Vertical Takeoff/Landing Corridor',          # VTP对应空中中继点
        'Customers Vertical Takeoff/Landing Corridor'     # 客户点对应空中中继点
    ]
    
    # 添加增大的图例
    legend = ax.legend(
        all_handles, all_labels,
        loc='upper right',
        fontsize=8,           # 增大图例字体
        ncol=2,               # 改为两列以便读取
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        framealpha=0.8,
        edgecolor='black',
        markerscale=0.8,      # 增大图标尺寸
        handlelength=1.5,     # 增大线段长度
        handletextpad=0.5,    # 调整间距
        columnspacing=1.0,    # 调整列间距
        borderpad=0.6,        # 增加边距
        title='Network Elements',
        title_fontsize=10     # 图例标题字体
    )
    legend.get_frame().set_linewidth(0.8)
    
    # 标题 - 增大字体
    plt.title('Vehicle-UAV Delivery Environment', fontweight='bold', pad=20, fontsize=14)
    
    # 优化3D视角
    ax.view_init(elev=30, azim=45)
    
    # 使用constrained_layout代替tight_layout，可以更好地处理3D图
    plt.tight_layout()
    save_path = r'E:\南开NKU小论文集合\TITS-基于空中走廊的车辆-无人机协同配送模型与算法研究\coding\tits_VDCD-AC\main_master\map_test.png'
    # 保存图像，使用相同的DPI确保一致性
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    
    # 显示图像，设置一致的DPI
    plt.show()
    
    return fig

# def visualize_tower_connections_3D(G_air, G_ground, all_data, air_node_types, ground_node_types, fig_size=(14, 12), dpi=300, save_path=None):
#     """
#     创建更美观的3D双层网络可视化，具有增强的立体效果和紧凑的布局
#     """
#     # 设置字体和样式
#     plt.rcParams.update({
#         'font.family': 'Arial',
#         'font.size': 8,        # 进一步减小基础字体
#         'axes.linewidth': 1.0,
#         'axes.labelsize': 10,  # 轴标签字体大小
#         'axes.titlesize': 12,  # 标题字体大小
#         'xtick.labelsize': 8,
#         'ytick.labelsize': 8,
#         'legend.fontsize': 6,  # 更小的图例字体
#         'legend.frameon': True,
#         'legend.framealpha': 0.6,
#         'legend.edgecolor': 'k'
#     })
    
#     # 创建图形和3D坐标系
#     fig = plt.figure(figsize=fig_size, dpi=dpi)
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 获取节点位置
#     air_positions = nx.get_node_attributes(G_air, 'pos')
#     ground_positions = nx.get_node_attributes(G_ground, 'pos')
    
#     # 更协调的配色方案
#     node_colors = {
#         'DEPOT': '#E63946',       # 鲜红色
#         'VTP': '#457B9D',         # 蓝绿色
#         'CUSTOMER': '#2A9D8F',    # 绿松石色
#         'AIR_RELAY': '#F3722C',   # 橙色
#         'AIR_VTP': '#8338EC',     # 紫色
#         'AIR_CUSTOMER': '#F8C4B4',# 淡粉色
#         'AIR_RELAY_NEW': '#FFC300'# 金色
#     }
    
#     node_markers = {
#         'DEPOT': '*',     # 星形
#         'VTP': 's',       # 方形
#         'CUSTOMER': 'o',  # 圆形
#         'AIR_RELAY': '^', # 三角形
#         'AIR_VTP': 'D',   # 菱形
#         'AIR_CUSTOMER': 'h', # 六边形
#         'AIR_RELAY_NEW': 'X'  # X形
#     }
    
#     # 更小的节点尺寸
#     node_sizes = {
#         'DEPOT': 40,      # 再减小50%
#         'VTP': 15,
#         'CUSTOMER': 12.5,
#         'AIR_RELAY': 15,
#         'AIR_VTP': 15,
#         'AIR_CUSTOMER': 12.5,
#         'AIR_RELAY_NEW': 10
#     }
    
#     # 创建节点ID与原始数据的映射
#     node_to_data = {}
#     for _, row in all_data.iterrows():
#         if 'CUST' in row and 'NODE_TYPE' in row:
#             node_to_data[row['CUST']] = row
    
#     # 处理中继节点
#     for _, row in all_data.iterrows():
#         if 'CUST' in row and str(row['CUST']).startswith('RELAY_'):
#             try:
#                 node_id = int(str(row['CUST']).replace('RELAY_', ''))
#                 node_to_data[node_id] = row
#             except:
#                 continue
    
#     # 整合所有节点类型
#     all_node_types = {}
#     all_node_types.update(air_node_types)
#     all_node_types.update(ground_node_types)
    
#     # 存储绘制的节点，用于避免在图例中重复
#     plotted_types = set()
    
#     # 设置背景颜色，增强3D效果
#     ax.set_facecolor('#f8f9fa')
    
#     # 绘制空中网络节点，添加阴影效果以增强立体感
#     for node in G_air.nodes():
#         node_type = air_node_types.get(node, 'AIR_RELAY')
#         pos = air_positions[node]
        
#         if node_type not in plotted_types:
#             scatter = ax.scatter(
#                 pos[0], pos[1], pos[2],
#                 s=node_sizes.get(node_type, 20),
#                 c=node_colors.get(node_type, 'gray'),
#                 marker=node_markers.get(node_type, 'o'),
#                 edgecolors='black',
#                 linewidths=0.3,  # 更细的边框
#                 alpha=0.9,
#                 label=node_type,
#                 zorder=10  # 确保节点在线条上方
#             )
#             plotted_types.add(node_type)
#         else:
#             scatter = ax.scatter(
#                 pos[0], pos[1], pos[2],
#                 s=node_sizes.get(node_type, 20),
#                 c=node_colors.get(node_type, 'gray'),
#                 marker=node_markers.get(node_type, 'o'),
#                 edgecolors='black',
#                 linewidths=0.3,
#                 alpha=0.9,
#                 zorder=10
#             )
    
#     # 绘制空中网络连接 - 使用虚线表示空中走廊
#     for edge in G_air.edges():
#         node1, node2 = edge
#         if node1 in air_positions and node2 in air_positions:
#             pos1 = air_positions[node1]
#             pos2 = air_positions[node2]
            
#             # 所有空中连接使用虚线
#             ax.plot(
#                 [pos1[0], pos2[0]],
#                 [pos1[1], pos2[1]],
#                 [pos1[2], pos2[2]],
#                 color='#7c7c7c',
#                 linestyle=(0, (2, 2)),  # 更细的虚线：2点线，2点空白
#                 linewidth=0.8,  # 更细的线宽
#                 alpha=0.5,      # 更高的透明度
#                 zorder=5        # 确保线在节点下方
#             )
    
#     # 绘制地面网络节点
#     for node in G_ground.nodes():
#         node_type = ground_node_types.get(node, 'UNKNOWN')
#         pos = ground_positions[node]
        
#         if node_type not in plotted_types:
#             ax.scatter(
#                 pos[0], pos[1], pos[2],
#                 s=node_sizes.get(node_type, 20),
#                 c=node_colors.get(node_type, 'gray'),
#                 marker=node_markers.get(node_type, 'o'),
#                 edgecolors='black',
#                 linewidths=0.3,
#                 alpha=0.9,
#                 label=node_type,
#                 zorder=10
#             )
#             plotted_types.add(node_type)
#         else:
#             ax.scatter(
#                 pos[0], pos[1], pos[2],
#                 s=node_sizes.get(node_type, 20),
#                 c=node_colors.get(node_type, 'gray'),
#                 marker=node_markers.get(node_type, 'o'),
#                 edgecolors='black',
#                 linewidths=0.3,
#                 alpha=0.9,
#                 zorder=10
#             )
    
#     # 添加VTP地面节点到空中节点的垂直连接线
#     vtp_ground_nodes = [n for n in G_ground.nodes() if ground_node_types.get(n) == 'VTP']
#     for g_node in vtp_ground_nodes:
#         g_pos = ground_positions[g_node]
        
#         matching_air_nodes = []
#         for a_node in G_air.nodes():
#             if air_node_types.get(a_node) == 'AIR_VTP':
#                 a_pos = air_positions[a_node]
#                 if abs(a_pos[0] - g_pos[0]) < 0.01 and abs(a_pos[1] - g_pos[1]) < 0.01:
#                     matching_air_nodes.append(a_node)
        
#         for a_node in matching_air_nodes:
#             a_pos = air_positions[a_node]
#             ax.plot(
#                 [g_pos[0], a_pos[0]],
#                 [g_pos[1], a_pos[1]],
#                 [g_pos[2], a_pos[2]],
#                 color='#457B9D',
#                 linestyle=(0, (1, 1)),  # 更小的虚线间隔
#                 linewidth=0.5,
#                 alpha=0.6,
#                 zorder=1  # 确保垂直线在背景上
#             )
    
#     # 添加客户地面节点到空中节点的虚线连接
#     customer_ground_nodes = [n for n in G_ground.nodes() if ground_node_types.get(n) == 'CUSTOMER']
#     for g_node in customer_ground_nodes:
#         g_pos = ground_positions[g_node]
        
#         matching_air_nodes = []
#         for a_node in G_air.nodes():
#             if air_node_types.get(a_node) == 'AIR_CUSTOMER':
#                 a_pos = air_positions[a_node]
#                 if abs(a_pos[0] - g_pos[0]) < 0.01 and abs(a_pos[1] - g_pos[1]) < 0.01:
#                     matching_air_nodes.append(a_node)
        
#         for a_node in matching_air_nodes:
#             a_pos = air_positions[a_node]
#             ax.plot(
#                 [g_pos[0], a_pos[0]],
#                 [g_pos[1], a_pos[1]],
#                 [g_pos[2], a_pos[2]],
#                 color='#2A9D8F',
#                 linestyle=(0, (1, 1)),
#                 linewidth=0.5,
#                 alpha=0.6,
#                 zorder=1
#             )
    
#     # 设置坐标轴范围和标签
#     x_values = [pos[0] for pos in air_positions.values()] + [pos[0] for pos in ground_positions.values()]
#     y_values = [pos[1] for pos in air_positions.values()] + [pos[1] for pos in ground_positions.values()]
#     z_values = [pos[2] for pos in air_positions.values()] + [pos[2] for pos in ground_positions.values()]
    
#     x_min, x_max = min(x_values), max(x_values)
#     y_min, y_max = min(y_values), max(y_values)
#     z_min, z_max = min(z_values), max(z_values)
    
#     # 增加一些边距
#     x_margin = (x_max - x_min) * 0.1
#     y_margin = (y_max - y_min) * 0.1
#     z_margin = max((z_max - z_min) * 0.1, 1)  # 确保至少有一些高度
    
#     ax.set_xlim(x_min - x_margin, x_max + x_margin)
#     ax.set_ylim(y_min - y_margin, y_max + y_margin)
#     ax.set_zlim(z_min, z_max + z_margin * 2)  # 增加顶部空间展示立体感
    
#     # # 设置坐标轴标签
#     # ax.set_xlabel('X', fontweight='bold', labelpad=5, fontsize = 8)
#     # ax.set_ylabel('Y', fontweight='bold', labelpad=5, fontsize = 8)
#     # ax.set_zlabel('Z', fontweight='bold', labelpad=5, fontsize = 8)
    
#     # 更轻的网格线
#     ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
    
#     # 设置刻度标记的格式和数量（减少刻度数量）
#     ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
#     ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
#     ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    
#     # 减少刻度数量
#     ax.xaxis.set_major_locator(plt.MaxNLocator(5))
#     ax.yaxis.set_major_locator(plt.MaxNLocator(5))
#     ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    
#     # 创建极小的自定义图例
#     from matplotlib.lines import Line2D
    
#     # 节点类型
#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
    
#     # 连接类型
#     custom_lines = [
#         Line2D([0], [0], color='#7c7c7c', lw=1, linestyle=(0, (2, 2))),
#         Line2D([0], [0], color='#457B9D', lw=1, linestyle=(0, (1, 1))),
#         Line2D([0], [0], color='#2A9D8F', lw=1, linestyle=(0, (1, 1)))
#     ]
    
#     # 整合所有图例元素
#     all_handles = list(by_label.values()) + custom_lines
#     # all_labels = list(by_label.keys()) + ['Air Route', 'VTP Link', 'Customer Link']
#     all_labels = list(by_label.keys()) + [
#     'Urban Aerial Corridor',       # 城市空中走廊
#     'VTP Aerial Relay',         # VTP对应空中中继点
#     'Customers Aerial Relay' # 客户点对应空中中继点
# ]
    
#     # 添加超小型图例到右下角
#     legend = ax.legend(
#         all_handles, all_labels,
#         # loc='lower right',
#         loc='upper right',
#         fontsize=4,        # 极小字体
#         ncol=3,            # 三列更紧凑
#         bbox_to_anchor=(1.02, 1.0),  # 关键修改
#         frameon=True,
#         framealpha=0.6,
#         edgecolor='black',
#         markerscale=0.4,   # 更小的图标
#         handlelength=1.0,  # 更短的线段
#         handletextpad=0.3, # 更小的间距
#         columnspacing=0.7, # 更小的列间距
#         borderpad=0.3,     # 更小的内边距
#     )
#     legend.get_frame().set_linewidth(0.5)
    
#     # 精简标题
#     plt.title('Vehicle-UAV Delivery Environment', fontweight='bold', pad=15, fontsize=12)
#     ax.set_xlabel('X (Operation Range, km)', fontweight='bold', labelpad=8, fontsize=8)  # 修改此处
#     ax.set_ylabel('Y (Operation Range, km)', fontweight='bold', labelpad=8, fontsize=8)  # 修改此处
#     ax.set_zlabel('Z (Altitude, m)', fontweight='bold', labelpad=8, fontsize=8)  # 修改此处
#     # 优化3D视角，更好地展示立体感
#     ax.view_init(elev=30, azim=45)
    
#     # 调整布局
#     plt.tight_layout()
#     save_path = r'tits_VDCD-AC\main_master\map_test\map_test.png'
#     if save_path:
#         plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    
#     plt.show()
    
#     return fig

# 示例代码
if __name__ == '__main__':
    # test_points,start_pos = generate_points(50, 6)  # 使用 generate_points 函数生成测试点
    # test_points = generates_points(50, 6)  # 使用 generates_points 函数生成测试点
    num_points = 50
    seed = 6
    Z_coord = 5  # 空中走廊高度
    uav_distance = 15  # 无人机最远飞行距离
    G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions, all_data, air_node_types, ground_node_types = generate_complex_network(num_points, seed, Z_coord, uav_distance)
    # G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions = generate_graph(test_points, 1)# 使用 generate_graph 函数生成空中图
    # 可视化塔杆节点和线路
    visualize_tower_connections_3D(G_air, G_ground, all_data, air_node_types, ground_node_types)
