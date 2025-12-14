VDAC框架数据结构类型。

win_cost：记录每代**不考虑空中走廊冲突**的纯时间窗违背成本。

uav_route_cost：记录每代**不考虑空中走廊冲突**的无人机路径规划配送成本。

vehicle_route_cost：记录每代**不考虑空中走廊冲突**的车辆路径配送成本。

final_uav_cost：记录每代**考虑空中走廊冲突**的无人机配送成本。

final_total_list：记录每代**考虑空中走廊冲突**的无人机配送总成本，配送成本+违约时间窗口的。

final_win_cost：记录每代**考虑空中走廊冲突**的无人机纯违背时间窗总成本

final_total_objective：记录每代**考虑空中走廊冲突**的总成本(违背时间窗+无人机+车辆)

y_cost:记录每代不**考虑空中走廊冲突**的总成本(违背时间窗+无人机+车辆)。

y_best：记录每代不**考虑空中走廊冲突**的总成本(违背时间窗+无人机+车辆)，只记录迭代后变优秀的，用于绘制曲线图案。

work_time：记录每代**不考虑空中走廊冲突**的完成任务时间。

final_work_time:记录每代**考虑空中走廊冲突**的完成任务时间。

best_final_uav_cost：最优方案中**考虑空中走廊冲突**的无人机成本。

best_final_objective：最优方案中的**最终成本**（违背时间窗+无人机+车辆)。

best_final_win_cost：最优方案中**考虑空中走廊冲突**的违背时间窗成本。

best_total_win_cost：最优方案中**考虑空中走廊冲突**的时间窗违背总成本+无人机路径成本。

best_final_global_max_time：最优方案中**考虑空中走廊冲突**的最终完成时间。

elapsed_time：完成算法运算的所有时间。

best_global_max_time：最优方案的整体任务完成时间。

best_window_total_cost：最优的方案中不考虑空中走廊冲突的时间窗惩罚成本+无人机路径

best_total_uav_tw_violation_cost:最优方案中不考虑空中走廊冲突的无人机违背惩罚成本情况。

best_total_vehicle_cost: 最优方案中不考虑空中走廊冲突的车辆路径成本。

