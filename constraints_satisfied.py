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
from initialize import deep_copy_vehicle_task_data

def is_constraints_satisfied(repaired_state, vehicle_task_data, scheme):
    # 检查是否满足约束条件
    is_satisfied = True
    temp_vehicle_task_data = deep_copy_vehicle_task_data(vehicle_task_data)
    temp_vehicle_route = [route[:] for route in repaired_state.vehicle_routes]
    try:
        temp_vehicle_task_data = update_vehicle_task(temp_vehicle_task_data, scheme, temp_vehicle_route)
        return True
    except Exception as e:
        print(f"在修复阶段产生违背约束条件的方案，定位在约束检查中constraints_satisfied.py")
        return False
