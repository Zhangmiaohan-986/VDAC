import random
import numpy as np
import copy
import numpy.random as rnd
from collections import defaultdict
import time
from parseCSV import *

from collections import defaultdict
import copy
from initialize import init_agent, initialize_drone_vehicle_assignments
from create_vehicle_route import *

import os
from main import find_keys_and_indices
from mfstsp_heuristic_1_partition import *
from mfstsp_heuristic_2_asgn_uavs import *
from mfstsp_heuristic_3_timing import *

from local_search import *
from rm_node_sort_node import rm_empty_node

import main
import endurance_calculator
import distance_functions


def destroy_random_removal(state):
    return None



