#!/usr/bin/env python
"""
Drone task consistency checker for:
- vehicle_routes
- customer_plan
- base_drone_assignment

The checker returns:
1) bool: whether constraints are satisfied
2) dict: detailed error entries (vehicle/node/drone/customer/reason)
"""

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def build_uniform_base_drone_assignment(
    vehicle_ids: Sequence[int], drone_ids: Sequence[int]
) -> Dict[int, List[int]]:
    """
    Build the same "uniform split" assignment used by initialization logic.
    """
    vehicles = list(vehicle_ids)
    drones = list(drone_ids)
    if not vehicles:
        return {}

    per_vehicle = len(drones) // len(vehicles)
    remainder = len(drones) % len(vehicles)

    assignment: Dict[int, List[int]] = {}
    cursor = 0
    for idx, vehicle_id in enumerate(vehicles):
        count = per_vehicle + (1 if idx < remainder else 0)
        assignment[int(vehicle_id)] = drones[cursor : cursor + count]
        cursor += count
    return assignment


def validate_destroyed_state_constraints(
    destroyed_state: Any,
    base_drone_assignment: Mapping[int, Sequence[int]],
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience wrapper for state object usage.
    """
    return validate_customer_plan_constraints(
        vehicle_routes=getattr(destroyed_state, "vehicle_routes", None),
        customer_plan=getattr(destroyed_state, "customer_plan", None),
        base_drone_assignment=base_drone_assignment,
        verbose=verbose,
    )


def validate_customer_plan_constraints(
    vehicle_routes: Any,
    customer_plan: Any,
    base_drone_assignment: Mapping[int, Sequence[int]],
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate whether drone launch/recovery sequences are consistent with:
    - routes (ignoring first/last depot nodes)
    - initial drone-vehicle assignment
    - customer_plan tuple/list records:
      (drone_id, launch_node, customer_node, recovery_node, launch_vehicle, recovery_vehicle)
    """
    report: Dict[str, Any] = {"errors": [], "by_vehicle": {}, "by_drone": {}}

    routes = _normalize_vehicle_routes(vehicle_routes)
    if not routes:
        _add_error(
            report,
            vehicle_id=None,
            node_id=None,
            drone_id=None,
            customer_id=None,
            rule="invalid_routes",
            reason="vehicle_routes is empty or invalid.",
        )
        return False, _finalize_report(report, verbose)

    route_positions = _build_route_positions(routes)
    parsed_plan, parsing_ok = _normalize_customer_plan(customer_plan, report)
    if not parsing_ok:
        return False, _finalize_report(report, verbose)

    normalized_base = _normalize_base_assignment(base_drone_assignment)
    start_vehicle_of_drone: Dict[int, int] = {}
    for vehicle_id, drones in normalized_base.items():
        for drone_id in drones:
            if drone_id in start_vehicle_of_drone and start_vehicle_of_drone[drone_id] != vehicle_id:
                _add_error(
                    report,
                    vehicle_id=vehicle_id,
                    node_id=None,
                    drone_id=drone_id,
                    customer_id=None,
                    rule="duplicate_initial_assignment",
                    reason=(
                        f"Drone {drone_id} appears in multiple initial vehicles: "
                        f"{start_vehicle_of_drone[drone_id]} and {vehicle_id}."
                    ),
                )
            else:
                start_vehicle_of_drone[drone_id] = vehicle_id

    tasks_by_drone: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for task in parsed_plan:
        drone_id = task["drone_id"]
        tasks_by_drone[drone_id].append(task)
        _validate_single_task_nodes(task, route_positions, report)

    _simulate_drone_sequences(
        tasks_by_drone=tasks_by_drone,
        route_positions=route_positions,
        start_vehicle_of_drone=start_vehicle_of_drone,
        report=report,
    )

    return len(report["errors"]) == 0, _finalize_report(report, verbose)


def _normalize_vehicle_routes(vehicle_routes: Any) -> Dict[int, List[int]]:
    routes: Dict[int, List[int]] = {}
    if isinstance(vehicle_routes, Mapping):
        for vehicle_id, route in vehicle_routes.items():
            if route is None:
                continue
            routes[int(vehicle_id)] = list(route)
        return routes

    if isinstance(vehicle_routes, (list, tuple)):
        for idx, route in enumerate(vehicle_routes, start=1):
            if route is None:
                continue
            routes[idx] = list(route)
        return routes

    return {}


def _normalize_base_assignment(
    base_drone_assignment: Optional[Mapping[int, Sequence[int]]],
) -> Dict[int, List[int]]:
    if not base_drone_assignment:
        return {}

    normalized: Dict[int, List[int]] = {}
    for vehicle_id, drones in base_drone_assignment.items():
        if drones is None:
            normalized[int(vehicle_id)] = []
            continue
        if isinstance(drones, (list, tuple, set)):
            normalized[int(vehicle_id)] = [int(d) for d in drones]
        else:
            normalized[int(vehicle_id)] = [int(drones)]
    return normalized


def _normalize_customer_plan(
    customer_plan: Any, report: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], bool]:
    if not isinstance(customer_plan, Mapping):
        _add_error(
            report,
            vehicle_id=None,
            node_id=None,
            drone_id=None,
            customer_id=None,
            rule="invalid_customer_plan",
            reason="customer_plan must be a dict-like mapping.",
        )
        return [], False

    parsed: List[Dict[str, Any]] = []
    ok = True
    for customer_id, raw in customer_plan.items():
        task = _parse_task(customer_id, raw)
        if task is None:
            _add_error(
                report,
                vehicle_id=None,
                node_id=None,
                drone_id=None,
                customer_id=customer_id,
                rule="invalid_task_format",
                reason=f"Invalid task format for customer {customer_id}: {raw}",
            )
            ok = False
            continue
        parsed.append(task)
    return parsed, ok


def _parse_task(customer_id: Any, raw: Any) -> Optional[Dict[str, Any]]:
    try:
        if isinstance(raw, Mapping):
            drone_id = int(raw["drone_id"])
            launch_node = int(raw["launch_node"])
            recovery_node = int(raw["recovery_node"])
            launch_vehicle = int(raw["launch_vehicle"])
            recovery_vehicle = int(raw["recovery_vehicle"])
        elif isinstance(raw, (list, tuple)) and len(raw) >= 6:
            drone_id = int(raw[0])
            launch_node = int(raw[1])
            recovery_node = int(raw[3])
            launch_vehicle = int(raw[4])
            recovery_vehicle = int(raw[5])
        else:
            return None
    except Exception:
        return None

    return {
        "customer_id": customer_id,
        "drone_id": drone_id,
        "launch_node": launch_node,
        "recovery_node": recovery_node,
        "launch_vehicle": launch_vehicle,
        "recovery_vehicle": recovery_vehicle,
    }


def _build_route_positions(routes: Dict[int, List[int]]) -> Dict[int, Dict[int, List[int]]]:
    positions: Dict[int, Dict[int, List[int]]] = {}
    for vehicle_id, route in routes.items():
        node_to_positions: Dict[int, List[int]] = defaultdict(list)
        if len(route) >= 3:
            for idx in range(1, len(route) - 1):
                node_to_positions[route[idx]].append(idx)
        positions[vehicle_id] = dict(node_to_positions)
    return positions


def _validate_single_task_nodes(
    task: Dict[str, Any],
    route_positions: Dict[int, Dict[int, List[int]]],
    report: Dict[str, Any],
) -> None:
    customer_id = task["customer_id"]
    drone_id = task["drone_id"]
    launch_vehicle = task["launch_vehicle"]
    launch_node = task["launch_node"]
    recovery_vehicle = task["recovery_vehicle"]
    recovery_node = task["recovery_node"]

    launch_positions = route_positions.get(launch_vehicle, {}).get(launch_node, [])
    recovery_positions = route_positions.get(recovery_vehicle, {}).get(recovery_node, [])

    if launch_vehicle not in route_positions:
        _add_error(
            report,
            vehicle_id=launch_vehicle,
            node_id=launch_node,
            drone_id=drone_id,
            customer_id=customer_id,
            rule="launch_vehicle_missing",
            reason=f"Launch vehicle {launch_vehicle} does not exist in vehicle_routes.",
        )
        return

    if recovery_vehicle not in route_positions:
        _add_error(
            report,
            vehicle_id=recovery_vehicle,
            node_id=recovery_node,
            drone_id=drone_id,
            customer_id=customer_id,
            rule="recovery_vehicle_missing",
            reason=f"Recovery vehicle {recovery_vehicle} does not exist in vehicle_routes.",
        )
        return

    if not launch_positions:
        _add_error(
            report,
            vehicle_id=launch_vehicle,
            node_id=launch_node,
            drone_id=drone_id,
            customer_id=customer_id,
            rule="launch_node_not_in_route",
            reason=(
                f"Launch node {launch_node} is not in the interior path of vehicle "
                f"{launch_vehicle}."
            ),
        )
    if not recovery_positions:
        _add_error(
            report,
            vehicle_id=recovery_vehicle,
            node_id=recovery_node,
            drone_id=drone_id,
            customer_id=customer_id,
            rule="recovery_node_not_in_route",
            reason=(
                f"Recovery node {recovery_node} is not in the interior path of vehicle "
                f"{recovery_vehicle}."
            ),
        )

    if launch_vehicle == recovery_vehicle and launch_positions and recovery_positions:
        valid_pair_exists = False
        for launch_idx in launch_positions:
            if _first_index_greater_than(recovery_positions, launch_idx) is not None:
                valid_pair_exists = True
                break
        if not valid_pair_exists:
            _add_error(
                report,
                vehicle_id=launch_vehicle,
                node_id=launch_node,
                drone_id=drone_id,
                customer_id=customer_id,
                rule="same_vehicle_order_invalid",
                reason=(
                    "For same-vehicle mission, launch index must be before recovery index."
                ),
            )


def _simulate_drone_sequences(
    tasks_by_drone: Dict[int, List[Dict[str, Any]]],
    route_positions: Dict[int, Dict[int, List[int]]],
    start_vehicle_of_drone: Dict[int, int],
    report: Dict[str, Any],
) -> None:
    all_drones = set(tasks_by_drone.keys()) | set(start_vehicle_of_drone.keys())

    for drone_id in sorted(all_drones):
        drone_tasks = list(tasks_by_drone.get(drone_id, []))
        if not drone_tasks:
            continue

        if drone_id in start_vehicle_of_drone:
            current_vehicle = start_vehicle_of_drone[drone_id]
        else:
            first_task = sorted(
                drone_tasks,
                key=lambda t: (t["launch_vehicle"], t["launch_node"], str(t["customer_id"])),
            )[0]
            current_vehicle = first_task["launch_vehicle"]
            _add_error(
                report,
                vehicle_id=first_task["launch_vehicle"],
                node_id=first_task["launch_node"],
                drone_id=drone_id,
                customer_id=first_task["customer_id"],
                rule="missing_initial_assignment",
                reason=(
                    f"Drone {drone_id} appears in customer_plan but not in base_drone_assignment."
                ),
            )

        current_index = 0
        remaining = list(drone_tasks)
        max_steps = max(5, len(remaining) * 3)
        steps = 0

        while remaining and steps < max_steps:
            steps += 1

            launch_candidates: List[Tuple[int, Dict[str, Any]]] = []
            for task in remaining:
                if task["launch_vehicle"] != current_vehicle:
                    continue
                launch_positions = route_positions.get(current_vehicle, {}).get(
                    task["launch_node"], []
                )
                launch_idx = _first_index_greater_than(launch_positions, current_index)
                if launch_idx is not None:
                    launch_candidates.append((launch_idx, task))

            if not launch_candidates:
                for task in remaining:
                    _add_error(
                        report,
                        vehicle_id=task["launch_vehicle"],
                        node_id=task["launch_node"],
                        drone_id=drone_id,
                        customer_id=task["customer_id"],
                        rule="task_without_available_drone",
                        reason=(
                            f"Drone {drone_id} is on vehicle {current_vehicle} (index "
                            f"{current_index}), but remaining task requires launch on "
                            f"vehicle {task['launch_vehicle']}."
                        ),
                    )
                remaining = []
                break

            launch_candidates.sort(
                key=lambda x: (x[0], x[1]["launch_node"], str(x[1]["customer_id"]))
            )
            launch_idx, chosen_task = launch_candidates[0]

            same_node_conflicts = [
                task
                for idx, task in launch_candidates[1:]
                if idx == launch_idx and task["launch_node"] == chosen_task["launch_node"]
            ]
            for conflict_task in same_node_conflicts:
                _add_error(
                    report,
                    vehicle_id=current_vehicle,
                    node_id=chosen_task["launch_node"],
                    drone_id=drone_id,
                    customer_id=conflict_task["customer_id"],
                    rule="multiple_launch_same_node",
                    reason=(
                        f"Drone {drone_id} has multiple launch tasks at the same vehicle/node "
                        f"({current_vehicle}, {chosen_task['launch_node']})."
                    ),
                )

            rec_vehicle = chosen_task["recovery_vehicle"]
            rec_node = chosen_task["recovery_node"]
            rec_positions = route_positions.get(rec_vehicle, {}).get(rec_node, [])

            if rec_vehicle == current_vehicle:
                rec_idx = _first_index_greater_than(rec_positions, launch_idx)
                if rec_idx is None:
                    _add_error(
                        report,
                        vehicle_id=rec_vehicle,
                        node_id=rec_node,
                        drone_id=drone_id,
                        customer_id=chosen_task["customer_id"],
                        rule="recovery_before_launch_or_missing",
                        reason=(
                            "Recovery is not after launch on the same vehicle route."
                        ),
                    )
                    remaining.remove(chosen_task)
                    continue

                for other_task in remaining:
                    if other_task is chosen_task:
                        continue
                    if other_task["launch_vehicle"] != current_vehicle:
                        continue
                    other_positions = route_positions.get(current_vehicle, {}).get(
                        other_task["launch_node"], []
                    )
                    other_idx = _first_index_greater_than(other_positions, launch_idx)
                    if other_idx is not None and other_idx < rec_idx:
                        _add_error(
                            report,
                            vehicle_id=current_vehicle,
                            node_id=other_task["launch_node"],
                            drone_id=drone_id,
                            customer_id=other_task["customer_id"],
                            rule="overlap_same_vehicle_task",
                            reason=(
                                "Drone has another launch task before the current mission "
                                "is recovered on the same vehicle."
                            ),
                        )
            else:
                if not rec_positions:
                    _add_error(
                        report,
                        vehicle_id=rec_vehicle,
                        node_id=rec_node,
                        drone_id=drone_id,
                        customer_id=chosen_task["customer_id"],
                        rule="cross_vehicle_recovery_missing",
                        reason="Cross-vehicle recovery node is missing in recovery vehicle route.",
                    )
                    remaining.remove(chosen_task)
                    continue

                rec_idx = rec_positions[0]

                for other_task in remaining:
                    if other_task is chosen_task:
                        continue
                    if other_task["launch_vehicle"] != current_vehicle:
                        continue
                    other_positions = route_positions.get(current_vehicle, {}).get(
                        other_task["launch_node"], []
                    )
                    other_idx = _first_index_greater_than(other_positions, launch_idx)
                    if other_idx is not None:
                        _add_error(
                            report,
                            vehicle_id=current_vehicle,
                            node_id=other_task["launch_node"],
                            drone_id=drone_id,
                            customer_id=other_task["customer_id"],
                            rule="task_after_cross_vehicle_launch",
                            reason=(
                                "Drone left this vehicle via cross-vehicle recovery but still "
                                "has later launch task on the original vehicle."
                            ),
                        )

            remaining.remove(chosen_task)
            current_vehicle = rec_vehicle
            current_index = rec_idx

        if remaining:
            for task in remaining:
                _add_error(
                    report,
                    vehicle_id=task["launch_vehicle"],
                    node_id=task["launch_node"],
                    drone_id=drone_id,
                    customer_id=task["customer_id"],
                    rule="unresolved_task",
                    reason="Task cannot be placed into a feasible launch/recovery sequence.",
                )


def _first_index_greater_than(indexes: Sequence[int], threshold: int) -> Optional[int]:
    for idx in indexes:
        if idx > threshold:
            return idx
    return None


def _add_error(
    report: Dict[str, Any],
    vehicle_id: Optional[int],
    node_id: Optional[int],
    drone_id: Optional[int],
    customer_id: Any,
    rule: str,
    reason: str,
) -> None:
    entry = {
        "vehicle_id": vehicle_id,
        "node_id": node_id,
        "drone_id": drone_id,
        "customer_id": customer_id,
        "rule": rule,
        "reason": reason,
    }
    report["errors"].append(entry)

    if vehicle_id is not None:
        vehicle_bucket = report["by_vehicle"].setdefault(vehicle_id, [])
        vehicle_bucket.append(entry)

    if drone_id is not None:
        drone_bucket = report["by_drone"].setdefault(drone_id, [])
        drone_bucket.append(entry)


def _finalize_report(report: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
    report["error_count"] = len(report["errors"])
    report["is_valid"] = report["error_count"] == 0

    if verbose:
        if report["is_valid"]:
            print("[ConstraintCheck] pass")
        else:
            print(f"[ConstraintCheck] failed: {report['error_count']} issue(s)")
            for issue in report["errors"]:
                print(
                    f"  - vehicle={issue['vehicle_id']} node={issue['node_id']} "
                    f"drone={issue['drone_id']} customer={issue['customer_id']} "
                    f"rule={issue['rule']} reason={issue['reason']}"
                )
    return report
