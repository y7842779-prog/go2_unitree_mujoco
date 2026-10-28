import argparse
import csv
import json
import math
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence
import mujoco
import numpy as np
import torch


"""
命令跟踪评估脚本
================

评估流程对齐论文描述：
1. 机器人在 MuJoCo 中连续运行 10 分钟。
2. 每 10 秒从 [-1, 1] 范围内为 [vx, vy, yaw_rate] 均匀采样一次随机指令。
3. 所有控制器在同一个 seed 下共用完全一致的随机指令序列。
4. 每个控制器用 5 个不同随机 seed 重复运行。
5. 使用 ATE (Absolute Tracking Error) 作为指标：

   ATE = mean_t( mean( abs([vx, vy, yaw]cmd - [vx, vy, yaw]meas) ) )

脚本输出：
- run_metrics.csv            每次运行的 ATE 结果（每个 seed 一个平均值）
- pairwise_t_tests.csv      相对参考控制器的配对 t 检验（仍基于每个 seed 的平均 ATE）
- axis_pairwise_t_tests.csv 分轴配对 t 检验（仍基于每个 seed 的平均 ATE）
- ate_barplot.png           ATE 条形图（均值 ± 标准差）
- tracking_error_boxplot.png 论文式逐时刻误差 boxplot（5 个 seed 内所有时刻误差合并）
- instant_boxplot_stats.csv tracking_error_boxplot.png 使用的 min/Q1/median/Q3/max 统计量
- command_sequences.csv     每个 seed 对应的随机指令时间表
- summary.json              便于后续做论文表格/画图的汇总文件

如需增加更多控制器，直接在 CONTROLLERS 里继续加配置即可。

默认是“离屏批量评估”，不会打开 MuJoCo viewer。
如需可视化，请加 `--viewer`。
"""


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_XML_PATH = (SCRIPT_DIR / "../unitree_robots/go2/scene_flat.xml").resolve()
DEFAULT_OUTPUT_ROOT = (SCRIPT_DIR / "eval_outputs").resolve()

# DreamWaQ / CENet 模型配置
HISTORY_LEN = 5
CURRENT_OBS_DIM = 45
POLICY_OBS_DIM = 270
NUM_ACTIONS = 12

# 控制参数直接复用 dreamwaq_cenet.py
ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5
ACTION_CLIP = 14.0
DECIMATION = 10
POLICY_RAMP_TIME = 0.5
INIT_BASE_HEIGHT = 0.445
FALL_BASE_HEIGHT_THRESHOLD = 0.18

DEFAULT_DOF_POS_LEG = np.array(
    [
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        1.0,
        -1.5,
        -0.1,
        1.0,
        -1.5,
    ],
    dtype=np.float32,
)

LEG2ISAACLAB_ORDER = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ISAACLAB2LEG_ORDER = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
CTRL_MUJOCO_IDX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
TORQUE_LIMIT_LEG = np.array(
    [
        23.7,
        23.7,
        45.43,
        23.7,
        23.7,
        45.43,
        23.7,
        23.7,
        45.43,
        23.7,
        23.7,
        45.43,
    ],
    dtype=np.float32,
)

COMMAND_LOW = np.array([-1, -1, -1], dtype=np.float32)
COMMAND_HIGH = np.array([1, 1, 1], dtype=np.float32)


@dataclass(frozen=True)
class ControllerConfig:
    name: str
    policy_path: Path
    display_name: str | None = None
    color: str | None = None


@dataclass
class RunResult:
    controller: str
    seed: int
    ate_total: float
    ate_vx: float
    ate_vy: float
    ate_yaw: float
    fell: bool
    simulated_time_s: float
    evaluated_samples: int
    # shape = [num_samples, 3], columns = [abs_err_vx, abs_err_vy, abs_err_yaw]
    # 这个字段只用于“论文式逐时刻误差 boxplot”，不会写入 run_metrics.csv。
    instant_errors: np.ndarray | None = None


CONTROLLERS: List[ControllerConfig] = [
    ControllerConfig(
        name="DreamWaQ_AdaBoot",
        policy_path=(SCRIPT_DIR / "DreamWaQ_models/policy_5N_foce.pt").resolve(),
        display_name="DreamWaQ w/ AdaBoot",
        color="#3776ab",
    ),
    ControllerConfig(
        name="DreamWaQ_CENet",
        policy_path=(SCRIPT_DIR / "DreamWaQ_models/policy_cenet.pt").resolve(),
        display_name="DreamWaQ w/ CENet",
        color="#d84b3a",
    ),
    ControllerConfig(
        name="DreamWaQ_NoForce",
        policy_path=(SCRIPT_DIR / "DreamWaQ_models/policy_no_foce.pt").resolve(),
        display_name="DreamWaQ w/o Force",
        color="#7a5ea8",
    ),
]

REFERENCE_CONTROLLER = "DreamWaQ_AdaBoot"
DEFAULT_CONTROLLER_SELECTION = "DreamWaQ_AdaBoot,DreamWaQ_CENet"
DEFAULT_CONTROLLER_COLORS = ["#3776ab", "#f08a24", "#3c963c", "#d64541", "#8f6fb0", "#6d6d6d"]
AXIS_METRICS = [
    ("ate_vx", r"$v_x^e$"),
    ("ate_vy", r"$v_y^e$"),
    ("ate_yaw", r"$\omega_z^e$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo command tracking ATE evaluation")
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=DEFAULT_XML_PATH,
        help="MuJoCo XML path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="有效评估时长（秒），默认 600s",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="起步稳定时间（秒），warmup 内不计入指标",
    )
    parser.add_argument(
        "--command-interval",
        type=float,
        default=10.0,
        help="随机指令重采样周期（秒）",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="逗号分隔的随机种子列表",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default=DEFAULT_CONTROLLER_SELECTION,
        help='要运行的控制器名称，逗号分隔；默认只比较 DreamWaQ_AdaBoot 和 DreamWaQ_CENet，显式传 "all" 才会包含全部控制器',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认自动生成时间戳子目录",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="保存每个运行的逐时刻轨迹 CSV",
    )
    parser.add_argument(
        "--fall-penalty",
        type=float,
        default=2.0,
        help="机器人跌倒后，对剩余样本填充的单轴绝对误差惩罚值",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="打开 MuJoCo viewer，可视化当前评估运行",
    )
    parser.add_argument(
        "--plot-only-run-metrics",
        type=Path,
        default=None,
        help="从已有 run_metrics.csv 重绘图，不重新运行仿真",
    )
    parser.add_argument(
        "--plot-only-trace-dir",
        type=Path,
        default=None,
        help="配合 --plot-only-run-metrics 使用：从已有 traces 目录读取逐时刻误差，重绘论文式 boxplot",
    )
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("至少需要一个随机 seed")
    return seeds


def resolve_controllers(selection: str) -> List[ControllerConfig]:
    if selection.strip().lower() == "all":
        return CONTROLLERS

    wanted = {name.strip() for name in selection.split(",") if name.strip()}
    selected = [cfg for cfg in CONTROLLERS if cfg.name in wanted]
    missing = sorted(wanted - {cfg.name for cfg in selected})
    if missing:
        raise ValueError(f"未知控制器: {missing}")
    return selected


def get_controller_config(controller_name: str) -> ControllerConfig | None:
    for controller in CONTROLLERS:
        if controller.name == controller_name:
            return controller
    return None


def get_controller_display_name(controller_name: str) -> str:
    controller = get_controller_config(controller_name)
    if controller is not None and controller.display_name:
        return controller.display_name
    return controller_name


def get_controller_color(controller_name: str, fallback_index: int) -> str:
    controller = get_controller_config(controller_name)
    if controller is not None and controller.color:
        return controller.color
    return DEFAULT_CONTROLLER_COLORS[fallback_index % len(DEFAULT_CONTROLLER_COLORS)]


def get_controller_order(results: Sequence[RunResult]) -> List[str]:
    available = {result.controller for result in results}
    ordered = [controller.name for controller in CONTROLLERS if controller.name in available]
    remaining = sorted(available - set(ordered))
    return ordered + remaining


def filter_results_by_controller_names(
    results: Sequence[RunResult],
    selection: str,
) -> List[RunResult]:
    if selection.strip().lower() == "all":
        return list(results)

    wanted = {name.strip() for name in selection.split(",") if name.strip()}
    filtered = [result for result in results if result.controller in wanted]
    if not filtered:
        raise ValueError(f"run_metrics.csv 中找不到这些控制器: {sorted(wanted)}")
    return filtered


def find_body_id(model: mujoco.MjModel, possible_names: Sequence[str]) -> int:
    for name in possible_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            return body_id
    raise RuntimeError(f"找不到 base body，尝试过: {possible_names}")


def get_joint_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joint_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[0]) == -1:
        joint_names = [name.replace("_joint", "") for name in joint_names]

    qpos_idx = []
    qvel_idx = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise RuntimeError(f"找不到关节: {name}")
        qpos_idx.append(model.jnt_qposadr[joint_id])
        qvel_idx.append(model.jnt_dofadr[joint_id])

    return np.asarray(qpos_idx, dtype=np.int32), np.asarray(qvel_idx, dtype=np.int32)


def build_current_obs(
    data: mujoco.MjData,
    base_body_id: int,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    last_action_leg: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    qpos_leg = data.qpos[qpos_idx].copy().astype(np.float32)
    qvel_leg = data.qvel[qvel_idx].copy().astype(np.float32)
    joint_pos_rel_leg = qpos_leg - DEFAULT_DOF_POS_LEG

    base_mat = data.xmat[base_body_id].reshape(3, 3).copy()
    base_ang_vel = data.qvel[3:6].copy().astype(np.float32)
    projected_gravity = (base_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(
        np.float32
    )

    current_obs = np.concatenate(
        [
            base_ang_vel,
            projected_gravity,
            command,
            joint_pos_rel_leg[LEG2ISAACLAB_ORDER],
            qvel_leg[LEG2ISAACLAB_ORDER],
            last_action_leg[LEG2ISAACLAB_ORDER],
        ]
    ).astype(np.float32)

    if current_obs.shape[0] != CURRENT_OBS_DIM:
        raise RuntimeError(f"current_obs 维度错误: {current_obs.shape[0]}")
    return current_obs


def build_policy_obs(current_obs: np.ndarray, history_buffer: deque[np.ndarray]) -> np.ndarray:
    while len(history_buffer) < HISTORY_LEN:
        history_buffer.append(current_obs.copy())

    history_frames = np.stack(list(history_buffer), axis=0).astype(np.float32)
    term_slices = (
        slice(0, 3),
        slice(3, 6),
        slice(6, 9),
        slice(9, 21),
        slice(21, 33),
        slice(33, 45),
    )
    history_obs = np.concatenate(
        [history_frames[:, term_slice].reshape(-1) for term_slice in term_slices]
    ).astype(np.float32)
    policy_obs = np.concatenate([current_obs, history_obs]).astype(np.float32)

    if policy_obs.shape[0] != POLICY_OBS_DIM:
        raise RuntimeError(f"policy_obs 维度错误: {policy_obs.shape[0]}")
    return policy_obs


def reset_sim_state(
    model: mujoco.MjModel,
    qpos_idx: np.ndarray,
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[2] = INIT_BASE_HEIGHT
    for i in range(NUM_ACTIONS):
        data.qpos[qpos_idx[i]] = DEFAULT_DOF_POS_LEG[i]
    mujoco.mj_forward(model, data)
    return data


def generate_command_sequence(
    seed: int,
    duration_s: float,
    interval_s: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_segments = int(math.ceil(duration_s / interval_s))
    commands = rng.uniform(COMMAND_LOW, COMMAND_HIGH, size=(num_segments, 3)).astype(np.float32)
    return commands


def get_command_for_time(commands: np.ndarray, eval_time_s: float, interval_s: float) -> np.ndarray:
    segment_idx = min(int(eval_time_s // interval_s), len(commands) - 1)
    return commands[segment_idx]


def get_tracking_measurement(data: mujoco.MjData, base_body_id: int) -> np.ndarray:
    base_mat = data.xmat[base_body_id].reshape(3, 3)
    base_lin_vel_world = data.qvel[0:3].copy()
    base_lin_vel_body = (base_mat.T @ base_lin_vel_world).astype(np.float32)
    base_ang_vel_body = data.qvel[3:6].copy().astype(np.float32)
    return np.array(
        [base_lin_vel_body[0], base_lin_vel_body[1], base_ang_vel_body[2]],
        dtype=np.float32,
    )


def run_single_controller_seed(
    controller: ControllerConfig,
    model: mujoco.MjModel,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    base_body_id: int,
    policy: torch.jit.ScriptModule,
    commands: np.ndarray,
    seed: int,
    duration_s: float,
    warmup_s: float,
    command_interval_s: float,
    fall_penalty: float,
    enable_viewer: bool,
    save_traces: bool,
    trace_dir: Path | None,
) -> RunResult:
    data = reset_sim_state(model, qpos_idx)
    control_dt = model.opt.timestep * DECIMATION
    total_time_s = warmup_s + duration_s
    total_steps = int(math.ceil(total_time_s / model.opt.timestep))

    actions_leg = np.zeros(NUM_ACTIONS, dtype=np.float32)
    history_buffer: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)

    zero_command = np.zeros(3, dtype=np.float32)
    first_current_obs = build_current_obs(
        data=data,
        base_body_id=base_body_id,
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        last_action_leg=actions_leg,
        command=zero_command,
    )
    for _ in range(HISTORY_LEN):
        history_buffer.append(first_current_obs.copy())

    times: List[float] = []
    cmd_log: List[np.ndarray] = []
    meas_log: List[np.ndarray] = []
    err_log: List[np.ndarray] = []
    fell = False
    simulated_time_s = 0.0
    viewer_context = None
    viewer = None
    viewer_closed_early = False

    if enable_viewer:
        try:
            from mujoco import viewer as mujoco_viewer

            viewer_context = mujoco_viewer.launch_passive(model, data)
            viewer = viewer_context.__enter__()
        except Exception as exc:
            print(f"[WARN] viewer 启动失败，回退到无界面模式: {exc}")
            viewer_context = None
            viewer = None

    try:
        block_wall_start = None

        for step_counter in range(total_steps):
            if step_counter % DECIMATION == 0:
                block_wall_start = time.time()

            sim_time = step_counter * model.opt.timestep
            qpos_leg = data.qpos[qpos_idx].copy()
            qvel_leg = data.qvel[qvel_idx].copy()

            if viewer is not None and not viewer.is_running():
                viewer_closed_early = True
                simulated_time_s = sim_time
                break

            if step_counter % DECIMATION == 0:
                if data.xpos[base_body_id][2] < FALL_BASE_HEIGHT_THRESHOLD and sim_time >= warmup_s:
                    fell = True
                    simulated_time_s = sim_time
                    break

                if sim_time < warmup_s:
                    command = zero_command
                else:
                    eval_time_s = sim_time - warmup_s
                    command = get_command_for_time(commands, eval_time_s, command_interval_s)
                    measurement = get_tracking_measurement(data, base_body_id)
                    abs_error = np.abs(command - measurement)
                    times.append(eval_time_s)
                    cmd_log.append(command.copy())
                    meas_log.append(measurement.copy())
                    err_log.append(abs_error.copy())

                current_obs = build_current_obs(
                    data=data,
                    base_body_id=base_body_id,
                    qpos_idx=qpos_idx,
                    qvel_idx=qvel_idx,
                    last_action_leg=actions_leg,
                    command=command,
                )
                history_buffer.append(current_obs.copy())

                if sim_time < warmup_s:
                    actions_leg = np.zeros(NUM_ACTIONS, dtype=np.float32)
                else:
                    obs_np = build_policy_obs(current_obs, history_buffer)
                    obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0)
                    with torch.no_grad():
                        raw_actions_net = policy(obs_tensor)
                        if isinstance(raw_actions_net, (tuple, list)):
                            raw_actions_net = raw_actions_net[0]
                        raw_actions_net = raw_actions_net.cpu().numpy().squeeze().astype(np.float32)

                    if raw_actions_net.shape[0] != NUM_ACTIONS:
                        raise RuntimeError(
                            f"{controller.name} action 维度错误: {raw_actions_net.shape}, 期待 12"
                        )

                    raw_actions_leg = raw_actions_net[ISAACLAB2LEG_ORDER]
                    clipped_actions_leg = np.clip(raw_actions_leg, -ACTION_CLIP, ACTION_CLIP)

                    if sim_time < warmup_s + POLICY_RAMP_TIME:
                        alpha = (sim_time - warmup_s) / POLICY_RAMP_TIME
                        alpha = float(np.clip(alpha, 0.0, 1.0))
                        actions_leg = clipped_actions_leg * alpha
                    else:
                        actions_leg = clipped_actions_leg

            target_dof_pos_leg = DEFAULT_DOF_POS_LEG + ACTION_SCALE * actions_leg
            tau_leg = KP * (target_dof_pos_leg - qpos_leg) - KD * qvel_leg
            tau_leg = np.clip(tau_leg, -TORQUE_LIMIT_LEG, TORQUE_LIMIT_LEG)
            tau_ctrl = tau_leg[CTRL_MUJOCO_IDX]

            if data.ctrl.shape[0] == NUM_ACTIONS:
                data.ctrl[:] = tau_ctrl
            else:
                data.ctrl[:NUM_ACTIONS] = tau_ctrl

            mujoco.mj_step(model, data)
            simulated_time_s = sim_time + model.opt.timestep

            if viewer is not None and (step_counter + 1) % DECIMATION == 0:
                viewer.sync()
                if block_wall_start is not None:
                    elapsed = time.time() - block_wall_start
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)

    expected_samples = int(math.ceil(duration_s / control_dt))
    if fell and len(err_log) < expected_samples:
        remaining = expected_samples - len(err_log)
        penalty_vec = np.full(3, fall_penalty, dtype=np.float32)
        for pad_idx in range(remaining):
            times.append(duration_s - remaining * control_dt + pad_idx * control_dt)
            cmd_log.append(np.full(3, np.nan, dtype=np.float32))
            meas_log.append(np.full(3, np.nan, dtype=np.float32))
            err_log.append(penalty_vec.copy())
    elif not fell:
        simulated_time_s = total_time_s

    if viewer_closed_early:
        print("[WARN] viewer 被关闭，当前运行提前结束；本次指标只基于已采样时段。")

    if not err_log:
        raise RuntimeError(f"{controller.name} 在 seed={seed} 下没有采到任何评估样本")

    errors = np.stack(err_log, axis=0)
    ate_axes = errors.mean(axis=0)
    ate_total = float(ate_axes.mean())

    if save_traces and trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"trace_{controller.name}_seed{seed}.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "eval_time_s",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_yaw",
                    "meas_vx",
                    "meas_vy",
                    "meas_yaw",
                    "abs_err_vx",
                    "abs_err_vy",
                    "abs_err_yaw",
                ]
            )
            for time_s, cmd, meas, err in zip(times, cmd_log, meas_log, err_log):
                writer.writerow(
                    [
                        f"{time_s:.6f}",
                        *[f"{value:.6f}" for value in cmd],
                        *[f"{value:.6f}" for value in meas],
                        *[f"{value:.6f}" for value in err],
                    ]
                )

    return RunResult(
        controller=controller.name,
        seed=seed,
        ate_total=ate_total,
        ate_vx=float(ate_axes[0]),
        ate_vy=float(ate_axes[1]),
        ate_yaw=float(ate_axes[2]),
        fell=fell,
        simulated_time_s=simulated_time_s,
        evaluated_samples=len(err_log),
        instant_errors=errors,
    )


def summarize_results(results: Sequence[RunResult]) -> Dict[str, Dict[str, float]]:
    by_controller: Dict[str, List[RunResult]] = {}
    for result in results:
        by_controller.setdefault(result.controller, []).append(result)

    summary: Dict[str, Dict[str, float]] = {}
    for controller_name, runs in by_controller.items():
        ate_total = np.array([run.ate_total for run in runs], dtype=np.float64)
        ate_vx = np.array([run.ate_vx for run in runs], dtype=np.float64)
        ate_vy = np.array([run.ate_vy for run in runs], dtype=np.float64)
        ate_yaw = np.array([run.ate_yaw for run in runs], dtype=np.float64)
        summary[controller_name] = {
            "mean_ate_total": float(ate_total.mean()),
            "std_ate_total": float(ate_total.std(ddof=1)) if len(ate_total) > 1 else 0.0,
            "mean_ate_vx": float(ate_vx.mean()),
            "mean_ate_vy": float(ate_vy.mean()),
            "mean_ate_yaw": float(ate_yaw.mean()),
            "num_runs": float(len(runs)),
            "num_falls": float(sum(run.fell for run in runs)),
        }
    return summary


def paired_t_tests_for_metric(
    results: Sequence[RunResult],
    reference_controller: str,
    metric_key: str,
) -> List[Dict[str, float | str]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
            from scipy import stats
    except Exception as exc:
        print(f"[WARN] SciPy 不可用，跳过 {metric_key} 的配对 t 检验: {exc}")
        return []

    per_controller: Dict[str, Dict[int, RunResult]] = {}
    for result in results:
        per_controller.setdefault(result.controller, {})[result.seed] = result

    if reference_controller not in per_controller:
        return []

    reference_runs = per_controller[reference_controller]
    comparisons: List[Dict[str, float | str]] = []

    for controller_name, run_map in per_controller.items():
        if controller_name == reference_controller:
            continue

        shared_seeds = sorted(set(reference_runs.keys()) & set(run_map.keys()))
        if len(shared_seeds) < 2:
            continue

        ref_values = np.array(
            [float(getattr(reference_runs[seed], metric_key)) for seed in shared_seeds],
            dtype=np.float64,
        )
        other_values = np.array(
            [float(getattr(run_map[seed], metric_key)) for seed in shared_seeds],
            dtype=np.float64,
        )

        # 使用 baseline - reference 的差值做单侧配对 t 检验：
        # H1: reference 的 ATE 更小，即 baseline - reference > 0
        t_stat, p_two_sided = stats.ttest_rel(other_values, ref_values)
        if np.isnan(t_stat) or np.isnan(p_two_sided):
            p_one_sided = float("nan")
        elif t_stat >= 0.0:
            p_one_sided = float(p_two_sided / 2.0)
        else:
            p_one_sided = float(1.0 - p_two_sided / 2.0)

        comparisons.append(
            {
                "reference_controller": reference_controller,
                "other_controller": controller_name,
                "metric_key": metric_key,
                "num_pairs": float(len(shared_seeds)),
                "t_statistic": float(t_stat),
                "p_value_one_sided": p_one_sided,
                "reference_mean": float(ref_values.mean()),
                "other_mean": float(other_values.mean()),
            }
        )

    return comparisons


def paired_t_tests(
    results: Sequence[RunResult],
    reference_controller: str,
) -> List[Dict[str, float | str]]:
    return paired_t_tests_for_metric(results, reference_controller, "ate_total")


def axis_paired_t_tests(
    results: Sequence[RunResult],
    reference_controller: str,
) -> List[Dict[str, float | str]]:
    comparisons: List[Dict[str, float | str]] = []
    for metric_key, _ in AXIS_METRICS:
        comparisons.extend(paired_t_tests_for_metric(results, reference_controller, metric_key))
    return comparisons


def save_command_sequences(
    output_dir: Path,
    command_sequences: Dict[int, np.ndarray],
    interval_s: float,
    duration_s: float,
) -> None:
    path = output_dir / "command_sequences.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "segment_id", "start_time_s", "end_time_s", "cmd_vx", "cmd_vy", "cmd_yaw"])
        for seed, commands in command_sequences.items():
            for idx, command in enumerate(commands):
                start_time = idx * interval_s
                end_time = min(duration_s, (idx + 1) * interval_s)
                writer.writerow(
                    [
                        seed,
                        idx,
                        f"{start_time:.6f}",
                        f"{end_time:.6f}",
                        f"{command[0]:.6f}",
                        f"{command[1]:.6f}",
                        f"{command[2]:.6f}",
                    ]
                )


def save_run_metrics(output_dir: Path, results: Sequence[RunResult]) -> None:
    path = output_dir / "run_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "controller",
                "seed",
                "ate_total",
                "ate_vx",
                "ate_vy",
                "ate_yaw",
                "fell",
                "simulated_time_s",
                "evaluated_samples",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.controller,
                    result.seed,
                    f"{result.ate_total:.8f}",
                    f"{result.ate_vx:.8f}",
                    f"{result.ate_vy:.8f}",
                    f"{result.ate_yaw:.8f}",
                    int(result.fell),
                    f"{result.simulated_time_s:.6f}",
                    result.evaluated_samples,
                ]
            )


def save_pairwise_tests(output_dir: Path, comparisons: Sequence[Dict[str, float | str]]) -> None:
    path = output_dir / "pairwise_t_tests.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reference_controller",
                "other_controller",
                "num_pairs",
                "t_statistic",
                "p_value_one_sided",
                "reference_mean",
                "other_mean",
            ]
        )
        for item in comparisons:
            writer.writerow(
                [
                    item["reference_controller"],
                    item["other_controller"],
                    int(item["num_pairs"]),
                    f"{float(item['t_statistic']):.8f}",
                    f"{float(item['p_value_one_sided']):.8f}",
                    f"{float(item['reference_mean']):.8f}",
                    f"{float(item['other_mean']):.8f}",
                ]
            )


def save_axis_pairwise_tests(output_dir: Path, comparisons: Sequence[Dict[str, float | str]]) -> None:
    path = output_dir / "axis_pairwise_t_tests.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "reference_controller",
                "other_controller",
                "metric_key",
                "num_pairs",
                "t_statistic",
                "p_value_one_sided",
                "reference_mean",
                "other_mean",
            ]
        )
        for item in comparisons:
            writer.writerow(
                [
                    item["reference_controller"],
                    item["other_controller"],
                    item["metric_key"],
                    int(item["num_pairs"]),
                    f"{float(item['t_statistic']):.8f}",
                    f"{float(item['p_value_one_sided']):.8f}",
                    f"{float(item['reference_mean']):.8f}",
                    f"{float(item['other_mean']):.8f}",
                ]
            )


def load_run_metrics_csv(csv_path: Path) -> List[RunResult]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 run_metrics.csv: {csv_path}")

    results: List[RunResult] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            results.append(
                RunResult(
                    controller=row["controller"],
                    seed=int(row["seed"]),
                    ate_total=float(row["ate_total"]),
                    ate_vx=float(row["ate_vx"]),
                    ate_vy=float(row["ate_vy"]),
                    ate_yaw=float(row["ate_yaw"]),
                    fell=bool(int(row["fell"])),
                    simulated_time_s=float(row["simulated_time_s"]),
                    evaluated_samples=int(row["evaluated_samples"]),
                )
            )

    if not results:
        raise RuntimeError(f"{csv_path} 里没有任何结果行")
    return results


def plot_ate_bars(
    output_dir: Path,
    summary: Dict[str, Dict[str, float]],
    comparisons: Sequence[Dict[str, float | str]],
    reference_controller: str,
    controller_order: Sequence[str] | None = None,
) -> None:
    try:
        mpl_config_dir = (output_dir / ".mplconfig").resolve()
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib 不可用，跳过条形图绘制: {exc}")
        return

    controller_names = list(controller_order) if controller_order is not None else list(summary.keys())
    means = np.array([summary[name]["mean_ate_total"] for name in controller_names], dtype=np.float64)
    stds = np.array([summary[name]["std_ate_total"] for name in controller_names], dtype=np.float64)
    colors = [get_controller_color(name, idx) for idx, name in enumerate(controller_names)]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(controller_names))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
    )

    for bar, mean_value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(stds.max(), 0.01) * 0.35 + 0.005,
            f"{mean_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    comparison_map = {item["other_controller"]: item for item in comparisons}
    y_top = float((means + stds).max()) if len(means) else 0.1
    line_height = y_top + 0.03
    line_step = 0.05

    if reference_controller in controller_names:
        ref_idx = controller_names.index(reference_controller)
        comparison_rank = 0
        for controller_name in controller_names:
            if controller_name == reference_controller or controller_name not in comparison_map:
                continue
            other_idx = controller_names.index(controller_name)
            current_y = line_height + comparison_rank * line_step
            item = comparison_map[controller_name]
            p_value = float(item["p_value_one_sided"])
            if p_value < 0.001:
                marker = "***"
            elif p_value < 0.01:
                marker = "**"
            elif p_value < 0.05:
                marker = "*"
            else:
                marker = "n.s."
            ax.plot(
                [ref_idx, ref_idx, other_idx, other_idx],
                [current_y, current_y + 0.01, current_y + 0.01, current_y],
                color="black",
                linewidth=1.0,
            )
            ax.text(
                (ref_idx + other_idx) / 2.0,
                current_y + 0.012,
                f"{marker}  p={p_value:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            comparison_rank += 1

    ax.set_title("Absolute Tracking Error Across Controllers")
    ax.set_ylabel("ATE")
    ax.set_xticks(x)
    ax.set_xticklabels([get_controller_display_name(name) for name in controller_names], rotation=12, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "ate_barplot.png", dpi=300)
    plt.close(fig)


def significance_marker(p_value: float) -> str:
    if np.isnan(p_value):
        return "n/a"
    if p_value < 1.0e-4:
        return "****"
    if p_value < 1.0e-3:
        return "***"
    if p_value < 1.0e-2:
        return "**"
    if p_value < 5.0e-2:
        return "*"
    return "ns"



def has_instant_errors(results: Sequence[RunResult]) -> bool:
    return any(result.instant_errors is not None and len(result.instant_errors) > 0 for result in results)


def attach_instant_errors_from_trace_dir(
    results: Sequence[RunResult],
    trace_dir: Path,
) -> None:
    """从已有 trace_控制器_seedX.csv 中读取逐时刻误差，挂回 RunResult。

    这个函数主要用于：已经跑完实验并保存了 traces，之后只想重绘论文式 boxplot。
    """
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"找不到 traces 目录: {trace_dir}")

    missing: List[Path] = []
    for result in results:
        trace_path = trace_dir / f"trace_{result.controller}_seed{result.seed}.csv"
        if not trace_path.is_file():
            missing.append(trace_path)
            continue

        errors: List[List[float]] = []
        with trace_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                errors.append(
                    [
                        float(row["abs_err_vx"]),
                        float(row["abs_err_vy"]),
                        float(row["abs_err_yaw"]),
                    ]
                )
        if errors:
            result.instant_errors = np.asarray(errors, dtype=np.float64)

    if missing:
        print("[WARN] 以下 trace 文件不存在，对应运行会退回 seed 平均误差画 boxplot：")
        for path in missing[:10]:
            print(f"       {path}")
        if len(missing) > 10:
            print(f"       ... 还有 {len(missing) - 10} 个")


def collect_axis_boxplot_data(
    results: Sequence[RunResult],
    controller_names: Sequence[str],
) -> tuple[Dict[str, List[np.ndarray]], str]:
    """准备 boxplot 数据。

    优先使用逐时刻误差：
        5 个 seed 内所有时刻 err_vx / err_vy / err_yaw 合并后画 boxplot。
    如果没有逐时刻误差，则退回旧逻辑：
        5 个 seed 的 ate_vx / ate_vy / ate_yaw 画 boxplot。
    """
    grouped_results: Dict[str, List[RunResult]] = {}
    for result in results:
        grouped_results.setdefault(result.controller, []).append(result)

    use_instant = has_instant_errors(results)
    sample_type = "instant_time_step_errors" if use_instant else "per_seed_mean_ate"
    axis_data: Dict[str, List[np.ndarray]] = {}

    for controller_name in controller_names:
        runs = grouped_results.get(controller_name, [])
        metric_arrays: List[np.ndarray] = []

        for axis_idx, (metric_key, _) in enumerate(AXIS_METRICS):
            if use_instant:
                chunks = [
                    run.instant_errors[:, axis_idx].astype(np.float64)
                    for run in runs
                    if run.instant_errors is not None and run.instant_errors.size > 0
                ]
                if chunks:
                    values = np.concatenate(chunks, axis=0)
                else:
                    values = np.array([float(getattr(run, metric_key)) for run in runs], dtype=np.float64)
            else:
                values = np.array([float(getattr(run, metric_key)) for run in runs], dtype=np.float64)

            values = values[np.isfinite(values)]
            metric_arrays.append(values)

        axis_data[controller_name] = metric_arrays

    return axis_data, sample_type


def save_instant_boxplot_stats(
    output_dir: Path,
    axis_data: Dict[str, List[np.ndarray]],
    sample_type: str,
) -> None:
    """保存 tracking_error_boxplot.png 实际使用的数据统计量。

    这个文件可直接用于论文表格或检查每个箱子的 min/Q1/median/Q3/max。
    """
    path = output_dir / "instant_boxplot_stats.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_type",
                "controller",
                "axis",
                "num_samples",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "mean",
            ]
        )
        for controller_name, metric_arrays in axis_data.items():
            for values, (metric_key, _) in zip(metric_arrays, AXIS_METRICS):
                if len(values) == 0:
                    writer.writerow([sample_type, controller_name, metric_key, 0, "", "", "", "", "", ""])
                    continue
                writer.writerow(
                    [
                        sample_type,
                        controller_name,
                        metric_key,
                        int(len(values)),
                        f"{float(np.min(values)):.8f}",
                        f"{float(np.percentile(values, 25)):.8f}",
                        f"{float(np.percentile(values, 50)):.8f}",
                        f"{float(np.percentile(values, 75)):.8f}",
                        f"{float(np.max(values)):.8f}",
                        f"{float(np.mean(values)):.8f}",
                    ]
                )

def plot_axis_error_boxplot(
    output_dir: Path,
    results: Sequence[RunResult],
    axis_comparisons: Sequence[Dict[str, float | str]],
    reference_controller: str,
    controller_order: Sequence[str] | None = None,
) -> None:
    try:
        mpl_config_dir = (output_dir / ".mplconfig").resolve()
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:
        print(f"[WARN] matplotlib 不可用，跳过 boxplot 绘制: {exc}")
        return

    controller_names = list(controller_order) if controller_order is not None else get_controller_order(results)
    if not controller_names:
        print("[WARN] 没有可用于 boxplot 的控制器结果")
        return

    axis_data, sample_type = collect_axis_boxplot_data(results, controller_names)
    save_instant_boxplot_stats(output_dir, axis_data, sample_type)

    if sample_type == "instant_time_step_errors":
        print("[INFO] tracking_error_boxplot.png 使用论文式逐时刻误差：5 个 seed 内所有时刻 err 合并后画 boxplot")
    else:
        print("[WARN] 没有找到逐时刻误差，只能退回旧逻辑：5 个 seed 的平均 ATE 画 boxplot")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    num_groups = len(AXIS_METRICS)
    num_controllers = len(controller_names)
    group_centers = np.arange(num_groups, dtype=np.float64)
    group_width = min(0.78, 0.14 * num_controllers + 0.16)
    box_width = group_width / max(num_controllers, 1) * 0.92
    offsets = np.linspace(
        -group_width / 2.0 + box_width / 2.0,
        group_width / 2.0 - box_width / 2.0,
        num_controllers,
    )

    legend_handles: List[Patch] = []
    legend_labels: List[str] = []
    global_max = 0.0

    for controller_idx, controller_name in enumerate(controller_names):
        controller_color = get_controller_color(controller_name, controller_idx)
        metric_arrays = axis_data.get(controller_name, [])
        if not metric_arrays or all(len(values) == 0 for values in metric_arrays):
            continue

        for values in metric_arrays:
            if len(values) > 0:
                global_max = max(global_max, float(np.max(values)))

        positions = group_centers + offsets[controller_idx]
        boxplot = ax.boxplot(
            metric_arrays,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            manage_ticks=False,
        )

        for box in boxplot["boxes"]:
            box.set(facecolor=controller_color, edgecolor="#404040", linewidth=1.8, alpha=0.95)
        for whisker in boxplot["whiskers"]:
            whisker.set(color="#444444", linewidth=1.5)
        for cap in boxplot["caps"]:
            cap.set(color="#444444", linewidth=1.5)
        for median in boxplot["medians"]:
            median.set(color="#4b4b4b", linewidth=1.7)

        legend_handles.append(Patch(facecolor=controller_color, edgecolor="#404040"))
        legend_labels.append(get_controller_display_name(controller_name))

    comparison_lookup = {
        (str(item["metric_key"]), str(item["other_controller"])): item for item in axis_comparisons
    }

    y_padding = max(0.01, global_max * 0.08)
    line_step = max(0.012, global_max * 0.075)

    if reference_controller in controller_names:
        ref_idx = controller_names.index(reference_controller)
        ref_pos = offsets[ref_idx]
        for group_idx, (metric_key, _) in enumerate(AXIS_METRICS):
            other_names = [name for name in controller_names if name != reference_controller]
            other_names = sorted(other_names, key=lambda name: controller_names.index(name))

            metric_max = 0.0
            for controller_name in controller_names:
                metric_arrays = axis_data.get(controller_name, [])
                if group_idx >= len(metric_arrays):
                    continue
                metric_values = metric_arrays[group_idx]
                if len(metric_values) > 0:
                    metric_max = max(metric_max, float(np.max(metric_values)))

            current_y = metric_max + y_padding
            for other_name in other_names:
                item = comparison_lookup.get((metric_key, other_name))
                if item is None:
                    continue
                other_idx = controller_names.index(other_name)
                x1 = group_centers[group_idx] + ref_pos
                x2 = group_centers[group_idx] + offsets[other_idx]
                left = min(x1, x2)
                right = max(x1, x2)
                cap_h = line_step * 0.28
                ax.plot(
                    [left, left, right, right],
                    [current_y, current_y + cap_h, current_y + cap_h, current_y],
                    color="#404040",
                    linewidth=1.6,
                )
                ax.text(
                    (left + right) / 2.0,
                    current_y + cap_h + line_step * 0.02,
                    significance_marker(float(item["p_value_one_sided"])),
                    ha="center",
                    va="bottom",
                    fontsize=16,
                )
                current_y += line_step

    ax.set_ylabel("Tracking error", fontsize=20)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([axis_label for _, axis_label in AXIS_METRICS], fontsize=20)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", linestyle="-", color="#b5b5b5", linewidth=1.0)

    y_upper = ax.get_ylim()[1]
    ax.set_ylim(
        0.0,
        max(y_upper, global_max + y_padding + line_step * max(len(controller_names), 1) + 0.01),
    )

    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(2, len(legend_labels)),
        fontsize=14,
        frameon=True,
        framealpha=0.95,
    )
    legend.get_frame().set_edgecolor("#c8c8c8")

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_dir / "tracking_error_boxplot.png", dpi=300)
    plt.close(fig)

def build_output_dir(custom_output_dir: Path | None) -> Path:
    if custom_output_dir is not None:
        output_dir = custom_output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (DEFAULT_OUTPUT_ROOT / f"command_tracking_ate_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_summary_json(
    output_dir: Path,
    args: argparse.Namespace,
    controllers: Sequence[ControllerConfig],
    seeds: Sequence[int],
    reference_controller: str,
    summary: Dict[str, Dict[str, float]],
    comparisons: Sequence[Dict[str, float | str]],
) -> None:
    payload = {
        "xml_path": str(args.xml_path.resolve()),
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "command_interval_s": args.command_interval,
        "seeds": list(seeds),
        "reference_controller": reference_controller,
        "controllers": [
            {"name": controller.name, "policy_path": str(controller.policy_path.resolve())}
            for controller in controllers
        ],
        "summary": summary,
        "pairwise_t_tests": comparisons,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def validate_controller_files(controllers: Sequence[ControllerConfig]) -> None:
    for controller in controllers:
        if not controller.policy_path.is_file():
            raise FileNotFoundError(f"找不到策略文件: {controller.policy_path}")


def render_plots_and_stats(
    output_dir: Path,
    results: Sequence[RunResult],
    reference_controller: str,
    controller_order: Sequence[str] | None = None,
) -> tuple[Dict[str, Dict[str, float]], List[Dict[str, float | str]], List[Dict[str, float | str]]]:
    summary = summarize_results(results)
    total_comparisons = paired_t_tests(results, reference_controller)
    axis_comparisons = axis_paired_t_tests(results, reference_controller)

    save_run_metrics(output_dir, results)
    save_pairwise_tests(output_dir, total_comparisons)
    save_axis_pairwise_tests(output_dir, axis_comparisons)
    plot_ate_bars(output_dir, summary, total_comparisons, reference_controller, controller_order)
    plot_axis_error_boxplot(output_dir, results, axis_comparisons, reference_controller, controller_order)
    return summary, total_comparisons, axis_comparisons


def format_run_tracking_error(result: RunResult) -> str:
    return (
        f"tracking_error={result.ate_total:.4f} "
        f"(vx={result.ate_vx:.4f}, vy={result.ate_vy:.4f}, yaw={result.ate_yaw:.4f})"
    )


def format_summary_tracking_error(controller_name: str, item: Dict[str, float]) -> str:
    return (
        f"{controller_name}: "
        f"mean_tracking_error={item['mean_ate_total']:.4f} "
        f"+/- {item['std_ate_total']:.4f} "
        f"(vx={item['mean_ate_vx']:.4f}, vy={item['mean_ate_vy']:.4f}, yaw={item['mean_ate_yaw']:.4f}), "
        f"falls={int(item['num_falls'])}"
    )


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()

    if args.plot_only_run_metrics is not None:
        run_metrics_path = args.plot_only_run_metrics.resolve()
        output_dir = build_output_dir(args.output_dir) if args.output_dir is not None else run_metrics_path.parent
        results = filter_results_by_controller_names(load_run_metrics_csv(run_metrics_path), args.controllers)
        if args.plot_only_trace_dir is not None:
            attach_instant_errors_from_trace_dir(results, args.plot_only_trace_dir.resolve())
        controller_order = get_controller_order(results)
        reference_controller = (
            REFERENCE_CONTROLLER if REFERENCE_CONTROLLER in set(controller_order) else controller_order[0]
        )
        summary, comparisons, axis_comparisons = render_plots_and_stats(
            output_dir=output_dir,
            results=results,
            reference_controller=reference_controller,
            controller_order=controller_order,
        )
        print("[INFO] 已从现有 run_metrics.csv 重绘图表")
        print(f"[INFO] run_metrics.csv: {run_metrics_path}")
        print(f"[INFO] 输出目录: {output_dir}")
        print(f"[INFO] 参考控制器: {reference_controller}")
        print(f"[INFO] 总体 t 检验数量: {len(comparisons)}")
        print(f"[INFO] 轴向 t 检验数量: {len(axis_comparisons)}")
        print(f"[INFO] 生成文件: {output_dir / 'tracking_error_boxplot.png'}")
        print(f"[INFO] boxplot 统计量: {output_dir / 'instant_boxplot_stats.csv'}")
        return

    seeds = parse_seed_list(args.seeds)
    controllers = resolve_controllers(args.controllers)
    validate_controller_files(controllers)
    controller_order = [controller.name for controller in controllers]
    reference_controller = (
        REFERENCE_CONTROLLER if REFERENCE_CONTROLLER in set(controller_order) else controller_order[0]
    )

    xml_path = args.xml_path.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"找不到 XML 文件: {xml_path}")

    output_dir = build_output_dir(args.output_dir)
    trace_dir = output_dir / "traces" if args.save_traces else None

    print("[INFO] 加载 MuJoCo 模型...")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos_idx, qvel_idx = get_joint_indices(model)
    base_body_id = find_body_id(model, ["base_link", "trunk", "base", "body"])
    print(f"[INFO] XML: {xml_path}")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] 控制器: {controller_order}")
    print(f"[INFO] seeds: {seeds}")
    print(f"[INFO] 参考控制器: {reference_controller}")
    if args.viewer:
        print("[INFO] viewer 模式已开启")
        if len(controllers) * len(seeds) > 1:
            print("[INFO] 当前是批量运行，viewer 会按 controller/seed 顺序逐次打开。")

    command_sequences = {
        seed: generate_command_sequence(seed, args.duration, args.command_interval) for seed in seeds
    }
    save_command_sequences(output_dir, command_sequences, args.command_interval, args.duration)

    loaded_policies: Dict[str, torch.jit.ScriptModule] = {}
    for controller in controllers:
        print(f"[INFO] 加载策略: {controller.name} -> {controller.policy_path}")
        policy = torch.jit.load(str(controller.policy_path), map_location="cpu")
        policy.eval()
        loaded_policies[controller.name] = policy

    all_results: List[RunResult] = []

    for controller in controllers:
        policy = loaded_policies[controller.name]
        for seed in seeds:
            print(f"[RUN] controller={controller.name} seed={seed}")
            result = run_single_controller_seed(
                controller=controller,
                model=model,
                qpos_idx=qpos_idx,
                qvel_idx=qvel_idx,
                base_body_id=base_body_id,
                policy=policy,
                commands=command_sequences[seed],
                seed=seed,
                duration_s=args.duration,
                warmup_s=args.warmup,
                command_interval_s=args.command_interval,
                fall_penalty=args.fall_penalty,
                enable_viewer=args.viewer,
                save_traces=args.save_traces,
                trace_dir=trace_dir,
            )
            all_results.append(result)
            status = "FELL" if result.fell else "OK"
            print(
                f"[DONE] {controller.name} seed={seed} "
                f"{format_run_tracking_error(result)} "
                f"status={status}"
            )

    summary, comparisons, axis_comparisons = render_plots_and_stats(
        output_dir=output_dir,
        results=all_results,
        reference_controller=reference_controller,
        controller_order=controller_order,
    )
    save_summary_json(
        output_dir,
        args,
        controllers,
        seeds,
        reference_controller,
        summary,
        comparisons,
    )

    print("\n================ 评估完成 ================")
    for controller_name, item in summary.items():
        print(format_summary_tracking_error(controller_name, item))
    if comparisons:
        print("------------------------------------------")
        for item in comparisons:
            print(
                f"{item['reference_controller']} vs {item['other_controller']}: "
                f"t={float(item['t_statistic']):.4f}, p(one-sided)={float(item['p_value_one_sided']):.6f}"
            )
    if axis_comparisons:
        print("------------------------------------------")
        print(f"轴向 boxplot 已保存: {output_dir / 'tracking_error_boxplot.png'}")
    print(f"结果目录: {output_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()