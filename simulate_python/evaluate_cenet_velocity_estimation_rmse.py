#!/usr/bin/env python3

"""Evaluate DreamWaQ/CENet body linear velocity estimation RMSE in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import torch.nn as nn


SIM_DIR = Path("/home/hyc/unitree_mujoco/simulate_python").resolve()
LEGACY_EVAL_PATH = (SIM_DIR / "evaluate_command_tracking_ate.py").resolve()
DEFAULT_XML_PATH = (SIM_DIR / "../unitree_robots/go2/scene_flat.xml").resolve()
DEFAULT_OUTPUT_ROOT = (SIM_DIR / "eval_outputs").resolve()

RL_LAB_LOG_ROOT = Path("/home/hyc/unitree_rl_lab/scripts/rsl_rl/logs/rsl_rl").resolve()
DEFAULT_DREAMWAQ_RUN_DIR = (
    RL_LAB_LOG_ROOT / "unitree_go2_dreamwaq_cenet/2026-07-26_18-34-42"
).resolve()
DEFAULT_POLICY_PATH = (DEFAULT_DREAMWAQ_RUN_DIR / "exported/policy.pt").resolve()
DEFAULT_CHECKPOINT_PATH = (DEFAULT_DREAMWAQ_RUN_DIR / "model_4999.pt").resolve()

CURRENT_OBS_DIM = 45
POLICY_OBS_DIM = 270
HISTORY_LEN = 5
VELOCITY_DIM = 3
NUM_ACTIONS = 12

COMMAND_LOW = np.array([-1.0, -0.4, -1.0], dtype=np.float32)
COMMAND_HIGH = np.array([1.0, 0.4, 1.0], dtype=np.float32)


@dataclass
class RunMetrics:
    seed: int
    rmse_3d: float
    rmse_xy: float
    rmse_vx: float
    rmse_vy: float
    rmse_vz: float
    mae_vx: float
    mae_vy: float
    mae_vz: float
    num_samples: int
    fell: bool
    simulated_time_s: float
    errors: np.ndarray


class CENetVelocityEstimator(nn.Module):
    def __init__(self, encoder: nn.Sequential, velocity_head: nn.Linear, current_obs_dim: int):
        super().__init__()
        self.encoder = encoder
        self.velocity_head = velocity_head
        self.current_obs_dim = current_obs_dim

    def forward(self, policy_obs: torch.Tensor) -> torch.Tensor:
        history_obs = policy_obs[:, self.current_obs_dim :]
        encoded_history = self.encoder(history_obs)
        return self.velocity_head(encoded_history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DreamWaQ/CENet velocity estimation RMSE evaluation")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH, help="MuJoCo XML path")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH, help="Exported policy.pt path")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Training checkpoint path containing actor.encoder and actor.velocity_head",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Effective evaluation duration in seconds")
    parser.add_argument("--warmup", type=float, default=2.0, help="PD-only warmup duration before scoring")
    parser.add_argument("--command-interval", type=float, default=10.0, help="Random command resample interval")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seed list")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--save-traces", action="store_true", help="Save per-timestep velocity traces")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        choices=("elu", "relu", "tanh"),
        help="CENet encoder activation used during training",
    )
    return parser.parse_args()


def load_module(path: Path, module_name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def load_cenet_velocity_estimator(
    checkpoint_path: Path,
    activation_name: str,
    current_obs_dim: int,
    device: torch.device,
) -> CENetVelocityEstimator:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    linear_indices = sorted(
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("actor.encoder.") and key.endswith(".weight")
    )
    if not linear_indices:
        raise RuntimeError("No actor.encoder linear weights found in checkpoint.")

    layers: list[nn.Module] = []
    for index in linear_indices:
        weight = state_dict[f"actor.encoder.{index}.weight"]
        bias = state_dict[f"actor.encoder.{index}.bias"]
        out_features, in_features = weight.shape
        linear = nn.Linear(in_features, out_features)
        if linear.bias is None or bias is None:
            raise RuntimeError(f"Unexpected missing bias for actor.encoder.{index}")
        layers.append(linear)
        layers.append(make_activation(activation_name))

    encoder = nn.Sequential(*layers)
    encoder_state = {
        key.removeprefix("actor.encoder."): value
        for key, value in state_dict.items()
        if key.startswith("actor.encoder.")
    }
    encoder.load_state_dict(encoder_state)

    head_weight = state_dict["actor.velocity_head.weight"]
    head_bias = state_dict["actor.velocity_head.bias"]
    velocity_head = nn.Linear(head_weight.shape[1], head_weight.shape[0])
    velocity_head.load_state_dict({"weight": head_weight, "bias": head_bias})

    estimator = CENetVelocityEstimator(encoder, velocity_head, current_obs_dim=current_obs_dim)
    estimator.to(device)
    estimator.eval()
    return estimator


def parse_seed_list(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def build_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        path = output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (DEFAULT_OUTPUT_ROOT / f"cenet_velocity_rmse_{timestamp}").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_command_sequence(seed: int, duration_s: float, interval_s: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_segments = int(math.ceil(duration_s / interval_s))
    return rng.uniform(COMMAND_LOW, COMMAND_HIGH, size=(num_segments, 3)).astype(np.float32)


def command_for_time(commands: np.ndarray, eval_time_s: float, interval_s: float) -> np.ndarray:
    segment_idx = min(int(eval_time_s // interval_s), len(commands) - 1)
    return commands[segment_idx]


def save_command_sequences(
    output_dir: Path,
    command_sequences: dict[int, np.ndarray],
    interval_s: float,
    duration_s: float,
) -> None:
    path = output_dir / "command_sequences.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "segment_id", "start_time_s", "end_time_s", "cmd_vx", "cmd_vy", "cmd_yaw"])
        for seed, commands in command_sequences.items():
            for index, command in enumerate(commands):
                start_time = index * interval_s
                end_time = min((index + 1) * interval_s, duration_s)
                writer.writerow(
                    [
                        seed,
                        index,
                        f"{start_time:.6f}",
                        f"{end_time:.6f}",
                        f"{command[0]:.6f}",
                        f"{command[1]:.6f}",
                        f"{command[2]:.6f}",
                    ]
                )


def base_linear_velocity_body(data: mujoco.MjData, base_body_id: int) -> np.ndarray:
    base_mat = data.xmat[base_body_id].reshape(3, 3)
    base_lin_vel_world = data.qvel[0:3].copy()
    return (base_mat.T @ base_lin_vel_world).astype(np.float32)


def compute_metrics(errors: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    rmse_axis = np.sqrt(np.mean(np.square(errors), axis=0))
    mae_axis = np.mean(np.abs(errors), axis=0)
    rmse_3d = float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1))))
    rmse_xy = float(np.sqrt(np.mean(np.sum(np.square(errors[:, :2]), axis=1))))
    return rmse_3d, rmse_xy, rmse_axis, mae_axis


def run_single_seed(
    *,
    legacy: Any,
    model: mujoco.MjModel,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    base_body_id: int,
    policy: torch.jit.ScriptModule,
    estimator: CENetVelocityEstimator,
    commands: np.ndarray,
    seed: int,
    duration_s: float,
    warmup_s: float,
    command_interval_s: float,
    device: torch.device,
    enable_viewer: bool,
    save_traces: bool,
    trace_dir: Path | None,
) -> RunMetrics:
    data = legacy.reset_sim_state(model, qpos_idx)
    control_dt = model.opt.timestep * legacy.DECIMATION
    total_time_s = warmup_s + duration_s
    total_steps = int(math.ceil(total_time_s / model.opt.timestep))

    actions_leg = np.zeros(NUM_ACTIONS, dtype=np.float32)
    history_buffer: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)

    zero_command = np.zeros(3, dtype=np.float32)
    first_current_obs = legacy.build_current_obs(
        data=data,
        base_body_id=base_body_id,
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        last_action_leg=actions_leg,
        command=zero_command,
    )
    for _ in range(HISTORY_LEN):
        history_buffer.append(first_current_obs.copy())

    times: list[float] = []
    cmd_log: list[np.ndarray] = []
    gt_log: list[np.ndarray] = []
    est_log: list[np.ndarray] = []
    err_log: list[np.ndarray] = []

    fell = False
    simulated_time_s = 0.0
    viewer_context = None
    viewer = None

    if enable_viewer:
        try:
            from mujoco import viewer as mujoco_viewer

            viewer_context = mujoco_viewer.launch_passive(model, data)
            viewer = viewer_context.__enter__()
        except Exception as exc:
            print(f"[WARN] viewer failed, fallback to headless mode: {exc}")

    try:
        block_wall_start = None
        for step_counter in range(total_steps):
            if step_counter % legacy.DECIMATION == 0:
                block_wall_start = time.time()

            sim_time = step_counter * model.opt.timestep
            qpos_leg = data.qpos[qpos_idx].copy()
            qvel_leg = data.qvel[qvel_idx].copy()

            if viewer is not None and not viewer.is_running():
                simulated_time_s = sim_time
                break

            if step_counter % legacy.DECIMATION == 0:
                if data.xpos[base_body_id][2] < legacy.FALL_BASE_HEIGHT_THRESHOLD and sim_time >= warmup_s:
                    fell = True
                    simulated_time_s = sim_time
                    break

                if sim_time < warmup_s:
                    command = zero_command
                else:
                    eval_time_s = sim_time - warmup_s
                    command = command_for_time(commands, eval_time_s, command_interval_s)

                current_obs = legacy.build_current_obs(
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
                    eval_time_s = sim_time - warmup_s
                    policy_obs_np = legacy.build_policy_obs(current_obs, history_buffer)
                    if policy_obs_np.shape[0] != POLICY_OBS_DIM:
                        raise RuntimeError(f"policy_obs dim mismatch: {policy_obs_np.shape[0]}")

                    obs_tensor = torch.from_numpy(policy_obs_np).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        velocity_estimate = estimator(obs_tensor).cpu().numpy().squeeze().astype(np.float32)
                        raw_actions_net = policy(obs_tensor.cpu())
                        if isinstance(raw_actions_net, (tuple, list)):
                            raw_actions_net = raw_actions_net[0]
                        raw_actions_net = raw_actions_net.cpu().numpy().squeeze().astype(np.float32)

                    if raw_actions_net.shape[0] != NUM_ACTIONS:
                        raise RuntimeError(f"Policy action dim mismatch: {raw_actions_net.shape}")
                    if velocity_estimate.shape[0] != VELOCITY_DIM:
                        raise RuntimeError(f"Velocity estimate dim mismatch: {velocity_estimate.shape}")

                    gt_velocity = base_linear_velocity_body(data, base_body_id)
                    velocity_error = velocity_estimate - gt_velocity

                    times.append(eval_time_s)
                    cmd_log.append(command.copy())
                    gt_log.append(gt_velocity.copy())
                    est_log.append(velocity_estimate.copy())
                    err_log.append(velocity_error.copy())

                    raw_actions_leg = raw_actions_net[legacy.ISAACLAB2LEG_ORDER]
                    clipped_actions_leg = np.clip(raw_actions_leg, -legacy.ACTION_CLIP, legacy.ACTION_CLIP)
                    if sim_time < warmup_s + legacy.POLICY_RAMP_TIME:
                        alpha = float(np.clip((sim_time - warmup_s) / legacy.POLICY_RAMP_TIME, 0.0, 1.0))
                        actions_leg = clipped_actions_leg * alpha
                    else:
                        actions_leg = clipped_actions_leg

            target_dof_pos_leg = legacy.DEFAULT_DOF_POS_LEG + legacy.ACTION_SCALE * actions_leg
            tau_leg = legacy.KP * (target_dof_pos_leg - qpos_leg) - legacy.KD * qvel_leg
            tau_leg = np.clip(tau_leg, -legacy.TORQUE_LIMIT_LEG, legacy.TORQUE_LIMIT_LEG)
            tau_ctrl = tau_leg[legacy.CTRL_MUJOCO_IDX]
            if data.ctrl.shape[0] == NUM_ACTIONS:
                data.ctrl[:] = tau_ctrl
            else:
                data.ctrl[:NUM_ACTIONS] = tau_ctrl

            mujoco.mj_step(model, data)
            simulated_time_s = sim_time + model.opt.timestep

            if viewer is not None and (step_counter + 1) % legacy.DECIMATION == 0:
                viewer.sync()
                if block_wall_start is not None:
                    elapsed = time.time() - block_wall_start
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)

    if not err_log:
        raise RuntimeError(f"No evaluation samples collected for seed={seed}")

    errors = np.stack(err_log, axis=0)
    rmse_3d, rmse_xy, rmse_axis, mae_axis = compute_metrics(errors)

    if save_traces and trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"velocity_estimation_seed{seed}.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "eval_time_s",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_yaw",
                    "gt_vx",
                    "gt_vy",
                    "gt_vz",
                    "est_vx",
                    "est_vy",
                    "est_vz",
                    "err_vx",
                    "err_vy",
                    "err_vz",
                    "err_l2",
                ]
            )
            for eval_time, command, gt_vel, est_vel, err in zip(times, cmd_log, gt_log, est_log, err_log):
                writer.writerow(
                    [
                        f"{eval_time:.6f}",
                        f"{command[0]:.8f}",
                        f"{command[1]:.8f}",
                        f"{command[2]:.8f}",
                        f"{gt_vel[0]:.8f}",
                        f"{gt_vel[1]:.8f}",
                        f"{gt_vel[2]:.8f}",
                        f"{est_vel[0]:.8f}",
                        f"{est_vel[1]:.8f}",
                        f"{est_vel[2]:.8f}",
                        f"{err[0]:.8f}",
                        f"{err[1]:.8f}",
                        f"{err[2]:.8f}",
                        f"{float(np.linalg.norm(err)):.8f}",
                    ]
                )

    return RunMetrics(
        seed=seed,
        rmse_3d=rmse_3d,
        rmse_xy=rmse_xy,
        rmse_vx=float(rmse_axis[0]),
        rmse_vy=float(rmse_axis[1]),
        rmse_vz=float(rmse_axis[2]),
        mae_vx=float(mae_axis[0]),
        mae_vy=float(mae_axis[1]),
        mae_vz=float(mae_axis[2]),
        num_samples=len(err_log),
        fell=fell,
        simulated_time_s=simulated_time_s,
        errors=errors,
    )


def save_run_metrics(output_dir: Path, metrics: list[RunMetrics]) -> None:
    path = output_dir / "velocity_estimation_run_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "seed",
                "rmse_3d",
                "rmse_xy",
                "rmse_vx",
                "rmse_vy",
                "rmse_vz",
                "mae_vx",
                "mae_vy",
                "mae_vz",
                "num_samples",
                "fell",
                "simulated_time_s",
            ]
        )
        for item in metrics:
            writer.writerow(
                [
                    item.seed,
                    f"{item.rmse_3d:.8f}",
                    f"{item.rmse_xy:.8f}",
                    f"{item.rmse_vx:.8f}",
                    f"{item.rmse_vy:.8f}",
                    f"{item.rmse_vz:.8f}",
                    f"{item.mae_vx:.8f}",
                    f"{item.mae_vy:.8f}",
                    f"{item.mae_vz:.8f}",
                    item.num_samples,
                    int(item.fell),
                    f"{item.simulated_time_s:.6f}",
                ]
            )


def build_summary(metrics: list[RunMetrics], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    all_errors = np.concatenate([item.errors for item in metrics], axis=0)
    overall_rmse_3d, overall_rmse_xy, overall_rmse_axis, overall_mae_axis = compute_metrics(all_errors)

    def mean_std(name: str) -> dict[str, float]:
        values = np.array([getattr(item, name) for item in metrics], dtype=np.float64)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    return {
        "xml_path": str(args.xml_path.resolve()),
        "policy_path": str(args.policy_path.resolve()),
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "output_dir": str(output_dir),
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "command_interval_s": args.command_interval,
        "command_low": COMMAND_LOW.tolist(),
        "command_high": COMMAND_HIGH.tolist(),
        "seeds": [item.seed for item in metrics],
        "num_runs": len(metrics),
        "num_falls": int(sum(item.fell for item in metrics)),
        "total_samples": int(all_errors.shape[0]),
        "overall": {
            "rmse_3d": overall_rmse_3d,
            "rmse_xy": overall_rmse_xy,
            "rmse_vx": float(overall_rmse_axis[0]),
            "rmse_vy": float(overall_rmse_axis[1]),
            "rmse_vz": float(overall_rmse_axis[2]),
            "mae_vx": float(overall_mae_axis[0]),
            "mae_vy": float(overall_mae_axis[1]),
            "mae_vz": float(overall_mae_axis[2]),
        },
        "run_mean_std": {
            "rmse_3d": mean_std("rmse_3d"),
            "rmse_xy": mean_std("rmse_xy"),
            "rmse_vx": mean_std("rmse_vx"),
            "rmse_vy": mean_std("rmse_vy"),
            "rmse_vz": mean_std("rmse_vz"),
        },
    }


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()

    xml_path = args.xml_path.resolve()
    policy_path = args.policy_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    for path in (xml_path, policy_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    output_dir = build_output_dir(args.output_dir)
    trace_dir = output_dir / "traces" if args.save_traces else None
    seeds = parse_seed_list(args.seeds)
    device = torch.device("cpu")

    legacy = load_module(LEGACY_EVAL_PATH, "legacy_evaluate_command_tracking_ate_for_velocity_rmse")
    legacy.HISTORY_LEN = HISTORY_LEN
    legacy.CURRENT_OBS_DIM = CURRENT_OBS_DIM
    legacy.POLICY_OBS_DIM = POLICY_OBS_DIM

    print("[INFO] Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos_idx, qvel_idx = legacy.get_joint_indices(model)
    base_body_id = legacy.find_body_id(model, ["base_link", "trunk", "base", "body"])

    print("[INFO] Loading DreamWaQ policy and CENet velocity estimator...")
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    estimator = load_cenet_velocity_estimator(
        checkpoint_path=checkpoint_path,
        activation_name=args.activation,
        current_obs_dim=CURRENT_OBS_DIM,
        device=device,
    )

    command_sequences = {
        seed: generate_command_sequence(seed, args.duration, args.command_interval) for seed in seeds
    }
    save_command_sequences(output_dir, command_sequences, args.command_interval, args.duration)

    print(f"[INFO] XML: {xml_path}")
    print(f"[INFO] policy: {policy_path}")
    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] seeds: {seeds}")
    print(f"[INFO] command_low: {COMMAND_LOW.tolist()}")
    print(f"[INFO] command_high: {COMMAND_HIGH.tolist()}")

    all_metrics: list[RunMetrics] = []
    for seed in seeds:
        print(f"[RUN] DreamWaQ_CENet velocity estimation seed={seed}")
        metrics = run_single_seed(
            legacy=legacy,
            model=model,
            qpos_idx=qpos_idx,
            qvel_idx=qvel_idx,
            base_body_id=base_body_id,
            policy=policy,
            estimator=estimator,
            commands=command_sequences[seed],
            seed=seed,
            duration_s=args.duration,
            warmup_s=args.warmup,
            command_interval_s=args.command_interval,
            device=device,
            enable_viewer=args.viewer,
            save_traces=args.save_traces,
            trace_dir=trace_dir,
        )
        all_metrics.append(metrics)
        status = "FELL" if metrics.fell else "OK"
        print(
            "[DONE] seed={} rmse_3d={:.4f} rmse_xy={:.4f} "
            "(vx={:.4f}, vy={:.4f}, vz={:.4f}) samples={} status={}".format(
                seed,
                metrics.rmse_3d,
                metrics.rmse_xy,
                metrics.rmse_vx,
                metrics.rmse_vy,
                metrics.rmse_vz,
                metrics.num_samples,
                status,
            )
        )

    save_run_metrics(output_dir, all_metrics)
    summary = build_summary(all_metrics, args, output_dir)
    with (output_dir / "velocity_estimation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    overall = summary["overall"]
    run_mean_std = summary["run_mean_std"]
    print("\n================ Velocity Estimation Complete ================")
    print(
        "DreamWaQ_CENet overall: rmse_3d={:.4f} m/s, rmse_xy={:.4f} m/s, "
        "vx={:.4f}, vy={:.4f}, vz={:.4f}, falls={}/{}".format(
            overall["rmse_3d"],
            overall["rmse_xy"],
            overall["rmse_vx"],
            overall["rmse_vy"],
            overall["rmse_vz"],
            summary["num_falls"],
            summary["num_runs"],
        )
    )
    print(
        "Run mean/std rmse_3d={:.4f} +/- {:.4f}, rmse_xy={:.4f} +/- {:.4f}".format(
            run_mean_std["rmse_3d"]["mean"],
            run_mean_std["rmse_3d"]["std"],
            run_mean_std["rmse_xy"]["mean"],
            run_mean_std["rmse_xy"]["std"],
        )
    )
    print(f"[INFO] run_metrics.csv: {output_dir / 'velocity_estimation_run_metrics.csv'}")
    print(f"[INFO] summary.json: {output_dir / 'velocity_estimation_summary.json'}")
    if trace_dir is not None:
        print(f"[INFO] traces: {trace_dir}")


if __name__ == "__main__":
    main()
