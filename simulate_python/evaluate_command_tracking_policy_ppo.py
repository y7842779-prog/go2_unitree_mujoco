#!/usr/bin/env python3

"""Mixed-input ATE evaluation for standard PPO and DreamWaQ-style Go2 policies."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch


LEGACY_SIM_DIR = Path("/home/hyc/unitree_mujoco/simulate_python").resolve()
LEGACY_EVAL_PATH = (LEGACY_SIM_DIR / "evaluate_command_tracking_ate.py").resolve()
DEFAULT_XML_PATH = (LEGACY_SIM_DIR / "../unitree_robots/go2/scene_flat.xml").resolve()
DEFAULT_OUTPUT_ROOT = (LEGACY_SIM_DIR / "eval_outputs").resolve()
RL_LAB_LOG_ROOT = Path("/home/hyc/unitree_rl_lab/scripts/rsl_rl/logs/rsl_rl").resolve()
DEFAULT_PPO_POLICY_PATH = (
    RL_LAB_LOG_ROOT
    / "unitree_go2_ppo_dr_rough_curriculum/2026-07-20_11-50-26/exported/policy.pt"
).resolve()
DEFAULT_DREAMWAQ_CENET_POLICY_PATH = (
    RL_LAB_LOG_ROOT / "unitree_go2_dreamwaq_cenet/2026-07-26_18-34-42/exported/policy.pt"
).resolve()
COMMAND_LOW = np.array([-1.0, -0.4, -1.0], dtype=np.float32)
COMMAND_HIGH = np.array([1.0, 0.4, 1.0], dtype=np.float32)

PPO_HISTORY_LEN = 1
PPO_CURRENT_OBS_DIM = 235
PPO_POLICY_OBS_DIM = 235
PPO45_HISTORY_LEN = 1
PPO45_CURRENT_OBS_DIM = 45
PPO45_POLICY_OBS_DIM = 45
PPO45_BASE_ANG_VEL_SCALE = np.array([0.2, 0.2, 0.2], dtype=np.float32)
PPO45_JOINT_VEL_SCALE = np.full(12, 0.05, dtype=np.float32)
DREAMWAQ_HISTORY_LEN = 5
DREAMWAQ_CURRENT_OBS_DIM = 45
DREAMWAQ_POLICY_OBS_DIM = 270

HEIGHT_SCAN_X = np.arange(-0.8, 0.8001, 0.1, dtype=np.float32)
HEIGHT_SCAN_Y = np.arange(-0.5, 0.5001, 0.1, dtype=np.float32)
HEIGHT_SCAN_CLIP_MIN = -1.0
HEIGHT_SCAN_CLIP_MAX = 1.0
HEIGHT_SCAN_RAY_START_Z = 1.0
HEIGHT_SCAN_OFFSET = 0.5
RAYCAST_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float32)


@dataclass(frozen=True)
class MixedControllerConfig:
    name: str
    policy_path: Path
    input_mode: str
    display_name: str | None = None
    color: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo command tracking ATE evaluation for PPO + DreamWaQ mixed-input policies"
    )
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH, help="MuJoCo XML path")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=DEFAULT_PPO_POLICY_PATH,
        help="TorchScript path for the standard PPO controller",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="PolicyPPO,DreamWaQ_CENet",
        help='Comma-separated controller names; pass "all" to include every configured controller',
    )
    parser.add_argument(
        "--reference-controller",
        type=str,
        default="PolicyPPO",
        help="Reference controller for paired t-tests; falls back to the first selected controller if absent",
    )
    parser.add_argument("--duration", type=float, default=600.0, help="Effective evaluation duration in seconds")
    parser.add_argument("--warmup", type=float, default=2.0, help="Warmup duration before scoring")
    parser.add_argument("--command-interval", type=float, default=10.0, help="Random command resample interval")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seed list")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--save-traces", action="store_true", help="Save per-run traces")
    parser.add_argument("--fall-penalty", type=float, default=2.0, help="Per-axis penalty after falling")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument(
        "--plot-only-run-metrics",
        type=Path,
        default=None,
        help="Redraw plots from an existing run_metrics.csv without rerunning simulation",
    )
    parser.add_argument(
        "--plot-only-trace-dir",
        type=Path,
        default=None,
        help="Trace directory used together with --plot-only-run-metrics",
    )
    return parser.parse_args()


def build_controller_catalog(policy_path: Path) -> list[MixedControllerConfig]:
    return [
        MixedControllerConfig(
            name="PolicyPPO",
            policy_path=policy_path.resolve(),
            input_mode="ppo45",
            display_name="PPO rough curriculum",
            color="#3c963c",
        ),
        MixedControllerConfig(
            name="DreamWaQ_AdaBoot",
            policy_path=(LEGACY_SIM_DIR / "DreamWaQ_models/policy_5N_foce.pt").resolve(),
            input_mode="dreamwaq270",
            display_name="DreamWaQ w/ AdaBoot",
            color="#3776ab",
        ),
        MixedControllerConfig(
            name="DreamWaQ_CENet",
            policy_path=DEFAULT_DREAMWAQ_CENET_POLICY_PATH,
            input_mode="dreamwaq270",
            display_name="DreamWaQ w/ CENet",
            color="#d84b3a",
        ),
    ]


def resolve_controllers(catalog: list[MixedControllerConfig], selection: str) -> list[MixedControllerConfig]:
    if selection.strip().lower() == "all":
        return catalog

    wanted = {name.strip() for name in selection.split(",") if name.strip()}
    selected = [controller for controller in catalog if controller.name in wanted]
    missing = sorted(wanted - {controller.name for controller in selected})
    if missing:
        raise ValueError(f"Unknown controllers: {missing}")
    if not selected:
        raise ValueError("At least one controller must be selected")
    return selected


def to_legacy_controller(legacy: Any, controller: MixedControllerConfig):
    return legacy.ControllerConfig(
        name=controller.name,
        policy_path=controller.policy_path,
        display_name=controller.display_name,
        color=controller.color,
    )


def load_legacy_module() -> Any:
    if not LEGACY_EVAL_PATH.is_file():
        raise FileNotFoundError(f"Legacy ATE evaluator not found: {LEGACY_EVAL_PATH}")

    module_name = "legacy_evaluate_command_tracking_ate"
    spec = importlib.util.spec_from_file_location(module_name, LEGACY_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {LEGACY_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def configure_legacy_metadata(legacy: Any, controllers: list[MixedControllerConfig], reference_controller: str) -> None:
    legacy.CONTROLLERS = [to_legacy_controller(legacy, controller) for controller in controllers]
    legacy.REFERENCE_CONTROLLER = reference_controller
    legacy.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    legacy.COMMAND_LOW = COMMAND_LOW
    legacy.COMMAND_HIGH = COMMAND_HIGH


def build_height_scan(model: mujoco.MjModel, data: mujoco.MjData, base_body_id: int) -> np.ndarray:
    base_pos = data.xpos[base_body_id].copy().astype(np.float32)
    base_mat = data.xmat[base_body_id].reshape(3, 3).copy().astype(np.float32)
    yaw = math.atan2(float(base_mat[1, 0]), float(base_mat[0, 0]))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    yaw_rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    grid_x, grid_y = np.meshgrid(HEIGHT_SCAN_X, HEIGHT_SCAN_Y, indexing="xy")
    scan_values = np.empty(grid_x.size, dtype=np.float32)
    for index, (offset_x, offset_y) in enumerate(zip(grid_x.ravel(), grid_y.ravel(), strict=True)):
        offset_body = np.array([offset_x, offset_y, 0.0], dtype=np.float32)
        offset_world = yaw_rot @ offset_body
        ray_start = base_pos + offset_world
        ray_start[2] += HEIGHT_SCAN_RAY_START_Z
        distance = mujoco.mj_ray(model, data, ray_start, RAYCAST_DIR, None, 1, -1, None)
        if distance > 0.0:
            ground_z = float(ray_start[2] - distance)
            scan_value = float(base_pos[2] - ground_z - HEIGHT_SCAN_OFFSET)
        else:
            scan_value = HEIGHT_SCAN_CLIP_MAX
        scan_values[index] = np.clip(scan_value, HEIGHT_SCAN_CLIP_MIN, HEIGHT_SCAN_CLIP_MAX)
    return scan_values


def build_ppo_observers(legacy: Any, model: mujoco.MjModel):
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
        joint_pos_rel_leg = qpos_leg - legacy.DEFAULT_DOF_POS_LEG

        base_mat = data.xmat[base_body_id].reshape(3, 3).copy().astype(np.float32)
        base_lin_vel_world = data.qvel[0:3].copy().astype(np.float32)
        base_lin_vel = (base_mat.T @ base_lin_vel_world).astype(np.float32)
        base_ang_vel = data.qvel[3:6].copy().astype(np.float32)
        projected_gravity = (base_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
        height_scan = build_height_scan(model, data, base_body_id)

        policy_obs = np.concatenate(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                command.astype(np.float32),
                joint_pos_rel_leg[legacy.LEG2ISAACLAB_ORDER],
                qvel_leg[legacy.LEG2ISAACLAB_ORDER],
                last_action_leg[legacy.LEG2ISAACLAB_ORDER],
                height_scan,
            ]
        ).astype(np.float32)

        if policy_obs.shape[0] != PPO_POLICY_OBS_DIM:
            raise RuntimeError(
                f"policy_obs dim mismatch for PPO: got {policy_obs.shape[0]}, expected {PPO_POLICY_OBS_DIM}"
            )
        return policy_obs

    def build_policy_obs(current_obs: np.ndarray, history_buffer) -> np.ndarray:
        _ = history_buffer
        if current_obs.shape[0] != PPO_POLICY_OBS_DIM:
            raise RuntimeError(
                f"policy_obs dim mismatch for PPO: got {current_obs.shape[0]}, expected {PPO_POLICY_OBS_DIM}"
            )
        return current_obs.astype(np.float32, copy=False)

    return build_current_obs, build_policy_obs


def build_ppo45_observers(legacy: Any, build_current_obs):
    def build_scaled_current_obs(*args, **kwargs) -> np.ndarray:
        current_obs = build_current_obs(*args, **kwargs).copy()
        current_obs[0:3] *= PPO45_BASE_ANG_VEL_SCALE
        current_obs[21:33] *= PPO45_JOINT_VEL_SCALE
        return current_obs

    def build_policy_obs(current_obs: np.ndarray, history_buffer) -> np.ndarray:
        _ = history_buffer
        if current_obs.shape[0] != PPO45_POLICY_OBS_DIM:
            raise RuntimeError(
                f"policy_obs dim mismatch for PPO45: got {current_obs.shape[0]}, expected {PPO45_POLICY_OBS_DIM}"
            )
        return current_obs.astype(np.float32, copy=False)

    return build_scaled_current_obs, build_policy_obs


def apply_observer_mode(
    legacy: Any,
    controller: MixedControllerConfig,
    ppo45_observers,
    ppo235_observers,
    dreamwaq_observers,
) -> None:
    if controller.input_mode == "ppo45":
        legacy.HISTORY_LEN = PPO45_HISTORY_LEN
        legacy.CURRENT_OBS_DIM = PPO45_CURRENT_OBS_DIM
        legacy.POLICY_OBS_DIM = PPO45_POLICY_OBS_DIM
        legacy.build_current_obs, legacy.build_policy_obs = ppo45_observers
    elif controller.input_mode == "ppo235":
        legacy.HISTORY_LEN = PPO_HISTORY_LEN
        legacy.CURRENT_OBS_DIM = PPO_CURRENT_OBS_DIM
        legacy.POLICY_OBS_DIM = PPO_POLICY_OBS_DIM
        legacy.build_current_obs, legacy.build_policy_obs = ppo235_observers
    elif controller.input_mode == "dreamwaq270":
        legacy.HISTORY_LEN = DREAMWAQ_HISTORY_LEN
        legacy.CURRENT_OBS_DIM = DREAMWAQ_CURRENT_OBS_DIM
        legacy.POLICY_OBS_DIM = DREAMWAQ_POLICY_OBS_DIM
        legacy.build_current_obs, legacy.build_policy_obs = dreamwaq_observers
    else:
        raise ValueError(f"Unsupported input mode: {controller.input_mode}")


def infer_real_sample_count(result: Any, control_dt: float, warmup_s: float) -> int:
    if result.instant_errors is None:
        return 0
    if not result.fell:
        return len(result.instant_errors)

    active_eval_time = max(0.0, float(result.simulated_time_s) - warmup_s)
    real_samples = int(np.rint(active_eval_time / control_dt)) if control_dt > 0.0 else 0
    return max(0, min(len(result.instant_errors), real_samples))


def collect_no_fall_penalty_axis_data_from_results(
    legacy: Any,
    results,
    controller_names: list[str],
    control_dt: float,
    warmup_s: float,
) -> dict[str, list[np.ndarray]]:
    grouped_results: dict[str, list[Any]] = {}
    for result in results:
        grouped_results.setdefault(result.controller, []).append(result)

    axis_data: dict[str, list[np.ndarray]] = {}
    for controller_name in controller_names:
        runs = grouped_results.get(controller_name, [])
        metric_chunks: list[list[np.ndarray]] = [[] for _ in legacy.AXIS_METRICS]

        for run in runs:
            if run.instant_errors is None or run.instant_errors.size == 0:
                continue
            real_count = infer_real_sample_count(run, control_dt, warmup_s)
            if real_count <= 0:
                continue

            real_errors = run.instant_errors[:real_count].astype(np.float64, copy=False)
            for axis_idx in range(len(legacy.AXIS_METRICS)):
                metric_chunks[axis_idx].append(real_errors[:, axis_idx])

        axis_data[controller_name] = [
            np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float64)
            for chunks in metric_chunks
        ]

    return axis_data


def collect_no_fall_penalty_axis_data_from_trace_dir(
    legacy: Any,
    results,
    controller_names: list[str],
    trace_dir: Path,
) -> dict[str, list[np.ndarray]]:
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    grouped_results: dict[str, list[Any]] = {}
    for result in results:
        grouped_results.setdefault(result.controller, []).append(result)

    axis_data: dict[str, list[np.ndarray]] = {}
    missing_traces: list[Path] = []
    for controller_name in controller_names:
        runs = grouped_results.get(controller_name, [])
        metric_chunks: list[list[np.ndarray]] = [[] for _ in legacy.AXIS_METRICS]

        for run in runs:
            trace_path = trace_dir / f"trace_{run.controller}_seed{run.seed}.csv"
            if not trace_path.is_file():
                missing_traces.append(trace_path)
                continue

            real_rows: list[list[float]] = []
            with trace_path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    cmd = np.array(
                        [float(row["cmd_vx"]), float(row["cmd_vy"]), float(row["cmd_yaw"])],
                        dtype=np.float64,
                    )
                    meas = np.array(
                        [float(row["meas_vx"]), float(row["meas_vy"]), float(row["meas_yaw"])],
                        dtype=np.float64,
                    )
                    if not (np.all(np.isfinite(cmd)) and np.all(np.isfinite(meas))):
                        continue

                    real_rows.append(
                        [
                            float(row["abs_err_vx"]),
                            float(row["abs_err_vy"]),
                            float(row["abs_err_yaw"]),
                        ]
                    )

            if not real_rows:
                continue

            real_errors = np.asarray(real_rows, dtype=np.float64)
            for axis_idx in range(len(legacy.AXIS_METRICS)):
                metric_chunks[axis_idx].append(real_errors[:, axis_idx])

        axis_data[controller_name] = [
            np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float64)
            for chunks in metric_chunks
        ]

    if missing_traces:
        print("[WARN] Missing trace files for no-fall-penalty boxplot:")
        for path in missing_traces[:10]:
            print(f"       {path}")
        if len(missing_traces) > 10:
            print(f"       ... and {len(missing_traces) - 10} more")

    return axis_data


def save_boxplot_stats(
    output_path: Path,
    axis_data: dict[str, list[np.ndarray]],
    axis_metrics,
    sample_type: str,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
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
            for values, (metric_key, _) in zip(metric_arrays, axis_metrics):
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


def plot_axis_error_boxplot_from_axis_data(
    legacy: Any,
    output_dir: Path,
    axis_data: dict[str, list[np.ndarray]],
    reference_controller: str,
    controller_names: list[str],
    output_filename: str,
    stats_filename: str,
    sample_type: str,
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
        print(f"[WARN] matplotlib unavailable, skip {output_filename}: {exc}")
        return

    if not controller_names:
        print(f"[WARN] No controllers available for {output_filename}")
        return

    has_any_data = any(
        len(values) > 0 for metric_arrays in axis_data.values() for values in metric_arrays
    )
    if not has_any_data:
        print(f"[WARN] No real pre-fall tracking errors available for {output_filename}")
        return

    save_boxplot_stats(output_dir / stats_filename, axis_data, legacy.AXIS_METRICS, sample_type)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    num_groups = len(legacy.AXIS_METRICS)
    num_controllers = len(controller_names)
    group_centers = np.arange(num_groups, dtype=np.float64)
    group_width = min(0.78, 0.14 * num_controllers + 0.16)
    box_width = group_width / max(num_controllers, 1) * 0.92
    offsets = np.linspace(
        -group_width / 2.0 + box_width / 2.0,
        group_width / 2.0 - box_width / 2.0,
        num_controllers,
    )

    legend_handles = []
    legend_labels = []
    global_max = 0.0

    for controller_idx, controller_name in enumerate(controller_names):
        controller_color = legacy.get_controller_color(controller_name, controller_idx)
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
        legend_labels.append(legacy.get_controller_display_name(controller_name))

    ax.set_ylabel("Tracking error", fontsize=20)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([axis_label for _, axis_label in legacy.AXIS_METRICS], fontsize=20)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", linestyle="-", color="#b5b5b5", linewidth=1.0)

    y_upper = ax.get_ylim()[1]
    y_padding = max(0.01, global_max * 0.08)
    ax.set_ylim(0.0, max(y_upper, global_max + y_padding + 0.01))

    if legend_handles:
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
    fig.savefig(output_dir / output_filename, dpi=300)
    plt.close(fig)

    print(
        f"[INFO] Saved {output_filename} using real tracking errors before falling only; "
        "fall-penalty samples are excluded."
    )


def render_no_fall_penalty_boxplot(
    legacy: Any,
    output_dir: Path,
    results,
    controller_names: list[str],
    reference_controller: str,
    control_dt: float | None = None,
    warmup_s: float | None = None,
    trace_dir: Path | None = None,
) -> None:
    if trace_dir is not None:
        axis_data = collect_no_fall_penalty_axis_data_from_trace_dir(
            legacy, results, controller_names, trace_dir
        )
    elif control_dt is not None and warmup_s is not None:
        axis_data = collect_no_fall_penalty_axis_data_from_results(
            legacy, results, controller_names, control_dt, warmup_s
        )
    else:
        print(
            "[WARN] Skip tracking_error_boxplot_no_fall_penalty.png because no trace_dir "
            "or live timing metadata was provided."
        )
        return

    plot_axis_error_boxplot_from_axis_data(
        legacy=legacy,
        output_dir=output_dir,
        axis_data=axis_data,
        reference_controller=reference_controller,
        controller_names=controller_names,
        output_filename="tracking_error_boxplot_no_fall_penalty.png",
        stats_filename="instant_boxplot_stats_no_fall_penalty.csv",
        sample_type="instant_time_step_errors_no_fall_penalty",
    )


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    controller_catalog = build_controller_catalog(args.policy_path)
    controllers = resolve_controllers(controller_catalog, args.controllers)

    requested_reference = args.reference_controller.strip()
    available_names = {controller.name for controller in controllers}
    reference_controller = requested_reference if requested_reference in available_names else controllers[0].name

    legacy = load_legacy_module()
    configure_legacy_metadata(legacy, controllers, reference_controller)
    dreamwaq_observers = (legacy.build_current_obs, legacy.build_policy_obs)
    ppo45_observers = build_ppo45_observers(legacy, dreamwaq_observers[0])

    if args.plot_only_run_metrics is not None:
        run_metrics_path = args.plot_only_run_metrics.resolve()
        output_dir = legacy.build_output_dir(args.output_dir) if args.output_dir is not None else run_metrics_path.parent
        results = legacy.filter_results_by_controller_names(legacy.load_run_metrics_csv(run_metrics_path), args.controllers)
        trace_dir = args.plot_only_trace_dir.resolve() if args.plot_only_trace_dir is not None else None
        if args.plot_only_trace_dir is not None:
            legacy.attach_instant_errors_from_trace_dir(results, trace_dir)
        controller_order = legacy.get_controller_order(results)
        summary, comparisons, axis_comparisons = legacy.render_plots_and_stats(
            output_dir=output_dir,
            results=results,
            reference_controller=reference_controller if reference_controller in controller_order else controller_order[0],
            controller_order=controller_order,
        )
        render_no_fall_penalty_boxplot(
            legacy=legacy,
            output_dir=output_dir,
            results=results,
            controller_names=controller_order,
            reference_controller=reference_controller if reference_controller in controller_order else controller_order[0],
            trace_dir=trace_dir,
        )
        print("[INFO] Re-rendered plots from existing run_metrics.csv")
        print(f"[INFO] run_metrics.csv: {run_metrics_path}")
        print(f"[INFO] output_dir: {output_dir}")
        print(f"[INFO] reference_controller: {reference_controller}")
        print(f"[INFO] num_total_t_tests: {len(comparisons)}")
        print(f"[INFO] num_axis_t_tests: {len(axis_comparisons)}")
        print(f"[INFO] summary_keys: {list(summary.keys())}")
        return

    xml_path = args.xml_path.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    for controller in controllers:
        if not controller.policy_path.is_file():
            raise FileNotFoundError(f"Policy file not found: {controller.policy_path}")

    output_dir = legacy.build_output_dir(args.output_dir)
    trace_dir = output_dir / "traces" if args.save_traces else None
    seeds = legacy.parse_seed_list(args.seeds)

    print("[INFO] Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos_idx, qvel_idx = legacy.get_joint_indices(model)
    base_body_id = legacy.find_body_id(model, ["base_link", "trunk", "base", "body"])
    ppo235_observers = build_ppo_observers(legacy, model)

    print(f"[INFO] XML: {xml_path}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] controllers: {[controller.name for controller in controllers]}")
    print(f"[INFO] reference_controller: {reference_controller}")
    print(f"[INFO] seeds: {seeds}")
    control_dt = model.opt.timestep * legacy.DECIMATION

    command_sequences = {
        seed: legacy.generate_command_sequence(seed, args.duration, args.command_interval) for seed in seeds
    }
    legacy.save_command_sequences(output_dir, command_sequences, args.command_interval, args.duration)

    loaded_policies: dict[str, torch.jit.ScriptModule] = {}
    for controller in controllers:
        print(f"[INFO] Loading policy: {controller.name} -> {controller.policy_path}")
        policy = torch.jit.load(str(controller.policy_path), map_location="cpu")
        policy.eval()
        loaded_policies[controller.name] = policy

    all_results = []
    for controller in controllers:
        legacy_controller = to_legacy_controller(legacy, controller)
        apply_observer_mode(legacy, controller, ppo45_observers, ppo235_observers, dreamwaq_observers)
        policy = loaded_policies[controller.name]
        for seed in seeds:
            print(f"[RUN] controller={controller.name} seed={seed} input_mode={controller.input_mode}")
            result = legacy.run_single_controller_seed(
                controller=legacy_controller,
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
            print(f"[DONE] {controller.name} seed={seed} {legacy.format_run_tracking_error(result)} status={status}")

    controller_order = [controller.name for controller in controllers]
    summary, comparisons, axis_comparisons = legacy.render_plots_and_stats(
        output_dir=output_dir,
        results=all_results,
        reference_controller=reference_controller,
        controller_order=controller_order,
    )
    render_no_fall_penalty_boxplot(
        legacy=legacy,
        output_dir=output_dir,
        results=all_results,
        controller_names=controller_order,
        reference_controller=reference_controller,
        control_dt=control_dt,
        warmup_s=args.warmup,
    )
    legacy.save_summary_json(
        output_dir,
        args,
        [to_legacy_controller(legacy, controller) for controller in controllers],
        seeds,
        reference_controller,
        summary,
        comparisons,
    )

    print("\n================ Evaluation Complete ================")
    for controller_name, item in summary.items():
        print(legacy.format_summary_tracking_error(controller_name, item))
    print("-----------------------------------------------------")
    print(f"[INFO] run_metrics.csv: {output_dir / 'run_metrics.csv'}")
    print(f"[INFO] summary.json: {output_dir / 'summary.json'}")
    print(f"[INFO] boxplot: {output_dir / 'tracking_error_boxplot.png'}")
    print(f"[INFO] boxplot_no_fall_penalty: {output_dir / 'tracking_error_boxplot_no_fall_penalty.png'}")
    print(f"[INFO] barplot: {output_dir / 'ate_barplot.png'}")
    print(f"[INFO] total_t_tests: {len(comparisons)}")
    print(f"[INFO] axis_t_tests: {len(axis_comparisons)}")


if __name__ == "__main__":
    main()
