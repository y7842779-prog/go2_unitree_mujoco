import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from collections import deque
from pynput import keyboard

import sys
import termios
import atexit

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# ==========================================
# 0. 终端恢复设置，防止退出后终端不显示输入
# ==========================================
TERM_AVAILABLE = sys.stdin.isatty()

if TERM_AVAILABLE:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def restore_terminal():
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    atexit.register(restore_terminal)

    new_settings = termios.tcgetattr(fd)
    new_settings[3] = new_settings[3] & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)


# ==========================================
# 1. 路径和核心参数
# ==========================================

XML_PATH = "../unitree_robots/go2/scene_terrain.xml"
POLICY_PATH = "DreamWaQ_models/policy_5N_foce.pt"

# DreamWaQ / CENet 新模型观测维度
HISTORY_LEN = 5
CURRENT_OBS_DIM = 45
POLICY_OBS_DIM = 270
NUM_ACTIONS = 12

# 控制参数
ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5
ACTION_CLIP = 14.0

# MuJoCo 仿真步长通常是 0.002，DECIMATION=10 等价于 50Hz 控制
DECIMATION = 10

# 先只用 PD 固定默认站姿，之后再切到 policy，方便区分“默认姿态”与“policy 接管”
PD_ONLY_WARMUP_TIME = 2.0
POLICY_RAMP_TIME = 0.5
USE_FIXED_COMMAND_TEST = False
FIXED_COMMAND = np.array([0.4, 0.0, 0.0], dtype=np.float32)

# 初始身体高度
INIT_BASE_HEIGHT = 0.445

# Go2 默认关节角：物理腿分组顺序 FL / FR / RL / RR
DEFAULT_DOF_POS_LEG = np.array([
     0.1,  0.8, -1.5,  # FL: hip, thigh, calf
    -0.1,  0.8, -1.5,  # FR
     0.1,  1.0, -1.5,  # RL
    -0.1,  1.0, -1.5,  # RR
], dtype=np.float32)

# 你的旧脚本里的映射继续保留：
# MuJoCo / 物理腿顺序:
#   FL_hip, FL_thigh, FL_calf,
#   FR_hip, FR_thigh, FR_calf,
#   RL_hip, RL_thigh, RL_calf,
#   RR_hip, RR_thigh, RR_calf
#
# IsaacLab 网络顺序:
#   FL_hip, FR_hip, RL_hip, RR_hip,
#   FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#   FL_calf, FR_calf, RL_calf, RR_calf
LEG2ISAACLAB_ORDER = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ISAACLAB2LEG_ORDER = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

# MuJoCo actuator ctrl 顺序，沿用你原来的
CTRL_MUJOCO_IDX = [
    3, 4, 5,      # FL
    0, 1, 2,      # FR
    9, 10, 11,    # RL
    6, 7, 8,      # RR
]

# 按 go2.xml 的 actuator ctrlrange 做关节级限幅：
# hip(abduction) / thigh 使用 +/-23.7，calf(knee) 使用 +/-45.43。
TORQUE_LIMIT_LEG = np.array([
    23.7, 23.7, 45.43,  # FL
    23.7, 23.7, 45.43,  # FR
    23.7, 23.7, 45.43,  # RL
    23.7, 23.7, 45.43,  # RR
], dtype=np.float32)

LEG_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# 诊断参数：周期性打印控制/姿态摘要，并在接近力矩饱和或姿态过大时报警。
DIAG_PRINT_INTERVAL = 1.0
SATURATION_WARN_RATIO = 0.90
POSTURE_WARN_ANGLE_RAD = 0.60
WARN_COOLDOWN = 0.5
ACTION_CLIP_WARN_COOLDOWN = 1.0


# ==========================================
# 2. 工具函数
# ==========================================

def find_body_id(model, possible_names):
    for name in possible_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            print(f"[INFO] 使用 base body: {name}")
            return body_id

    raise RuntimeError(f"找不到 base body，尝试过: {possible_names}")


def get_joint_indices(model):
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    # 有些 XML 里 joint 名字没有 "_joint"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[0]) == -1:
        joint_names = [name.replace("_joint", "") for name in joint_names]

    qpos_idx = []
    qvel_idx = []

    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id == -1:
            raise RuntimeError(f"找不到关节: {name}")

        qpos_idx.append(model.jnt_qposadr[jnt_id])
        qvel_idx.append(model.jnt_dofadr[jnt_id])

    print("[INFO] 关节顺序:")
    for i, name in enumerate(joint_names):
        print(f"  {i:02d}: {name}")

    return np.array(qpos_idx), np.array(qvel_idx)


def build_current_obs(data, base_body_id, qpos_idx, qvel_idx, last_action_leg, command):
    """
    构造 DreamWaQ 当前 45 维 obs。

    current_obs =
        base_ang_vel       3
        projected_gravity  3
        command            3
        joint_pos_rel     12
        joint_vel         12
        last_action       12
    """

    qpos_leg = data.qpos[qpos_idx].copy().astype(np.float32)
    qvel_leg = data.qvel[qvel_idx].copy().astype(np.float32)

    joint_pos_rel_leg = qpos_leg - DEFAULT_DOF_POS_LEG

    base_mat = data.xmat[base_body_id].reshape(3, 3).copy()

    # MuJoCo free joint 的 qvel 约定是：前 3 维线速度在世界系，后 3 维角速度在机体系。
    # 所以这里直接取 qvel[3:6]，不要再乘一次 base_mat.T。
    base_ang_vel = data.qvel[3:6].copy().astype(np.float32)

    projected_gravity = (
        base_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    ).astype(np.float32)

    # 进入网络前，把物理腿顺序转成 IsaacLab 网络顺序
    joint_pos_rel_net = joint_pos_rel_leg[LEG2ISAACLAB_ORDER]
    joint_vel_net = qvel_leg[LEG2ISAACLAB_ORDER]
    last_action_net = last_action_leg[LEG2ISAACLAB_ORDER]

    current_obs = np.concatenate([
        base_ang_vel,          # 3
        projected_gravity,     # 3
        command,               # 3
        joint_pos_rel_net,     # 12
        joint_vel_net,         # 12
        last_action_net,       # 12
    ]).astype(np.float32)

    assert current_obs.shape[0] == CURRENT_OBS_DIM, (
        f"current_obs 维度错误: {current_obs.shape[0]}，期待 {CURRENT_OBS_DIM}"
    )

    return current_obs


def build_policy_obs(current_obs, history_buffer):
    """
    policy_obs = current_obs 45 + history_obs 225 = 270

    这里的 history_obs 必须匹配 IsaacLab 训练时 ObservationGroup 的拼接方式：
    不是 [frame1(45), frame2(45), ...] 的帧优先拼接，
    而是 [base_ang_vel_history, projected_gravity_history, ...] 的 term 优先拼接。
    """

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
    history_obs = np.concatenate([history_frames[:, term_slice].reshape(-1) for term_slice in term_slices]).astype(np.float32)

    policy_obs = np.concatenate([
        current_obs,
        history_obs,
    ]).astype(np.float32)

    assert history_obs.shape[0] == HISTORY_LEN * CURRENT_OBS_DIM, (
        f"history_obs 维度错误: {history_obs.shape[0]}，期待 {HISTORY_LEN * CURRENT_OBS_DIM}"
    )

    assert policy_obs.shape[0] == POLICY_OBS_DIM, (
        f"policy_obs 维度错误: {policy_obs.shape[0]}，期待 {POLICY_OBS_DIM}"
    )

    return policy_obs


def format_float_array(values, decimals=2):
    values = np.asarray(values, dtype=np.float64)
    return np.array2string(
        values,
        max_line_width=10_000,
        separator=", ",
        formatter={"float_kind": lambda x: f"{x:.{decimals}f}"},
    )


# ==========================================
# 3. 键盘控制
# ==========================================

cmd_vx = 0.0
cmd_vy = 0.0
cmd_yaw = 0.0


def on_press(key):
    global cmd_vx, cmd_vy, cmd_yaw
    try:
        if key.char == "w":
            cmd_vx = 0.5
        elif key.char == "s":
            cmd_vx = -0.3
        elif key.char == "a":
            cmd_vy = 0.3
        elif key.char == "d":
            cmd_vy = -0.3
        elif key.char == "q":
            cmd_yaw = 0.6
        elif key.char == "e":
            cmd_yaw = -0.6
    except AttributeError:
        pass


def on_release(key):
    global cmd_vx, cmd_vy, cmd_yaw
    try:
        if key.char in ["w", "s"]:
            cmd_vx = 0.0
        if key.char in ["a", "d"]:
            cmd_vy = 0.0
        if key.char in ["q", "e"]:
            cmd_yaw = 0.0
    except AttributeError:
        pass


# ==========================================
# 4. 初始化 MuJoCo 和 policy
# ==========================================

print("[INFO] 加载 MuJoCo 模型...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

qpos_idx, qvel_idx = get_joint_indices(model)
base_body_id = find_body_id(model, ["base_link", "trunk", "base", "body"])

print("[INFO] 加载策略...")
try:
    policy = torch.jit.load(POLICY_PATH, map_location="cpu")
    policy.eval()
    print(f"[INFO] policy 加载成功: {POLICY_PATH}")
except Exception as e:
    print(f"[ERROR] 策略加载失败: {e}")
    sys.exit(1)

# 初始化姿势
data.qpos[2] = INIT_BASE_HEIGHT

for i in range(NUM_ACTIONS):
    data.qpos[qpos_idx[i]] = DEFAULT_DOF_POS_LEG[i]

mujoco.mj_forward(model, data)

# 初始化控制量
actions_leg = np.zeros(NUM_ACTIONS, dtype=np.float32)
target_dof_pos_leg = DEFAULT_DOF_POS_LEG.copy()

# 初始化 history buffer
history_buffer = deque(maxlen=HISTORY_LEN)

# 先用零指令、零 action 填满 5 帧 history
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

# 启动键盘监听
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("=========================================")
print(" DreamWaQ / CENet MuJoCo Sim2Sim 已就绪")
print(" 当前模型输入: 270 = current_obs 45 + history_obs 225")
print(" [W] / [S] : 前进 / 后退")
print(" [A] / [D] : 左平移 / 右平移")
print(" [Q] / [E] : 左转 / 右转")
if USE_FIXED_COMMAND_TEST:
    print(f" 固定小指令测试已开启: command = {FIXED_COMMAND.tolist()}")
else:
    print(" 键盘控制已开启: 使用 W/S/A/D/Q/E 下发速度指令")
print(" 力矩限幅已对齐 go2.xml: hip/thigh=23.7 Nm, calf=45.43 Nm")
print(" 提示：键盘需要选中终端窗口，不是 MuJoCo 窗口")
print("=========================================")


# ==========================================
# 5. 主仿真循环
# ==========================================

print("[INFO] 启动 MuJoCo viewer...")

log_times = []
log_rolls = []
log_pitches = []

debug_printed = False
policy_started = False
last_diag_print_time = -1.0e9
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_counter = 0

        while viewer.is_running():
            step_start = time.time()
            sim_time = step_counter * model.opt.timestep

            qpos_leg = data.qpos[qpos_idx].copy()
            qvel_leg = data.qvel[qvel_idx].copy()

            if step_counter % DECIMATION == 0:
                base_mat = data.xmat[base_body_id].reshape(3, 3).copy()
                base_lin_vel_b = (base_mat.T @ data.qvel[0:3]).astype(np.float32)
                actual_vx = float(base_lin_vel_b[0])
                r = R.from_matrix(base_mat)
                euler_angles = r.as_euler("xyz", degrees=False)
                roll = float(euler_angles[0])
                pitch = float(euler_angles[1])

                # 记录姿态，跳过前 1.5s 起步阶段
                if sim_time > 1.5:
                    log_times.append(sim_time)
                    log_rolls.append(roll)
                    log_pitches.append(pitch)

                # 前 2 秒完全不用 policy，只用 PD 固定默认站姿，单独观察默认姿态本身长什么样。
                if sim_time < PD_ONLY_WARMUP_TIME:
                    command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                elif USE_FIXED_COMMAND_TEST:
                    command = FIXED_COMMAND.copy()
                else:
                    command = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float32)

                # 1. 构造当前 45 维 obs
                current_obs = build_current_obs(
                    data=data,
                    base_body_id=base_body_id,
                    qpos_idx=qpos_idx,
                    qvel_idx=qvel_idx,
                    last_action_leg=actions_leg,
                    command=command,
                )

                # 2. 无论是否启用 policy，都持续更新历史，让切换时 CENet 能看到真实最近 5 帧。
                history_buffer.append(current_obs.copy())

                # 3. warmup 期间完全不用 policy，直接把 action 置零，让 PD 只追默认站姿。
                if sim_time < PD_ONLY_WARMUP_TIME:
                    actions_leg = np.zeros(NUM_ACTIONS, dtype=np.float32)
                else:
                    obs_np = build_policy_obs(current_obs, history_buffer)

                    if not debug_printed:
                        print("[DEBUG] current_obs shape:", current_obs.shape)
                        print("[DEBUG] policy_obs shape:", obs_np.shape)
                        print("[DEBUG] first 10 obs:", format_float_array(obs_np[:10]))
                        debug_printed = True

                    if not policy_started:
                        print(f"[INFO] sim_time={sim_time:.2f}s，开始从纯 PD 站姿切换到 policy 控制")
                        policy_started = True

                    obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0)

                    # 4. policy.pt 内部会自动切片：
                    #    obs[:, :45]  -> current_obs
                    #    obs[:, 45:]  -> history_obs
                    #    history_obs -> CENet -> velocity_estimate, latent_mu
                    #    actor 输出 action
                    with torch.no_grad():
                        raw_actions_net = policy(obs_tensor)

                        # 防止某些导出模型返回 tuple/list
                        if isinstance(raw_actions_net, (tuple, list)):
                            raw_actions_net = raw_actions_net[0]

                        raw_actions_net = raw_actions_net.cpu().numpy().squeeze().astype(np.float32)

                    assert raw_actions_net.shape[0] == NUM_ACTIONS, (
                        f"action 维度错误: {raw_actions_net.shape}，期待 12"
                    )

                    # 5. 网络 action 顺序转回物理腿分组顺序。
                    raw_actions_leg = raw_actions_net[ISAACLAB2LEG_ORDER]

                    # 6. 部署侧显式裁剪 action，避免 sim-to-sim 偏差把某一拍输出放大后直接打爆 PD。
                    clipped_actions_leg = np.clip(raw_actions_leg, -ACTION_CLIP, ACTION_CLIP)
                    if sim_time - last_diag_print_time >= DIAG_PRINT_INTERVAL:
                        print(f"[POLICY] t={sim_time:6.2f}s")
                        print(
                            f"[POLICY] cmd=({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}) "
                            f"vx={actual_vx:+.2f}"
                        )
                        print(f"[POLICY] action={format_float_array(raw_actions_net)}")
                        last_diag_print_time = sim_time

                    # 7. 切换到 policy 时做一个短暂渐变，避免接管瞬间猛抽。
                    if sim_time < PD_ONLY_WARMUP_TIME + POLICY_RAMP_TIME:
                        alpha = (sim_time - PD_ONLY_WARMUP_TIME) / POLICY_RAMP_TIME
                        actions_leg = clipped_actions_leg * alpha
                    else:
                        actions_leg = clipped_actions_leg

            # PD 控制
            target_dof_pos_leg = DEFAULT_DOF_POS_LEG + ACTION_SCALE * actions_leg
            tau_leg = KP * (target_dof_pos_leg - qpos_leg) - KD * qvel_leg
            tau_leg = np.clip(tau_leg, -TORQUE_LIMIT_LEG, TORQUE_LIMIT_LEG)

            # 映射到 MuJoCo actuator ctrl 顺序
            tau_ctrl = tau_leg[CTRL_MUJOCO_IDX]

            if data.ctrl.shape[0] == NUM_ACTIONS:
                data.ctrl[:] = tau_ctrl
            else:
                data.ctrl[:NUM_ACTIONS] = tau_ctrl

            mujoco.mj_step(model, data)
            step_counter += 1

            if step_counter % DECIMATION == 0:
                viewer.sync()

                elapsed = time.time() - step_start
                step_dt = model.opt.timestep * DECIMATION
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)

finally:
    try:
        listener.stop()
    except Exception:
        pass


# ==========================================
# 6. 仿真结束：姿态统计
# ==========================================

if len(log_times) > 0:
    print("\n[INFO] 正在计算躯干姿态方差并生成图表...")

    rolls_np = np.array(log_rolls)
    pitches_np = np.array(log_pitches)

    roll_var = np.var(rolls_np)
    pitch_var = np.var(pitches_np)
    roll_std = np.std(rolls_np)
    pitch_std = np.std(pitches_np)
    roll_max = np.max(np.abs(rolls_np))
    pitch_max = np.max(np.abs(pitches_np))

    print("================ DreamWaQ 姿态测试结果 ================")
    print(f"Roll  -> 方差: {roll_var:.6f} rad^2 | 标差: {roll_std:.4f} rad | 最大偏差: {roll_max:.4f} rad")
    print(f"Pitch -> 方差: {pitch_var:.6f} rad^2 | 标差: {pitch_std:.4f} rad | 最大偏差: {pitch_max:.4f} rad")
    print("=======================================================")

    plt.figure(figsize=(10, 5))
    plt.plot(log_times, rolls_np, label=f"Roll (Var: {roll_var:.4f})", alpha=0.8)
    plt.plot(log_times, pitches_np, label=f"Pitch (Var: {pitch_var:.4f})", alpha=0.8)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.title("Go2 Base Orientation with DreamWaQ CENet Policy")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)

    plt.savefig("orientation_variance_dreamwaq_cenet.png", dpi=300)
    print("[INFO] 图表已保存为 orientation_variance_dreamwaq_cenet.png")
    plt.show()
else:
    print("[WARN] 运行时间太短，没有收集到足够的评估数据。")
