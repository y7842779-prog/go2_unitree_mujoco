import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
import sys
import termios
import atexit
import matplotlib.pyplot as plt

# ==========================================
# 终端设置清理
# ==========================================
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)

def restore_terminal():
    termios.tcflush(fd, termios.TCIFLUSH)
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

atexit.register(restore_terminal)
new_settings = termios.tcgetattr(fd)
new_settings[3] = new_settings[3] & ~termios.ECHO
termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

# ==========================================
# 1. 全局配置与平滑度测试参数
# ==========================================
XML_PATH = "../unitree_robots/go2_Oracle/scene.xml"
POLICY_PATH = "models_oracle/policy.pt"

TEST_TARGET_VX = 1.0  # 测试基准速度
MAX_SIM_TIME = 8.0    # 测试时长
START_EVAL_TIME = 1.5 # 避开前1.5秒的起步不稳阶段

ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5

DEFAULT_DOF_POS_ISAAC = np.array([
     0.1,  0.8, -1.5, -0.1,  0.8, -1.5, 0.1,  1.0, -1.5, -0.1,  1.0, -1.5   
], dtype=np.float32)
LEG2ISAACLAB_ORDER = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ISAACLAB2LEG_ORDER = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

# ==========================================
# 2. 初始化环境与模型
# ==========================================
print("[INFO] 加载 MuJoCo 模型与策略...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

isaac_joint_names = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, isaac_joint_names[0]) == -1:
    isaac_joint_names = [name.replace("_joint", "") for name in isaac_joint_names]

QPOS_ISAAC_IDX = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in isaac_joint_names]
QVEL_ISAAC_IDX = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in isaac_joint_names]
CTRL_MUJOCO_IDX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

policy = torch.jit.load(POLICY_PATH)
policy.eval()

base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_ball")

data.qpos[2] = 0.42  
ball_jnt_id = model.body_jntadr[ball_body_id]
data.qpos[model.jnt_qposadr[ball_jnt_id] + 2] = 0.48
for i in range(12): data.qpos[QPOS_ISAAC_IDX[i]] = DEFAULT_DOF_POS_ISAAC[i]

mujoco.mj_forward(model, data)
current_xml_mass = model.body_mass[ball_body_id]

# ==========================================
# 3. 平滑度数据记录器
# ==========================================
log_times = []
log_actions = [] # 记录神经网络输出的 raw actions
log_taus = []    # 记录底层电机实际输出的 torques

print("=========================================")
print(f" 🚀 控制平滑度测试准备就绪！")
print(f"    目标速度: {TEST_TARGET_VX} m/s")
print(f"    评估指标: 动作跳变率 & 力矩平滑度")
print("=========================================")

# ==========================================
# 4. 主仿真循环
# ==========================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    decimation = 10
    step_counter = 0
    actions_isaac = np.zeros(12, dtype=np.float32)

    while viewer.is_running():
        step_start = time.time()
        sim_time = step_counter * model.opt.timestep
        
        if sim_time > MAX_SIM_TIME:
            print(f"\n[INFO] 测试结束。")
            break
            
        if data.xpos[base_body_id][2] < 0.18:
            print(f"\n[💥 致命崩溃] 机器狗摔倒了！")
            break

        qpos_isaac = data.qpos[QPOS_ISAAC_IDX]
        qvel_isaac = data.qvel[QVEL_ISAAC_IDX]

        if step_counter % decimation == 0:
            joint_pos_rel = qpos_isaac - DEFAULT_DOF_POS_ISAAC
            base_mat = data.xmat[base_body_id].reshape(3, 3)

            base_lin_vel_world = data.qvel[0:3]
            base_lin_vel = base_mat.T @ base_lin_vel_world 
            base_ang_vel = data.qvel[3:6]
            projected_gravity = base_mat.T @ np.array([0.0, 0.0, -1.0])

            ball_pos_world = data.xpos[ball_body_id]
            ball_dof_idx = model.jnt_dofadr[ball_jnt_id]
            ball_vel_world = data.qvel[ball_dof_idx : ball_dof_idx + 3]
            ball_pos_relative = base_mat.T @ (ball_pos_world - data.xpos[base_body_id])
            ball_vel_relative = base_mat.T @ (ball_vel_world - base_lin_vel_world)
            ball_mass = np.array([current_xml_mass], dtype=np.float32)

            base_pos = data.xpos[base_body_id]
            scan_x, scan_y = np.meshgrid(np.linspace(0.4, -0.4, 5), np.linspace(0.4, -0.4, 5), indexing='ij')
            scan_offsets = np.column_stack((scan_x.flatten(), scan_y.flatten(), np.zeros(25)))
            
            real_heights = []
            for offset in scan_offsets:
                p_start = base_pos + base_mat @ offset + np.array([0, 0, 0.5])
                dist = mujoco.mj_ray(model, data, p_start, np.array([0, 0, -1.0]), None, 1, -1, None)
                real_heights.append((p_start[2] - dist - base_pos[2]) + 0.30 if dist > 0 else -0.5)
            fake_height_scan = np.array(real_heights, dtype=np.float32)
            
            command = np.array([0.0, 0.0, 0.0], dtype=np.float32) if sim_time < 1.5 else np.array([TEST_TARGET_VX, 0.0, 0.0], dtype=np.float32)

            obs_np = np.concatenate([
                base_lin_vel, base_ang_vel, projected_gravity, command,
                joint_pos_rel[LEG2ISAACLAB_ORDER], qvel_isaac[LEG2ISAACLAB_ORDER], actions_isaac[LEG2ISAACLAB_ORDER],
                fake_height_scan, ball_pos_relative, ball_vel_relative, ball_mass
            ]).astype(np.float32)
            
            with torch.no_grad():
                raw_actions_leg = policy(torch.tensor(obs_np).unsqueeze(0)).numpy().squeeze()[ISAACLAB2LEG_ORDER]

            if sim_time < 1.0: 
                actions_isaac = np.zeros(12, dtype=np.float32)
            elif sim_time < 1.5: 
                actions_isaac = raw_actions_leg * ((sim_time - 1.0) / 0.5)
            else: 
                actions_isaac = raw_actions_leg

            target_dof_pos_isaac = actions_isaac * ACTION_SCALE + DEFAULT_DOF_POS_ISAAC
            tau_isaac = np.clip(KP * (target_dof_pos_isaac - qpos_isaac) - KD * qvel_isaac, -20.0, 20.0)
            
            # 🌟🌟🌟 核心：记录数据供后续计算差值 🌟🌟🌟
            if sim_time >= START_EVAL_TIME:
                log_times.append(sim_time)
                log_actions.append(actions_isaac.copy()) # 保存 12 维动作
                log_taus.append(tau_isaac.copy())        # 保存 12 维力矩

        data.ctrl[:] = tau_isaac[CTRL_MUJOCO_IDX]

        mujoco.mj_step(model, data)
        step_counter += 1

        if step_counter % decimation == 0:
            viewer.sync()
            elapsed = time.time() - step_start
            step_dt = model.opt.timestep * decimation
            if elapsed < step_dt: time.sleep(step_dt - elapsed)

# ==========================================
# 5. 仿真结束：处理数据与直观绘图
# ==========================================
if len(log_times) > 1:
    print("\n[INFO] 正在计算平滑度指标...")
    
    times_np = np.array(log_times)
    actions_np = np.array(log_actions) # Shape: (T, 12)
    taus_np = np.array(log_taus)       # Shape: (T, 12)
    
    # 计算相邻步的差值 (Delta)
    delta_actions = np.diff(actions_np, axis=0) # Shape: (T-1, 12)
    delta_taus = np.diff(taus_np, axis=0)       # Shape: (T-1, 12)
    plot_times = times_np[1:]                   # 时间轴错开一位
    
    # 按照截图公式：计算每个时间步所有 12 个关节的差值平方和
    action_smoothness_cost = np.sum(delta_actions**2, axis=1) # 动作跳变惩罚
    torque_smoothness_cost = np.sum(delta_taus**2, axis=1)    # 力矩震荡惩罚
    
    # 计算整体累计惩罚分数 (越低越好)
    total_action_penalty = np.sum(action_smoothness_cost)
    total_torque_penalty = np.sum(torque_smoothness_cost)
    
    print(f"=========================================")
    print(f" 🏆 控制平滑度考核结果")
    print(f" 📉 动作总跳变罚分 (Action Rate): {total_action_penalty:.2f}")
    print(f" 📉 力矩总震荡罚分 (Torque Smoothness): {total_torque_penalty:.2f}")
    print(f"=========================================")
    
    # 导出 CSV
    export_data = np.column_stack((plot_times, action_smoothness_cost, torque_smoothness_cost))
    csv_filename = f'origin_smoothness_vx_{TEST_TARGET_VX}.csv'
    np.savetxt(csv_filename, export_data, delimiter=',', header='Time(s),Action_Jitter_Penalty,Torque_Jitter_Penalty', comments='')
    
    # 🌟🌟🌟 生成超级直观的上下双图 🌟🌟🌟
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- 图 1：动作指令的“裸眼透视” (直观看到网络输出有没有乱跳) ---
    for i in range(12):
        # 画出所有 12 个关节的 Action 轨迹
        ax1.plot(times_np, actions_np[:, i], alpha=0.4, linewidth=1.5)
    
    ax1.set_title('Neural Network Raw Output (12 Joints Actions)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Action Value', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # 把动作惩罚的总分贴在第一张图上
    bbox1 = dict(boxstyle="round,pad=0.4", fc="#e6f2ff", ec="#1f77b4", lw=1.5)
    ax1.text(times_np[-1]*0.98, np.max(actions_np)*0.9, 
             f'Total Action Penalty:\n{total_action_penalty:.1f}', 
             fontsize=11, fontweight='bold', ha='right', va='top', bbox=bbox1)

    # --- 图 2：力矩震荡惩罚曲线 (直观看到减速器在哪里受伤害最深) ---
    ax2.plot(plot_times, torque_smoothness_cost, color='#d62728', linewidth=1.5)
    ax2.fill_between(plot_times, 0, torque_smoothness_cost, color='#d62728', alpha=0.2)
    
    ax2.set_title('Torque Smoothness Penalty per Step $\sum(\\tau_t - \\tau_{t-1})^2$', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Simulation Time (s)', fontsize=12)
    ax2.set_ylabel('Jitter Penalty', fontsize=11)
    ax2.set_ylim(0, max(np.percentile(torque_smoothness_cost, 99), 1.0) * 1.5) # 过滤极端毛刺调整纵轴
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # 把力矩惩罚的总分贴在第二张图上
    bbox2 = dict(boxstyle="round,pad=0.4", fc="#ffe6e6", ec="#d62728", lw=1.5)
    ax2.text(times_np[-1]*0.98, ax2.get_ylim()[1]*0.9, 
             f'Total Torque Penalty:\n{total_torque_penalty:.1f}', 
             fontsize=11, fontweight='bold', ha='right', va='top', bbox=bbox2)

    # 美化边框
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    img_filename = 'academic_control_smoothness.png'
    plt.savefig(img_filename, dpi=300)
    print(f"[✅ 成功] 平滑度直观图表已保存为: {img_filename}")
    
    plt.show()
else:
    print("[WARN] 数据太少。")