import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from pynput import keyboard 
import sys
import termios
import atexit
import matplotlib.pyplot as plt  # <--- [恢复] 导入绘图库

# ==========================================
# 终端设置清理 (防止按键残留)
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
# 1. 全局配置与冲击测试参数 (🌟 核心配置区 🌟)
# ==========================================
XML_PATH = "../unitree_robots/go2_Oracle/scene.xml"
POLICY_PATH = "models_oracle/policy.pt"

TEST_TARGET_VX = 1.0  # 目标前进速度 (m/s)
MAX_SIM_TIME = 15.0   # 跑 15 秒就够了

# 👇👇👇 抗冲击测试参数 👇👇👇
IMPACT_TIME = 3.0       # 在第 3.0 秒时施加冲击
IMPACT_DURATION = 0.1   # 冲击持续时间 (秒)
IMPACT_FORCE_Y = 20.0   # 侧向冲击力大小 (N) - 先用 20N 测
SPEED_TOLERANCE = 0.15  # 速度恢复容忍误差 (m/s)
# 👆👆👆👆👆👆👆👆👆👆👆👆

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
# 3. 数据记录器初始化
# ==========================================
log_times = []
log_vx = []
log_vy = []

print("=========================================")
print(f" 🚀 抗侧向冲击测试准备就绪！")
print(f"    目标速度: {TEST_TARGET_VX} m/s")
print(f"    小球质量: {current_xml_mass:.2f} kg (来自XML)")
print(f"    冲击设定: 在 {IMPACT_TIME}s 施加 {IMPACT_FORCE_Y}N，持续 {IMPACT_DURATION}s")
print("=========================================")

# ==========================================
# 4. 主仿真循环
# ==========================================
impact_applied = False
recovered = False
is_crashed = False 
stable_steps = 0
recovery_duration = -1.0
actual_recovery_time = -1.0 # 记录最终恢复的时间点，绘图要用

with mujoco.viewer.launch_passive(model, data) as viewer:
    decimation = 10
    step_counter = 0
    actions_isaac = np.zeros(12, dtype=np.float32)

    while viewer.is_running():
        step_start = time.time()
        sim_time = step_counter * model.opt.timestep
        
        if sim_time > MAX_SIM_TIME:
            print(f"\n[INFO] 已到达 {MAX_SIM_TIME} 秒，测试结束。")
            break
            
        if data.xpos[base_body_id][2] < 0.18:
            print(f"\n[💥 致命崩溃] 机器狗未能扛住冲击，在 {sim_time:.2f} 秒时倒地！")
            is_crashed = True
            break

        # 施加瞬间侧向力
        if IMPACT_TIME <= sim_time <= (IMPACT_TIME + IMPACT_DURATION):
            data.xfrc_applied[base_body_id][:3] = [0.0, IMPACT_FORCE_Y, 0.0]
            if not impact_applied:
                print(f"\n[{sim_time:.2f}s] 💥 遭受侧向 {IMPACT_FORCE_Y}N 脉冲冲击！狗发生偏航！")
                impact_applied = True
        else:
            data.xfrc_applied[base_body_id][:] = 0.0

        qpos_isaac = data.qpos[QPOS_ISAAC_IDX]
        qvel_isaac = data.qvel[QVEL_ISAAC_IDX]

        if step_counter % decimation == 0:
            joint_pos_rel = qpos_isaac - DEFAULT_DOF_POS_ISAAC
            base_mat = data.xmat[base_body_id].reshape(3, 3)

            base_lin_vel_world = data.qvel[0:3]
            base_lin_vel = base_mat.T @ base_lin_vel_world 
            base_ang_vel = data.qvel[3:6]
            projected_gravity = base_mat.T @ np.array([0.0, 0.0, -1.0])

            # 记录数据供 Origin 绘图 (避开起步的 1 秒)
            if sim_time > 1.0:
                log_times.append(sim_time)
                log_vx.append(base_lin_vel[0]) 
                log_vy.append(base_lin_vel[1]) 

            # 检测速度恢复时间
            if impact_applied and sim_time > (IMPACT_TIME + IMPACT_DURATION) and not recovered:
                error_vx = abs(base_lin_vel[0] - TEST_TARGET_VX)
                error_vy = abs(base_lin_vel[1] - 0.0)
                
                if error_vx < SPEED_TOLERANCE and error_vy < SPEED_TOLERANCE:
                    stable_steps += 1
                else:
                    stable_steps = 0 
                
                if stable_steps >= 25:
                    recovered = True
                    actual_recovery_time = sim_time - (25 * (model.opt.timestep * decimation))
                    recovery_duration = actual_recovery_time - (IMPACT_TIME + IMPACT_DURATION)
                    print(f"[{actual_recovery_time:.2f}s] ✅ 机器狗已重新收敛至稳定步态！")
                    print(f"🏆 抗冲击恢复耗时 (Recovery Time): {recovery_duration:.3f} 秒")

            ball_pos_world = data.xpos[ball_body_id]
            ball_dof_idx = model.jnt_dofadr[ball_jnt_id]
            ball_vel_world = data.qvel[ball_dof_idx : ball_dof_idx + 3]

            ball_pos_relative = base_mat.T @ (ball_pos_world - data.xpos[base_body_id])
            ball_vel_relative = base_mat.T @ (ball_vel_world - base_lin_vel_world)
            ball_mass = np.array([current_xml_mass], dtype=np.float32)

            # 高度雷达
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

            if sim_time < 1.0: actions_isaac = np.zeros(12, dtype=np.float32)
            elif sim_time < 1.5: actions_isaac = raw_actions_leg * ((sim_time - 1.0) / 0.5)
            else: actions_isaac = raw_actions_leg

        tau_isaac = np.clip(KP * (actions_isaac * ACTION_SCALE + DEFAULT_DOF_POS_ISAAC - qpos_isaac) - KD * qvel_isaac, -20.0, 20.0)
        data.ctrl[:] = tau_isaac[CTRL_MUJOCO_IDX]

        mujoco.mj_step(model, data)
        step_counter += 1

        if step_counter % decimation == 0:
            viewer.sync()
            elapsed = time.time() - step_start
            step_dt = model.opt.timestep * decimation
            if elapsed < step_dt: time.sleep(step_dt - elapsed)

# ==========================================
# 5. 仿真结束：数据导出与学术绘图
# ==========================================
if len(log_times) > 0:
    print("\n[INFO] 正在打包速度恢复曲线数据...")
    export_data = np.column_stack((log_times, log_vx, log_vy))
    csv_filename = f'origin_impact_vx{TEST_TARGET_VX}_{int(IMPACT_FORCE_Y)}N.csv'
    
    # 1. 导出 CSV 给 Origin
    np.savetxt(csv_filename, export_data, delimiter=',', header='Time(s),Vx_Forward(m/s),Vy_Lateral(m/s)', comments='')
    print(f"[✅ 成功] 供 Origin 使用的冲击测试数据已导出至: {csv_filename}")
    
    if not recovered and impact_applied and not is_crashed:
        print("[WARN] 机器狗扛住了冲击没有摔倒，但直到测试结束也未能完全收敛回目标速度。")

    # 2. 生成学术级论文配图 (Matplotlib)
    print("[INFO] 正在绘制并保存学术级速度恢复曲线图...")
    plt.figure(figsize=(10, 6))
    
    # 配色方案 (学术常用深蓝与橙黄)
    color_vx = '#1f77b4' 
    color_vy = '#ff7f0e'
    color_impact = '#d62728' # 红色警示
    color_recover = '#2ca02c' # 绿色恢复
    
    # 画出实际速度曲线
    plt.plot(log_times, log_vx, label='Forward Velocity ($v_x$)', color=color_vx, linewidth=2)
    plt.plot(log_times, log_vy, label='Lateral Velocity ($v_y$)', color=color_vy, linewidth=2)
    
    # 画出目标速度基准线 (虚线)
    plt.axhline(y=TEST_TARGET_VX, color=color_vx, linestyle='--', alpha=0.5, label='Target $v_x$')
    plt.axhline(y=0.0, color=color_vy, linestyle='--', alpha=0.5, label='Target $v_y$')
    
    # 标注受击瞬间
    if impact_applied:
        plt.axvline(x=IMPACT_TIME, color=color_impact, linestyle='-.', linewidth=2, label='Impact Applied')
        
        # 如果成功恢复，标注恢复区间
        if recovered:
            plt.axvline(x=actual_recovery_time, color=color_recover, linestyle='-.', linewidth=2, label='Fully Recovered')
            # 使用阴影区域标出恢复所需的时间段
            plt.axvspan(IMPACT_TIME + IMPACT_DURATION, actual_recovery_time, color='gray', alpha=0.15, label=f'Recovery Time: {recovery_duration:.2f}s')

    # 图表细节修饰
    plt.title(f'Robustness against Lateral Impact ({IMPACT_FORCE_Y}N Pulse)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Simulation Time (s)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    
    # 设置合理的速度显示范围（自适应）
    plt.ylim(min(min(log_vx), min(log_vy)) - 0.5, max(max(log_vx), max(log_vy)) + 0.5)
    
    # 学术风边框清理
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    
    # 自动保存高分辨率图片并展示
    img_filename = f'academic_impact_recovery_vx{TEST_TARGET_VX}_{int(IMPACT_FORCE_Y)}N.png'
    plt.tight_layout()
    plt.savefig(img_filename, dpi=300)
    print(f"[✅ 成功] 速度恢复曲线图已保存为: {img_filename}")
    
    plt.show()