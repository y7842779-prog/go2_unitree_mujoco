import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from pynput import keyboard 
import sys
import termios
import atexit
import matplotlib.pyplot as plt

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
# 1. 全局配置、路径与映射钥匙
# ==========================================
XML_PATH = "../unitree_robots/go2_Oracle/scene.xml"
POLICY_PATH = "models_oracle/policy.pt"
TEST_TARGET_VX = 1.0  # 测试目标速度
ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5
MAX_SIM_TIME = 100.0  # 最大测试时长

# 【基准坐标系】物理世界默认按腿分组顺序：FL / FR / RL / RR
DEFAULT_DOF_POS_ISAAC = np.array([
     0.1,  0.8, -1.5,  # FL (hip, thigh, calf)
    -0.1,  0.8, -1.5,  # FR
     0.1,  1.0, -1.5,  # RL
    -0.1,  1.0, -1.5   # RR
], dtype=np.float32)

LEG2ISAACLAB_ORDER = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ISAACLAB2LEG_ORDER = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

# ==========================================
# 2. 初始化环境与模型
# ==========================================
print("[INFO] 加载 MuJoCo 模型...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

isaac_joint_names = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, isaac_joint_names[0]) == -1:
    isaac_joint_names = [name.replace("_joint", "") for name in isaac_joint_names]

QPOS_ISAAC_IDX = []
QVEL_ISAAC_IDX = []
for name in isaac_joint_names:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jnt_id == -1:
        print(f"[ERROR] 找不到关节: {name}")
        exit()
    QPOS_ISAAC_IDX.append(model.jnt_qposadr[jnt_id])
    QVEL_ISAAC_IDX.append(model.jnt_dofadr[jnt_id])

CTRL_MUJOCO_IDX = [
    3, 4, 5,    # FL
    0, 1, 2,    # FR
    9, 10,11,   # RL
    6, 7, 8     # RR
]

print("[INFO] 加载策略...")
try:
    policy = torch.jit.load(POLICY_PATH)
    policy.eval()
except Exception as e:
    print(f"[ERROR] 策略加载失败: {e}")
    exit()

base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_ball")

# 初始化姿势
data.qpos[2] = 0.42  
ball_jnt_id = model.body_jntadr[ball_body_id]
ball_qpos_idx = model.jnt_qposadr[ball_jnt_id]
data.qpos[ball_qpos_idx + 2] = 0.48

for i in range(12):
    data.qpos[QPOS_ISAAC_IDX[i]] = DEFAULT_DOF_POS_ISAAC[i]

mujoco.mj_forward(model, data)

# ==========================================
# 3. 键盘控制后台逻辑
# ==========================================
cmd_vx = 0.0
cmd_vy = 0.0
cmd_yaw = 0.0

def on_press(key):
    global cmd_vx, cmd_vy, cmd_yaw
    try:
        if key.char == 'w': cmd_vx = 0.5
        elif key.char == 's': cmd_vx = -0.3
        elif key.char == 'a': cmd_vy = 0.3
        elif key.char == 'd': cmd_vy = -0.3
        elif key.char == 'q': cmd_yaw = 0.6
        elif key.char == 'e': cmd_yaw = -0.6
    except AttributeError:
        pass

def on_release(key):
    global cmd_vx, cmd_vy, cmd_yaw
    try:
        if key.char in ['w', 's']: cmd_vx = 0.0
        if key.char in ['a', 'd']: cmd_vy = 0.0
        if key.char in ['q', 'e']: cmd_yaw = 0.0
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("=========================================")
print(" 键盘遥控器已就绪！")
print("=========================================")

# ==========================================
# 4. 主仿真循环
# ==========================================
print("[INFO] 启动可视化...")

log_times = []
log_masses = []

# --- 动态注水加重配置 ---
current_test_mass = 1.0       # 初始负载 1.0 kg
max_survived_mass = 1.0       

# 强制初始化物理引擎中的小球质量
model.body_mass[ball_body_id] = current_test_mass
mujoco.mj_setConst(model, data) 
print(f"\n🚀 开始平滑注水极限测试！起步负载: {current_test_mass} kg")

with mujoco.viewer.launch_passive(model, data) as viewer:
    decimation = 10
    step_counter = 0
    actions_isaac = np.zeros(12, dtype=np.float32)

    while viewer.is_running():
        step_start = time.time()
        sim_time = step_counter * model.opt.timestep
        
        # 退出条件 1: 撑过了 100 秒
        if sim_time > MAX_SIM_TIME:
            print(f"[INFO] 居然撑过了 {MAX_SIM_TIME} 秒！机器狗太强了，测试自动结束。")
            break
            
        # 退出条件 2: 摔倒检测 (机身跌破 0.18 米)
        if data.xpos[base_body_id][2] < 0.18:
            print(f"\n[💥 砰！] 机器狗在承载 {current_test_mass:.2f} kg 时被压垮摔倒了！")
            print(f"🏆 测试结论：机器狗在 {TEST_TARGET_VX} m/s 速度下的动态极限稳定负载是 -> {max_survived_mass:.2f} kg")
            break

        qpos_isaac = data.qpos[QPOS_ISAAC_IDX]
        qvel_isaac = data.qvel[QVEL_ISAAC_IDX]

        if step_counter % decimation == 0:
            # ========================================================
            # 🌟 平滑加重逻辑 (注水法)：放在控制周期(50Hz)里执行
            # ========================================================
            if sim_time > 1.5: 
                dt = model.opt.timestep * decimation # 每一步的时间差 (0.02秒)
                max_survived_mass = current_test_mass 
                
                # 真正的平滑加重：每秒增加 0.15kg
                current_test_mass += 0.15 * dt 
                
                # 每隔 2秒 (100个控制步) 刷新一次物理引擎，防止卡顿
                control_step = step_counter // decimation
                if control_step % 100 == 0:
                    model.body_mass[ball_body_id] = current_test_mass
                    mujoco.mj_setConst(model, data)
                    print(f"[{sim_time:.1f}s] 💧 正在注水... 当前负载: {current_test_mass:.2f} kg")
            # ========================================================

            joint_pos_rel = qpos_isaac - DEFAULT_DOF_POS_ISAAC
            base_mat = data.xmat[base_body_id].reshape(3, 3)
            
            # 记录时间与当前的质量，2.5秒后开始记录以避开起步晃动
            if sim_time > 2.5:
                log_times.append(sim_time)
                log_masses.append(current_test_mass)     

            base_lin_vel_world = data.qvel[0:3]
            base_lin_vel = base_mat.T @ base_lin_vel_world
            base_ang_vel = data.qvel[3:6]
            projected_gravity = base_mat.T @ np.array([0.0, 0.0, -1.0])

            base_pos_world = data.xpos[base_body_id]
            ball_pos_world = data.xpos[ball_body_id]

            ball_jnt_id = model.body_jntadr[ball_body_id]
            ball_dof_idx = model.jnt_dofadr[ball_jnt_id]
            ball_vel_world = data.qvel[ball_dof_idx : ball_dof_idx + 3]

            ball_pos_relative = base_mat.T @ (ball_pos_world - base_pos_world)
            ball_vel_relative = base_mat.T @ (ball_vel_world - base_lin_vel_world)
            
            # 【核心感知】把现在的重量传递给 AI 大脑
            ball_mass = np.array([current_test_mass], dtype=np.float32)

            base_pos = data.xpos[base_body_id]
            base_mat = data.xmat[base_body_id].reshape(3, 3)

            scan_range_x = np.linspace(0.4, -0.4, 5) 
            scan_range_y = np.linspace(0.4, -0.4, 5)
            scan_x, scan_y = np.meshgrid(scan_range_x, scan_range_y, indexing='ij')
            
            scan_offsets = np.zeros((25, 3))
            scan_offsets[:, 0] = scan_x.flatten()
            scan_offsets[:, 1] = scan_y.flatten()

            real_heights = []
            NOMINAL_HEIGHT = 0.30 

            for offset in scan_offsets:
                p_start = base_pos + base_mat @ offset + np.array([0, 0, 0.5])
                p_dir = np.array([0, 0, -1.0])
                dist = mujoco.mj_ray(model, data, p_start, p_dir, None, 1, -1, None)
                if dist > 0:
                    ground_z = p_start[2] - dist
                    h = (ground_z - base_pos[2]) + NOMINAL_HEIGHT
                    real_heights.append(h)
                else:
                    real_heights.append(-0.5)
                    
            fake_height_scan = np.array(real_heights, dtype=np.float32)
            
            # 键盘接管逻辑
            if sim_time < 1.5:
                command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                command = np.array([TEST_TARGET_VX + cmd_vx, cmd_vy, cmd_yaw], dtype=np.float32)

            joint_pos_rel_net = joint_pos_rel[LEG2ISAACLAB_ORDER]
            qvel_isaac_net    = qvel_isaac[LEG2ISAACLAB_ORDER]
            actions_isaac_net = actions_isaac[LEG2ISAACLAB_ORDER]

            obs_np = np.concatenate([
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                command,
                joint_pos_rel_net,
                qvel_isaac_net,
                actions_isaac_net,
                fake_height_scan,
                ball_pos_relative,
                ball_vel_relative,
                ball_mass
            ]).astype(np.float32)
            
            assert obs_np.shape[0] == 80, f"观测维度异常: {obs_np.shape[0]}，期待 80"
            obs_tensor = torch.tensor(obs_np).unsqueeze(0)

            with torch.no_grad():
                raw_actions_net = policy(obs_tensor).numpy().squeeze()

            raw_actions_leg = raw_actions_net[ISAACLAB2LEG_ORDER]

            if sim_time < 1.0:
                actions_isaac = np.zeros(12, dtype=np.float32)
            elif sim_time < 1.5:
                alpha = (sim_time - 1.0) / 0.5
                actions_isaac = raw_actions_leg * alpha
            else:
                actions_isaac = raw_actions_leg

        target_dof_pos_isaac = actions_isaac * ACTION_SCALE + DEFAULT_DOF_POS_ISAAC
        tau_isaac = KP * (target_dof_pos_isaac - qpos_isaac) - KD * qvel_isaac
        tau_isaac = np.clip(tau_isaac, -20.0, 20.0)

        data.ctrl[:] = tau_isaac[CTRL_MUJOCO_IDX]

        mujoco.mj_step(model, data)
        step_counter += 1

        if step_counter % decimation == 0:
            viewer.sync()
            elapsed = time.time() - step_start
            step_dt = model.opt.timestep * decimation
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

# ==========================================
# 5. 仿真结束：数据分析与学术绘图
# ==========================================
if len(log_times) > 0:
    print("\n[INFO] 正在生成负载阶梯测试图表...")
    
    masses_np = np.array(log_masses)
    
    # 导出为 CSV 文件
    export_data = np.column_stack((log_times, masses_np))
    csv_filename = f'origin_mass_data_vx_{TEST_TARGET_VX}.csv'
    np.savetxt(csv_filename, export_data, delimiter=',', header='Time(s),Payload_Mass(kg)', comments='')
    print(f"[INFO] 供 Origin 使用的质量数据已导出至: {csv_filename}")
    
    # 论文级学术风绘图 (Publication-Ready Plot)
    plt.figure(figsize=(10, 5))
    
    line_color = '#1f77b4'  # 学术深蓝
    fill_color = '#aec7e8'  # 浅蓝阴影
    crash_color = '#d62728' # 警示红
    
    plt.plot(log_times, masses_np, label='Dynamic Payload Mass (kg)', color=line_color, linewidth=2.5)
    plt.fill_between(log_times, masses_np, 0, color=fill_color, alpha=0.3)
    
    final_time = log_times[-1]
    final_mass = masses_np[-1]
    
    if final_time < (MAX_SIM_TIME - 1.0): 
        plt.axvline(x=final_time, color=crash_color, linestyle='--', linewidth=1.5, alpha=0.8)
        plt.scatter(final_time, final_mass, color=crash_color, s=120, zorder=5, label='Failure Point')
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=crash_color, lw=1.5, alpha=0.9)
        plt.text(final_time - 1.5, final_mass + 0.5, 
                 f'Critical Failure\n@ {final_mass:.2f} kg', 
                 color=crash_color, fontsize=12, fontweight='bold', 
                 ha='right', va='bottom', bbox=bbox_props)

    plt.title(f'Dynamic Payload Capacity Boundary (Target Velocity: {TEST_TARGET_VX} m/s)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Simulation Time (s)', fontsize=12)
    plt.ylabel('Payload Mass applied to Trunk (kg)', fontsize=12)
    
    plt.xlim(0, final_time + 5)
    plt.ylim(0, final_mass + 2.0) 
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout() 
    plt.savefig(f'academic_payload_capacity_vx_{TEST_TARGET_VX}.png', dpi=600) 
    print(f"[INFO] 论文级图表已保存为 academic_payload_capacity_vx_{TEST_TARGET_VX}.png")
    plt.show()
else:
    print("[WARN] 运行时间太短，没有收集到足够的数据。")