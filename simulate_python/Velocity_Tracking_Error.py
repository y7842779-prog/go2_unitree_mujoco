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
# 1. 全局配置与速度跟踪参数
# ==========================================
XML_PATH = "../unitree_robots/go2_Oracle/scene.xml"
POLICY_PATH = "models_oracle/policy.pt"

MAX_SIM_TIME = 12.0   # 测试总时长

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
log_target_vx = []
log_actual_vx = []

print("=========================================")
print(f" 🚀 阶梯式线速度跟踪测试准备就绪！")
print(f"    当前物理负载: {current_xml_mass:.2f} kg (务必确认是空载!)")
print(f"    测试时间: 0 -> {MAX_SIM_TIME}s")
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
            print(f"\n[INFO] 已到达 {MAX_SIM_TIME} 秒，速度跟踪测试结束。")
            break
            
        if data.xpos[base_body_id][2] < 0.18:
            print(f"\n[💥 致命崩溃] 机器狗摔倒了！请检查小球质量是否太重。")
            break

        # 🌟🌟🌟 核心：生成“阶跃变速”目标曲线 (Staircase Profile) 🌟🌟🌟
        if sim_time < 1.5:
            current_target_vx = 0.0
        elif sim_time < 4.0:
            current_target_vx = 0.4  # 起步慢走
        elif sim_time < 6.5:
            current_target_vx = 0.8  # 加速快跑
        elif sim_time < 9.0:
            current_target_vx = 0.3  # 急减速
        else:
            current_target_vx = 0.9  # 再次加速挑战极限

        qpos_isaac = data.qpos[QPOS_ISAAC_IDX]
        qvel_isaac = data.qvel[QVEL_ISAAC_IDX]

        if step_counter % decimation == 0:
            joint_pos_rel = qpos_isaac - DEFAULT_DOF_POS_ISAAC
            base_mat = data.xmat[base_body_id].reshape(3, 3)

            base_lin_vel_world = data.qvel[0:3]
            base_lin_vel = base_mat.T @ base_lin_vel_world 
            base_ang_vel = data.qvel[3:6]
            projected_gravity = base_mat.T @ np.array([0.0, 0.0, -1.0])

            # 避开开机掉落的初始晃动，从 1.0 秒开始记录
            if sim_time >= 1.0:
                log_times.append(sim_time)
                log_target_vx.append(current_target_vx)
                log_actual_vx.append(base_lin_vel[0]) 

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
            
            # 将生成的阶段性速度给到神经网络
            command = np.array([current_target_vx, 0.0, 0.0], dtype=np.float32)

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
# 5. 仿真结束：计算 RMSE 并生成学术绘图
# ==========================================
if len(log_times) > 0:
    print("\n[INFO] 正在计算阶梯追踪误差 (RMSE) 并打包数据...")
    
    times_np = np.array(log_times)
    target_vx_np = np.array(log_target_vx)
    actual_vx_np = np.array(log_actual_vx)
    
    # 计算均方根误差 (RMSE)
    mse = np.mean((target_vx_np - actual_vx_np)**2)
    rmse = np.sqrt(mse)
    
    print(f"=========================================")
    print(f" 🏆 阶梯线速度跟踪测试结果")
    print(f" 🎯 均方根误差 (RMSE) : {rmse:.4f} m/s")
    print(f"=========================================")
    
    # 导出 CSV
    export_data = np.column_stack((times_np, target_vx_np, actual_vx_np))
    csv_filename = f'origin_staircase_tracking_vx.csv'
    np.savetxt(csv_filename, export_data, delimiter=',', header='Time(s),Target_Vx(m/s),Actual_Vx(m/s)', comments='')
    print(f"[✅ 成功] 供 Origin 使用的数据已导出至: {csv_filename}")
    
    # 🌟 生成极其高级的阶梯学术配图 🌟
    plt.figure(figsize=(10, 5))
    
    color_target = '#d62728' # 红色目标线 (更加显眼)
    color_actual = '#1f77b4' # 蓝色实际线
    
    # 目标线用阶梯状线条绘制 (更贴合离散指令)
    plt.plot(times_np, target_vx_np, label='Commanded Velocity ($v_{x, cmd}$)', color=color_target, linestyle='--', linewidth=2.5)
    
    # 实际速度用平滑曲线绘制
    plt.plot(times_np, actual_vx_np, label='Actual Velocity ($v_{x, act}$)', color=color_actual, linewidth=2.0, alpha=0.85)
    
    plt.fill_between(times_np, target_vx_np, actual_vx_np, color='gray', alpha=0.2, label='Tracking Error')
    
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.2, alpha=0.9)
    plt.text(MAX_SIM_TIME*0.98, 0.1, 
             f'Overall RMSE: {rmse:.4f} m/s', 
             fontsize=13, fontweight='bold', ha='right', va='bottom', bbox=bbox_props)

    plt.title('Dynamic Linear Velocity Tracking (Staircase Profile)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Simulation Time (s)', fontsize=12)
    plt.ylabel('Forward Velocity $v_x$ (m/s)', fontsize=12)
    
    plt.xlim(1.0, MAX_SIM_TIME)
    plt.ylim(-0.1, 1.1)  # 严格限制在 0-1 稍微多一点的范围内
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', frameon=True, fontsize=11)
    
    img_filename = 'academic_staircase_tracking_error.png'
    plt.tight_layout()
    plt.savefig(img_filename, dpi=300)
    print(f"[✅ 成功] 速度跟踪误差图已保存为: {img_filename}")
    
    plt.show()
else:
    print("[WARN] 运行时间太短，没有收集到足够的数据。")