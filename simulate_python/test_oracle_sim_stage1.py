import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from pynput import keyboard # <--- [新增 1/3] 导入键盘监听库
import sys
import termios
import atexit
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# 获取当前终端的标准输入文件描述符
fd = sys.stdin.fileno()
# 保存终端当前的设置
old_settings = termios.tcgetattr(fd)

# 定义恢复终端设置的函数
def restore_terminal():
    # === 新增这一行：清空输入缓冲区中残留的按键 ===
    termios.tcflush(fd, termios.TCIFLUSH)
    
    # 恢复终端原来的设置（包括开启回显）
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# 注册到 atexit：无论你是正常退出还是按 Ctrl+C 强退，都能保证终端恢复正常！
atexit.register(restore_terminal)

# 修改终端设置，关闭 ECHO (回显)
new_settings = termios.tcgetattr(fd)
new_settings[3] = new_settings[3] & ~termios.ECHO
termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

# ==========================================
# 1. 全局配置、路径与映射钥匙
# ==========================================
XML_PATH = "../unitree_robots/go2_Oracle/scene_terrain.xml"
POLICY_PATH = "models_oracle/Nominal内八改进版.pt"
STAGE1_BALL_POS_REL = np.array([0.0, 0.0, 0.18], dtype=np.float32)
STAGE1_BALL_MASS = np.array([0.01], dtype=np.float32)

ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5

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

# 初始化姿势
data.qpos[2] = 0.445

for i in range(12):
    data.qpos[QPOS_ISAAC_IDX[i]] = DEFAULT_DOF_POS_ISAAC[i]

mujoco.mj_forward(model, data)


# ==========================================
# 3. 键盘控制后台逻辑 (后台线程)
# ==========================================
# <--- [新增 2/3] 定义键盘回调与启动监听器 --->
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
print(" 提示：请务必确保鼠标点击选中了当前的终端窗口，而不是 MuJoCo 的画面窗口！")
print("   [W] / [S] : 前进 / 后退")
print("   [A] / [D] : 左平移 / 右平移")
print("   [Q] / [E] : 左转 / 右转")
print("=========================================")

# ==========================================
# 4. 主仿真循环
# ==========================================
print("[INFO] 启动可视化...")
log_times = []
log_rolls = []
log_pitches = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    decimation = 10
    step_counter = 0
    actions_isaac = np.zeros(12, dtype=np.float32)

    while viewer.is_running():
        step_start = time.time()
        sim_time = step_counter * model.opt.timestep

        qpos_isaac = data.qpos[QPOS_ISAAC_IDX]
        qvel_isaac = data.qvel[QVEL_ISAAC_IDX]

        if step_counter % decimation == 0:
            joint_pos_rel = qpos_isaac - DEFAULT_DOF_POS_ISAAC
            base_mat = data.xmat[base_body_id].reshape(3, 3)#提取机身的旋转矩阵（代表机身在世界中的倾斜和朝向）
            # --- 新增：提取欧拉角并记录 (仅记录 1.5s 以后的平稳数据) ---
            if sim_time > 1.5:
                # 使用 scipy 将旋转矩阵转为欧拉角 (XYZ 顺序，对应 Roll, Pitch, Yaw)
                r = R.from_matrix(base_mat)
                euler_angles = r.as_euler('xyz', degrees=False) # 返回弧度
                
                log_times.append(sim_time)
                log_rolls.append(euler_angles[0])   # Roll (翻滚角，绕X轴)
                log_pitches.append(euler_angles[1]) # Pitch (俯仰角，绕Y轴)
            # -----------------------------------------------------       

            base_lin_vel_world = data.qvel[0:3]#提取机身在世界坐标系下的绝对速度
            base_lin_vel = base_mat.T @ base_lin_vel_world# 乘以 base_mat.T（旋转矩阵的转置），就是把外界速度转换成狗自己感觉到的速度

            base_ang_vel = data.qvel[3:6]#  机身的三轴旋转角速度 (Roll, Pitch, Yaw)
            projected_gravity = base_mat.T @ np.array([0.0, 0.0, -1.0])

            # 阶段1对应的是“固定在背上的轻球”，不是自由动态球。
            ball_pos_relative = STAGE1_BALL_POS_REL.copy()
            ball_vel_relative = np.zeros(3, dtype=np.float32)
            ball_mass = STAGE1_BALL_MASS.copy()

            # fake_height_scan = np.zeros(25, dtype=np.float32) 
           # 🌟 修正版真实高度雷达
            # ==========================================
            base_pos = data.xpos[base_body_id]
            base_mat = data.xmat[base_body_id].reshape(3, 3)

            # 1. 修复扫描网格方向 (极度关键)
            # 使用 indexing='ij' 匹配 Isaac Lab 的矩阵排列，确保 X 是前后，Y 是左右
            # 范围设为前后左右各 0.4 米（比 0.2 更能看清前方路况）
            scan_range_x = np.linspace(0.4, -0.4, 5) 
            scan_range_y = np.linspace(0.4, -0.4, 5)
            scan_x, scan_y = np.meshgrid(scan_range_x, scan_range_y, indexing='ij')
            
            scan_offsets = np.zeros((25, 3))
            scan_offsets[:, 0] = scan_x.flatten()
            scan_offsets[:, 1] = scan_y.flatten()

            real_heights = []
            # 2. 修复高度基准 (Go2 标称站立高度约为 0.3 米)
            # 加上这个补偿，让平地算出来的探测值为 0，和训练时保持一致
            NOMINAL_HEIGHT = 0.30 

            for offset in scan_offsets:
                p_start = base_pos + base_mat @ offset + np.array([0, 0, 0.5])
                p_dir = np.array([0, 0, -1.0])
                
                dist = mujoco.mj_ray(model, data, p_start, p_dir, None, 1, -1, None)
                
                if dist > 0:
                    ground_z = p_start[2] - dist
                    # 计算相对高度，并补偿站立高度
                    h = (ground_z - base_pos[2]) + NOMINAL_HEIGHT
                    real_heights.append(h)
                else:
                    real_heights.append(-0.5) # 如果射出界外，当做深坑处理
                    
            fake_height_scan = np.array(real_heights, dtype=np.float32)
            # ==========================================
            
            # <--- [新增 3/3] 在这里读取实时的键盘指令 --->
            if sim_time < 1.5:
                # 前 1.5 秒强制发零指令，让狗先站稳
                command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # 1.5 秒后接管键盘实时指令
                command = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float32)

            joint_pos_rel_net = joint_pos_rel[LEG2ISAACLAB_ORDER]
            qvel_isaac_net    = qvel_isaac[LEG2ISAACLAB_ORDER]
            actions_isaac_net = actions_isaac[LEG2ISAACLAB_ORDER]

            obs_np = np.concatenate([
                base_lin_vel,           # 1. 3维
                base_ang_vel,           # 2. 3维
                projected_gravity,      # 3. 3维
                command,                # 4. 3维
                joint_pos_rel_net,      # 5. 12维 (洗牌后)
                qvel_isaac_net,         # 6. 12维 (洗牌后)
                actions_isaac_net,      # 7. 12维 (洗牌后)
                fake_height_scan,       # 8. 25维 (地形)
                ball_pos_relative,      # 9. 3维  (球状态)
                ball_vel_relative,      # 10.3维
                ball_mass               # 11.1维
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

            # （可选）为了终端干净，注释掉了每500步打印动作的代码
            # if step_counter % 500 == 0:
            #     print(f"[{sim_time:.1f}s] 动作输出(FL):", np.round(actions_isaac[:3], 2))

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
# 5. 仿真结束：数据分析与绘图
# ==========================================
if len(log_times) > 0:
    print("\n[INFO] 正在计算躯干姿态方差并生成图表...")
    
    rolls_np = np.array(log_rolls)
    pitches_np = np.array(log_pitches)
    
    # 计算方差 (Variance), 标准差 (Std Dev) 和 最大偏差 (Max Deviation)
    roll_var = np.var(rolls_np)
    pitch_var = np.var(pitches_np)
    roll_std = np.std(rolls_np)
    pitch_std = np.std(pitches_np)
    roll_max = np.max(np.abs(rolls_np))
    pitch_max = np.max(np.abs(pitches_np))
    
    print("================ 姿态抗扰动测试结果 ================")
    print(f"Roll  (翻滚角) -> 方差: {roll_var:.6f} rad^2 | 标差: {roll_std:.4f} rad | 最大偏差: {roll_max:.4f} rad")
    print(f"Pitch (俯仰角) -> 方差: {pitch_var:.6f} rad^2 | 标差: {pitch_std:.4f} rad | 最大偏差: {pitch_max:.4f} rad")
    print("====================================================")
    
    # 绘制对比图
    plt.figure(figsize=(10, 5))
    plt.plot(log_times, rolls_np, label=f'Roll (Var: {roll_var:.4f})', color='blue', alpha=0.8)
    plt.plot(log_times, pitches_np, label=f'Pitch (Var: {pitch_var:.4f})', color='red', alpha=0.8)
    
    # 添加 0 弧度基准线
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    # 设置图表属性
    plt.title('Go2 Base Orientation under Payload Disturbance (Oracle Policy)')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)
    
    # 保存并显示
    plt.savefig('orientation_variance.png', dpi=300)
    print("[INFO] 图表已保存为 orientation_variance.png")
    plt.show()
else:
    print("[WARN] 运行时间太短 (不足1.5秒)，没有收集到足够的评估数据。")