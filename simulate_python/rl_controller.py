import time
import torch
import numpy as np
import os
from pynput import keyboard

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

class RLController:
    def __init__(self, policy_path):
        print("[状态] 正在建立 DDS 通信链路...")
        ChannelFactoryInitialize(1, "lo")
        
        self.low_state = None
        self.crc = CRC()
        
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.low_state_handler, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        
        # 加载模型
        self.device = torch.device("cpu")
        self.policy = torch.jit.load(policy_path).to(self.device)
        self.policy.eval()
        print("[模型] 策略模型加载成功。")

        # --- 控制参数 ---
        self.action_scale = 0.25 
        self.kp_stand, self.kd_stand = 40.0, 2.0
        self.kp_rl, self.kd_rl = 25.0, 0.5  # 对齐 YAML 中的 25, 0.5
        
        # --- 键盘控制变量 ---
        self.target_vx = 0.0    # W/S
        self.target_vy = 0.0    # A/D
        self.target_dyaw = 0.0  # Q/E
        
        # 默认关节姿态 (SDK 顺序)
        self.default_dof_pos_sdk = np.array([
            -0.1, 0.8, -1.5,  # FR
             0.1, 0.8, -1.5,  # FL
            -0.1, 1.0, -1.5,  # RR
             0.1, 1.0, -1.5   # RL
        ], dtype=np.float32)

        # 🚨 终极映射：适配 YAML 分组 (4 Hip, 4 Thigh, 4 Calf)
        self.unified_sdk_to_isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        self.unified_isaac_to_sdk = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
        self.default_dof_pos_isaac = self.default_dof_pos_sdk[self.unified_sdk_to_isaac]

        self.last_action = np.zeros(12, dtype=np.float32)
        self.low_cmd = self._init_low_cmd()
        self.current_target_pos = self.default_dof_pos_sdk.copy()
        
        # 启动键盘监听器
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        try:
            if key.char == 'w': self.target_vx = 0.5
            elif key.char == 's': self.target_vx = -0.5
            elif key.char == 'a': self.target_vy = 0.4
            elif key.char == 'd': self.target_vy = -0.4
            elif key.char == 'q': self.target_dyaw = 0.8
            elif key.char == 'e': self.target_dyaw = -0.8
        except AttributeError: pass

    def _on_release(self, key):
        try:
            if key.char in ['w', 's']: self.target_vx = 0.0
            if key.char in ['a', 'd']: self.target_vy = 0.0
            if key.char in ['q', 'e']: self.target_dyaw = 0.0
        except AttributeError: pass

    def _init_low_cmd(self):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0], cmd.head[1] = 0xFE, 0xEF
        cmd.level_flag = 0xFF
        for i in range(20): cmd.motor_cmd[i].mode = 0x01
        return cmd

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def get_projected_gravity(self, quat):
        w, x, y, z = quat
        # 重力在机身坐标系的投影公式
        gx = 2.0 * (w * y - x * z)
        gy = -2.0 * (y * z + w * x)
        gz = -(1.0 - 2.0 * (x**2 + y**2))
        return np.array([gx, gy, gz], dtype=np.float32)

    def run(self):
        print("[等待] 正在同步数据流...")
        while self.low_state is None: time.sleep(0.1)
        
        print("[1/2] 正在起立...")
        initial_q = np.array([self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32)
        for t in range(500):
            alpha = t / 500.0
            curr_q = initial_q + alpha * (self.default_dof_pos_sdk - initial_q)
            self._send_pos_cmd(curr_q, self.kp_stand, self.kd_stand)
            time.sleep(0.002)

        print("[2/2] 开始策略控制！请使用 W/A/S/D 控制平移，Q/E 控制旋转。")
        loop_count = 0
        with torch.no_grad():
            while True:
                start_time = time.time()
                
                # 大脑频率: 50Hz (每 10 次小脑循环更新一次)
                if loop_count % 10 == 0:
                    # 获取状态
                    quat = self.low_state.imu_state.quaternion
                    gyro = self.low_state.imu_state.gyroscope # FLU 坐标系
                    q_sdk = np.array([self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32)
                    dq_sdk = np.array([self.low_state.motor_state[i].dq for i in range(12)], dtype=np.float32)

                    # 映射到 Isaac 顺序
                    q_isaac = q_sdk[self.unified_sdk_to_isaac]
                    dq_isaac = dq_sdk[self.unified_sdk_to_isaac]

                    # 组装 Obs (核心修正)
                    obs_np = np.concatenate([
                        # 1. 估计线速度 (暂时给 0.3 的反馈系数防止拖腿)
                        np.array([0.3 * self.target_vx, 0.3 * self.target_vy, 0.0], dtype=np.float32),
                        # 2. 角速度 (直接使用原始值)
                        np.array([gyro[0], gyro[1], gyro[2]], dtype=np.float32),
                        # 3. 重力投影
                        self.get_projected_gravity(quat),
                        # 4. 指令 (Vx, Vy, Yaw_Rate)
                        np.array([self.target_vx, self.target_vy, self.target_dyaw], dtype=np.float32),
                        # 5. 关节位置误差
                        (q_isaac - self.default_dof_pos_isaac),
                        # 6. 关节速度
                        dq_isaac,
                        # 7. 上一次动作
                        self.last_action,
                        np.full(187, -0.3, dtype=np.float32)
                        
                        
                    ]).astype(np.float32)

                    obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device).float()
                    raw_action = self.policy(obs_tensor).squeeze(0).cpu().numpy()
                    
                    # 更新动作
                    self.last_action = raw_action.copy()
                    action_sdk = raw_action[self.unified_isaac_to_sdk]
                    self.current_target_pos = self.default_dof_pos_sdk + (action_sdk * self.action_scale)

                # 小脑频率: 500Hz
                self._send_pos_cmd(self.current_target_pos, self.kp_rl, self.kd_rl)
                
                loop_count += 1
                elapsed = time.time() - start_time
                time.sleep(max(0.0, 0.002 - elapsed))

    def _send_pos_cmd(self, pos, kp, kd):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(pos[i])
            self.low_cmd.motor_cmd[i].kp, self.low_cmd.motor_cmd[i].kd = float(kp), float(kd)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

# if __name__ == '__main__':
#     try:
#         controller = RLController("/home/ubuntu/exported/policy.pt")
#         controller.run()
#     except KeyboardInterrupt:
#         print("\n[退出] 用户手动中断")
if __name__ == '__main__':
    # 使用相对路径，这样代码挪到哪都能跑
    import os
    curr_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_path, "models/policy.pt")
    
    try:
        controller = RLController(model_path)
        controller.run()
    except KeyboardInterrupt:
        print("\n[退出] 用户手动中断")