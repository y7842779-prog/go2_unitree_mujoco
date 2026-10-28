#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <functional>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <unistd.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

// 仅用于真机关闭原有运控服务
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::b2;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

constexpr double PosStopF = 2.146E+9f;
constexpr double VelStopF = 16000.0f;

class Custom
{
public:
    // true：MuJoCo仿真
    // false：真实Go2
    explicit Custom(bool is_simulation)
        : is_simulation_(is_simulation)
    {
    }

    ~Custom() {}

    void Init();

private:
    void InitLowCmd();
    void LowStateMessageHandler(const void *message);
    void LowCmdWrite();

    // 仅真机调用
    bool DisableMotionService();

private:
    bool is_simulation_;

    double stand_up_joint_pos[12] = {
        0.00571868, 0.608813, -1.21763,
        -0.00571868, 0.608813, -1.21763,
        0.00571868, 0.608813, -1.21763,
        -0.00571868, 0.608813, -1.21763};

    double stand_down_joint_pos[12] = {
        0.0473455, 1.22187, -2.44375,
        -0.0473455, 1.22187, -2.44375,
        0.0473455, 1.22187, -2.44375,
        -0.0473455, 1.22187, -2.44375};

    double dt = 0.002;
    double runing_time = 0.0;
    double phase = 0.0;

    int motiontime = 0;
    int rate_count = 0;
    int sin_count = 0;

    double qInit[3] = {0.0, 0.0, 0.0};
    double qDes[3] = {0.0, 0.0, 0.0};
    double sin_mid_q[3] = {0.0, 1.2, -2.0};
    double Kp[3] = {0.0, 0.0, 0.0};
    double Kd[3] = {0.0, 0.0, 0.0};

    unitree_go::msg::dds_::LowCmd_ low_cmd{};
    unitree_go::msg::dds_::LowState_ low_state{};

    ChannelPublisherPtr<
        unitree_go::msg::dds_::LowCmd_>
        lowcmd_publisher;

    ChannelSubscriberPtr<
        unitree_go::msg::dds_::LowState_>
        lowstate_subscriber;

    ThreadPtr lowCmdWriteThreadPtr;

    // SDK封装的运控切换客户端
    MotionSwitcherClient motion_switcher_;
};

uint32_t crc32_core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1U << 31;
        data = ptr[i];

        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }

            xbit >>= 1;
        }
    }

    return CRC32;
}
void Custom::InitLowCmd()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++)
    {
        // 电机进入PMSM伺服模式
        low_cmd.motor_cmd()[i].mode() = 0x01;

        low_cmd.motor_cmd()[i].q() = PosStopF;
        low_cmd.motor_cmd()[i].kp() = 0;

        low_cmd.motor_cmd()[i].dq() = VelStopF;
        low_cmd.motor_cmd()[i].kd() = 0;

        low_cmd.motor_cmd()[i].tau() = 0;
    }
}
/*
 * 关闭真机当前运行的 ai、sport_mode 等运控模式。
 * MuJoCo模式不会调用此函数。
 */
bool Custom::DisableMotionService()
{
    motion_switcher_.SetTimeout(10.0f);
    motion_switcher_.Init();

    while (true)
    {
        std::string robot_form, motion_name;
        int32_t ret = motion_switcher_.CheckMode(robot_form, motion_name);

        if (ret != 0)
        {
            std::cerr << "CheckMode failed: " << ret << std::endl;
            return false;
        }

        if (motion_name.empty())
        {
            std::cout << "Motion service disabled." << std::endl;
            return true;
        }

        ret = motion_switcher_.ReleaseMode();

        if (ret != 0)
        {
            std::cerr << "ReleaseMode failed: " << ret << std::endl;
            return false;
        }

        sleep(5);
    }
}
double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos * (1 - rate) + targetPos * rate;
    return p;
}
void Custom::Init()
{
    InitLowCmd();

    // 创建rt/lowcmd发布器
    lowcmd_publisher.reset(
        new ChannelPublisher<
            unitree_go::msg::dds_::LowCmd_>(
            TOPIC_LOWCMD));

    lowcmd_publisher->InitChannel();

    // 创建rt/lowstate订阅器
    lowstate_subscriber.reset(
        new ChannelSubscriber<
            unitree_go::msg::dds_::LowState_>(
            TOPIC_LOWSTATE));

    lowstate_subscriber->InitChannel(
        std::bind(
            &Custom::LowStateMessageHandler,
            this,
            std::placeholders::_1),
        1);

    /*
     * 仿真模式跳过。
     * 真机模式关闭原有运控服务。
     */
    if (is_simulation_)
    {
        std::cout
            << "[SIM] Skip MotionSwitcherClient."
            << std::endl;
    }
    else
    {
        if (!DisableMotionService())
        {
            std::cerr
                << "[REAL] Failed to disable motion service."
                << std::endl;

            std::exit(EXIT_FAILURE);
        }
    }

    /*loop publishing thread*/
    lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, int(dt * 1000000), &Custom::LowCmdWrite, this);

}


void Custom::LowStateMessageHandler(
    const void *message)
{
    low_state =
        *(unitree_go::msg::dds_::LowState_ *)message;
}


void Custom::LowCmdWrite()
{
    motiontime++;

    if (motiontime >= 0)
    {
        // first, get record initial position
        if (motiontime >= 0 && motiontime < 20)
        {
            qInit[0] = low_state.motor_state()[0].q();
            qInit[1] = low_state.motor_state()[1].q();
            qInit[2] = low_state.motor_state()[2].q();
        }
        // second, move to the origin point of a sine movement with Kp Kd
        if (motiontime >= 10 && motiontime < 400)
        {
            rate_count++;

            double rate = rate_count / 200.0; // needs count to 200
            Kp[0] = 5.0; Kp[1] = 5.0; Kp[2] = 5.0;
            Kd[0] = 1.0; Kd[1] = 1.0; Kd[2] = 1.0;

            qDes[0] = jointLinearInterpolation(qInit[0], sin_mid_q[0], rate);
            qDes[1] = jointLinearInterpolation(qInit[1], sin_mid_q[1], rate);
            qDes[2] = jointLinearInterpolation(qInit[2], sin_mid_q[2], rate);
        }

        double sin_joint1, sin_joint2;
        // last, do sine wave
        float freq_Hz = 1;
        // float freq_Hz = 5;
        float freq_rad = freq_Hz * 2 * M_PI;
        float t = dt * sin_count;

        if (motiontime >= 400)
        {
            sin_count++;
            sin_joint1 = 0.6 * sin(t * freq_rad);
            sin_joint2 = -0.9 * sin(t * freq_rad);
            qDes[0] = sin_mid_q[0];
            qDes[1] = sin_mid_q[1] + sin_joint1;
            qDes[2] = sin_mid_q[2] + sin_joint2;
        }

        for (int i = 0; i < 3; i++)
        {
            low_cmd.motor_cmd()[i].q() = qDes[i];
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = Kp[i];
            low_cmd.motor_cmd()[i].kd() = Kd[i];
            low_cmd.motor_cmd()[i].tau() = 0;
        }
    }

    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    
    lowcmd_publisher->Write(low_cmd);
}
int main(int argc, const char **argv)
{
    // 不带网卡参数：MuJoCo仿真
    // 带网卡参数：真实Go2
    const bool is_simulation = argc < 2;

    if (is_simulation)
    {
        std::cout
            << "MuJoCo simulation mode"
            << std::endl;

        ChannelFactory::Instance()->Init(
            1,
            "lo");
    }
    else
    {
        std::cout
            << "Real Go2 mode, interface: "
            << argv[1]
            << std::endl;

        ChannelFactory::Instance()->Init(
            0,
            argv[1]);
    }

    std::cout
        << "Press Enter to start."
        << std::endl;

    std::cin.get();

    Custom custom(is_simulation);
    custom.Init();

    while (1)
    {
        sleep(10);
    }

    return 0;
}
