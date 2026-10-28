#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string>
#include <functional>
#include <stdexcept>
#include <unistd.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>

#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

// 真机运控服务切换接口
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
    // true：MuJoCo 仿真
    // false：真实 Go2
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

    // 查询当前是否还有主运控服务运行
    // 返回：
    //  1：存在运动服务
    //  0：运动服务已经关闭
    // -1：查询失败
    int QueryMotionStatus();

    // 把内部模式名称转换成容易理解的服务名称
    std::string QueryServiceName(
        const std::string &form,
        const std::string &name);

private:
    bool is_simulation_ = true;

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

    unitree_go::msg::dds_::LowCmd_ low_cmd{};
    unitree_go::msg::dds_::LowState_ low_state{};

    // 真机运控服务切换客户端
    MotionSwitcherClient msc;

    // LowCmd publisher
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>
        lowcmd_publisher;

    // LowState subscriber
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>
        lowstate_subscriber;

    // 500 Hz LowCmd 发送线程
    ThreadPtr lowCmdWriteThreadPtr;
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

void Custom::Init()
{
    InitLowCmd();

    // 创建 LowCmd publisher
    lowcmd_publisher.reset(
        new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(
            TOPIC_LOWCMD));

    lowcmd_publisher->InitChannel();

    // 创建 LowState subscriber
    lowstate_subscriber.reset(
        new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
            TOPIC_LOWSTATE));

    lowstate_subscriber->InitChannel(
        std::bind(
            &Custom::LowStateMessageHandler,
            this,
            std::placeholders::_1),
        1);

    /*
     * 只有真机模式才关闭 sport_mode / mcf 等运控服务。
     *
     * MuJoCo 仿真没有真实机器狗的 MotionSwitcher 服务，
     * 因此仿真模式必须跳过。
     */
    if (!is_simulation_)
    {
        std::cout
            << "[REAL] Initializing MotionSwitcherClient..."
            << std::endl;

        msc.SetTimeout(10.0f);
        msc.Init();

        int check_failure_count = 0;

        while (true)
        {
            int motion_status = QueryMotionStatus();

            if (motion_status == 0)
            {
                std::cout
                    << "[REAL] Motion control service is disabled."
                    << std::endl;
                break;
            }

            if (motion_status < 0)
            {
                check_failure_count++;

                std::cerr
                    << "[REAL] Failed to query motion mode. Attempt "
                    << check_failure_count << "/3"
                    << std::endl;

                if (check_failure_count >= 3)
                {
                    throw std::runtime_error(
                        "Cannot query the robot motion mode. "
                        "Check the network interface and robot connection.");
                }

                sleep(1);
                continue;
            }

            check_failure_count = 0;

            std::cout
                << "[REAL] Trying to deactivate the motion "
                   "control-related service..."
                << std::endl;

            int32_t ret = msc.ReleaseMode();

            if (ret == 0)
            {
                std::cout
                    << "[REAL] ReleaseMode succeeded."
                    << std::endl;
            }
            else
            {
                std::cerr
                    << "[REAL] ReleaseMode failed. Error code: "
                    << ret
                    << std::endl;
            }

            // 等待机器狗完成服务切换
            sleep(2);
        }
    }
    else
    {
        std::cout
            << "[SIM] Simulation mode: skip MotionSwitcherClient."
            << std::endl;
    }

    /*
     * 只有在真机运控服务关闭完成之后，
     * 才启动 LowCmd 发送线程。
     */
    lowCmdWriteThreadPtr = CreateRecurrentThreadEx(
        "writebasiccmd",
        UT_CPU_ID_NONE,
        static_cast<int>(dt * 1000000),
        &Custom::LowCmdWrite,
        this);

    std::cout
        << "LowCmd thread started at "
        << (1.0 / dt)
        << " Hz."
        << std::endl;
}

void Custom::InitLowCmd()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++)
    {
        // PMSM servo mode
        low_cmd.motor_cmd()[i].mode() = 0x01;

        low_cmd.motor_cmd()[i].q() = PosStopF;
        low_cmd.motor_cmd()[i].kp() = 0;

        low_cmd.motor_cmd()[i].dq() = VelStopF;
        low_cmd.motor_cmd()[i].kd() = 0;

        low_cmd.motor_cmd()[i].tau() = 0;
    }
}

int Custom::QueryMotionStatus()
{
    std::string robot_form;
    std::string motion_name;

    int32_t ret = msc.CheckMode(
        robot_form,
        motion_name);

    if (ret != 0)
    {
        std::cerr
            << "[REAL] CheckMode failed. Error code: "
            << ret
            << std::endl;

        return -1;
    }

    std::cout
        << "[REAL] CheckMode succeeded."
        << std::endl;

    if (motion_name.empty())
    {
        std::cout
            << "[REAL] No active motion control service."
            << std::endl;

        return 0;
    }

    std::string service_name =
        QueryServiceName(robot_form, motion_name);

    if (service_name.empty())
    {
        service_name = motion_name;
    }

    std::cout
        << "[REAL] Active service: "
        << service_name
        << " (mode: "
        << motion_name
        << ")"
        << std::endl;

    return 1;
}

std::string Custom::QueryServiceName(
    const std::string &form,
    const std::string &name)
{
    if (form == "0")
    {
        if (name == "normal")
        {
            return "sport_mode";
        }

        if (name == "ai")
        {
            return "ai_sport";
        }

        if (name == "advanced")
        {
            return "advanced_sport";
        }
    }
    else
    {
        if (name == "ai-w")
        {
            return "wheeled_sport(go2W)";
        }

        if (name == "normal-w")
        {
            return "wheeled_sport(b2W)";
        }
    }

    return "";
}

void Custom::LowStateMessageHandler(const void *message)
{
    low_state =
        *(unitree_go::msg::dds_::LowState_ *)message;
}

void Custom::LowCmdWrite()
{
    runing_time += dt;

    if (runing_time < 3.0)
    {
        // 前 3 秒：从趴下姿态逐渐站起
        phase = tanh(runing_time / 1.2);

        for (int i = 0; i < 12; i++)
        {
            low_cmd.motor_cmd()[i].q() =
                phase * stand_up_joint_pos[i] +
                (1.0 - phase) * stand_down_joint_pos[i];

            low_cmd.motor_cmd()[i].dq() = 0;

            low_cmd.motor_cmd()[i].kp() =
                phase * 50.0 +
                (1.0 - phase) * 20.0;

            low_cmd.motor_cmd()[i].kd() = 3.5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }
    }
    else
    {
        // 3 秒以后：从站立姿态逐渐趴下
        phase = tanh((runing_time - 3.0) / 1.2);

        for (int i = 0; i < 12; i++)
        {
            low_cmd.motor_cmd()[i].q() =
                phase * stand_down_joint_pos[i] +
                (1.0 - phase) * stand_up_joint_pos[i];

            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 50;
            low_cmd.motor_cmd()[i].kd() = 3.5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }
    }

    low_cmd.crc() = crc32_core(
        reinterpret_cast<uint32_t *>(&low_cmd),
        (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);

    lowcmd_publisher->Write(low_cmd);
}

int main(int argc, const char **argv)
{
    bool is_simulation = argc < 2;

    if (is_simulation)
    {
        std::cout
            << "========================================"
            << std::endl;
        std::cout
            << "MuJoCo simulation mode"
            << std::endl;
        std::cout
            << "DDS domain: 1"
            << std::endl;
        std::cout
            << "Interface: lo"
            << std::endl;
        std::cout
            << "========================================"
            << std::endl;

        ChannelFactory::Instance()->Init(1, "lo");
    }
    else
    {
        std::cout
            << "========================================"
            << std::endl;
        std::cout
            << "REAL ROBOT MODE"
            << std::endl;
        std::cout
            << "DDS domain: 0"
            << std::endl;
        std::cout
            << "Interface: "
            << argv[1]
            << std::endl;
        std::cout
            << "========================================"
            << std::endl;

        std::cout
            << "WARNING:"
            << std::endl;
        std::cout
            << "1. Make sure the Go2 is lying down "
               "or securely suspended."
            << std::endl;
        std::cout
            << "2. Keep people away from all four legs."
            << std::endl;
        std::cout
            << "3. ReleaseMode will disable the robot's "
               "built-in balance controller."
            << std::endl;
        std::cout
            << "Press Enter to continue..."
            << std::endl;

        std::cin.get();

        ChannelFactory::Instance()->Init(
            0,
            argv[1]);
    }

    try
    {
        Custom custom(is_simulation);
        custom.Init();

        while (true)
        {
            sleep(10);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Controller initialization failed: "
            << error.what()
            << std::endl;

        return 1;
    }

    return 0;
}