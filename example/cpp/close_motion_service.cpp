#include <iostream>
#include <string>
#include <unistd.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::robot;
using namespace unitree::robot::b2;

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        std::cerr
            << "用法: " << argv[0] << " 网络接口\n"
            << "例如: " << argv[0] << " enp5s0"
            << std::endl;

        return 1;
    }

    const std::string network_interface = argv[1];

    std::cout << "========================================\n";
    std::cout << "仅关闭 Go2 运控服务\n";
    std::cout << "网络接口: " << network_interface << "\n";
    std::cout << "不会发送 LowCmd，不会控制关节\n";
    std::cout << "========================================\n";

    std::cout
        << "警告：必须确保机器狗已经趴下或可靠悬空。\n"
        << "关闭运控服务后，机器狗将失去原来的平衡和姿态控制。\n"
        << "确认安全后按 Enter，取消请按 Ctrl+C。"
        << std::endl;

    std::cin.get();

    // 真机使用 DDS Domain 0
    ChannelFactory::Instance()->Init(
        0,
        network_interface);

    MotionSwitcherClient motion_switcher;

    motion_switcher.SetTimeout(10.0f);
    motion_switcher.Init();

    std::string robot_form;
    std::string motion_name;

    // 查询当前运控模式
    int32_t ret = motion_switcher.CheckMode(
        robot_form,
        motion_name);

    if (ret != 0)
    {
        std::cerr
            << "CheckMode 失败，错误码: "
            << ret
            << "\n请检查网卡、网线和机器狗连接。"
            << std::endl;

        return 1;
    }

    if (motion_name.empty())
    {
        std::cout
            << "当前已经没有正在运行的运控服务，"
            << "不需要重复关闭。"
            << std::endl;

        return 0;
    }

    std::cout
        << "当前运控模式: "
        << motion_name
        << std::endl;

    std::cout
        << "正在调用 ReleaseMode()..."
        << std::endl;

    ret = motion_switcher.ReleaseMode();

    if (ret != 0)
    {
        std::cerr
            << "ReleaseMode 失败，错误码: "
            << ret
            << std::endl;

        return 1;
    }

    std::cout
        << "ReleaseMode 调用成功，正在确认状态..."
        << std::endl;

    sleep(2);

    robot_form.clear();
    motion_name.clear();

    ret = motion_switcher.CheckMode(
        robot_form,
        motion_name);

    if (ret != 0)
    {
        std::cerr
            << "关闭后再次 CheckMode 失败，错误码: "
            << ret
            << std::endl;

        return 1;
    }

    if (!motion_name.empty())
    {
        std::cerr
            << "运控服务可能尚未完全关闭，当前模式仍为: "
            << motion_name
            << std::endl;

        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "运控服务已成功关闭。\n";
    std::cout << "程序现在退出，不会发送任何关节命令。\n";
    std::cout << "请保持机器狗趴下或悬空。\n";
    std::cout << "========================================"
              << std::endl;

    return 0;
}