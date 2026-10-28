cmake_minimum_required(VERSION 3.16)
project(stand_go2)

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)

add_executable(stand_go2 stand_go2.cpp)
target_link_libraries(stand_go2  unitree_sdk2)

add_executable(
    go2_low_level_sim2real
    go2_low_level_sim2real.cpp
)

target_link_libraries(
    go2_low_level_sim2real
    unitree_sdk2
)

add_executable(
    close_motion_service
    close_motion_service.cpp
)

target_link_libraries(
    close_motion_service
    unitree_sdk2
)
SET(CMAKE_BUILD_TYPE Release)
