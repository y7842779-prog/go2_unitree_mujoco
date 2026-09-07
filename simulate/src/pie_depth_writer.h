#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

#include <mujoco/mujoco.h>

#include "param.h"

namespace pie_depth
{

class PieDepthWriter
{
public:
  void Update(const mjModel *model, mjData *data)
  {
    if (param::config.enable_pie_depth == 0 || model == nullptr || data == nullptr)
    {
      return;
    }

    if (model != model_)
    {
      Reset(model);
    }
    if (camera_id_ < 0)
    {
      return;
    }

    const int update_steps = std::max(1, param::config.pie_depth_update_steps);
    if (++step_counter_ % update_steps != 0)
    {
      return;
    }

    RenderRawRayDepth(model, data);
    ProcessDepth();
    PublishDisplayDepth();
    WriteDepthFile();
  }

  bool CopyLatestDepth(std::vector<float> *depth, int *width, int *height) const
  {
    if (depth == nullptr || width == nullptr || height == nullptr)
    {
      return false;
    }

    std::lock_guard<std::mutex> lock(display_mutex_);
    if (!has_display_depth_)
    {
      return false;
    }

    *depth = display_depth_;
    *width = kOutputWidth;
    *height = kOutputHeight;
    return true;
  }

private:
  static constexpr const char *kCameraName = "pie_depth_camera";
  static constexpr int kRawWidth = 106;
  static constexpr int kRawHeight = 60;
  static constexpr int kCropLeft = 10;
  static constexpr int kCropRight = 10;
  static constexpr int kOutputWidth = kRawWidth - kCropLeft - kCropRight;
  static constexpr int kOutputHeight = kRawHeight;
  static constexpr double kPi = 3.14159265358979323846;
  static constexpr double kHorizontalFovDeg = 87.0;
  static constexpr float kMinDistance = 0.05f;
  static constexpr float kMaxDistance = 3.0f;
  static constexpr std::array<float, 9> kGaussianKernel = {
      0.07511361f, 0.12384140f, 0.07511361f,
      0.12384140f, 0.20417996f, 0.12384140f,
      0.07511361f, 0.12384140f, 0.07511361f};

  void Reset(const mjModel *model)
  {
    model_ = model;
    camera_id_ = mj_name2id(model, mjOBJ_CAMERA, kCameraName);
    base_body_id_ = mj_name2id(model, mjOBJ_BODY, "base_link");
    step_counter_ = 0;
    raw_depth_.assign(kRawWidth * kRawHeight, kMaxDistance);
    processed_depth_.assign(kOutputWidth * kOutputHeight, 1.0f);
    warned_write_failure_ = false;
    {
      std::lock_guard<std::mutex> lock(display_mutex_);
      display_depth_.clear();
      has_display_depth_ = false;
    }

    if (camera_id_ < 0)
    {
      std::cerr << "PIE depth camera '" << kCameraName << "' not found; PIE depth output disabled for this model.\n";
    }
    else if (base_body_id_ < 0)
    {
      std::cerr << "Go2 base body 'base_link' not found; PIE depth output disabled for this model.\n";
      camera_id_ = -1;
    }
    else
    {
      std::cout << "PIE depth output enabled: " << param::config.pie_depth_binary_path
                << " [" << kOutputHeight << "x" << kOutputWidth << ", raw "
                << kRawHeight << "x" << kRawWidth << "]\n";
    }
  }

  void RenderRawRayDepth(const mjModel *model, mjData *data)
  {
    const mjtNum *cam_pos = data->cam_xpos + 3 * camera_id_;
    const mjtNum *cam_mat = data->cam_xmat + 9 * camera_id_;

    const double focal_pixels = 0.5 * kRawWidth /
                                std::tan(0.5 * kHorizontalFovDeg * kPi / 180.0);
    const double cx = 0.5 * kRawWidth;
    const double cy = 0.5 * kRawHeight;
    mjtByte geomgroup[6] = {1, 1, 1, 0, 0, 0};

    for (int y = 0; y < kRawHeight; ++y)
    {
      for (int x = 0; x < kRawWidth; ++x)
      {
        const mjtNum pixel_x = static_cast<mjtNum>((x + 0.5 - cx) / focal_pixels);
        const mjtNum pixel_y = static_cast<mjtNum>((y + 0.5 - cy) / focal_pixels);
        const mjtNum local_dir[3] = {pixel_x, -pixel_y, -1.0};
        mjtNum world_dir[3];
        mju_mulMatVec3(world_dir, cam_mat, local_dir);
        mju_normalize3(world_dir);

        int geom_id = -1;
        const mjtNum distance = mj_ray(
            model,
            data,
            cam_pos,
            world_dir,
            geomgroup,
            1,
            base_body_id_,
            &geom_id);

        float depth = kMaxDistance;
        if (geom_id >= 0 && distance > 0.0)
        {
          depth = static_cast<float>(std::min<mjtNum>(distance, kMaxDistance));
        }
        raw_depth_[y * kRawWidth + x] = depth;
      }
    }
  }

  static int ReflectIndex(int index, int size)
  {
    if (index < 0)
    {
      return -index;
    }
    if (index >= size)
    {
      return 2 * size - index - 2;
    }
    return index;
  }

  float CroppedDepthAt(int y, int x) const
  {
    const int reflected_y = ReflectIndex(y, kOutputHeight);
    const int reflected_x = ReflectIndex(x, kOutputWidth);
    return raw_depth_[reflected_y * kRawWidth + reflected_x + kCropLeft];
  }

  void ProcessDepth()
  {
    for (int y = 0; y < kOutputHeight; ++y)
    {
      for (int x = 0; x < kOutputWidth; ++x)
      {
        float blurred = 0.0f;
        int kernel_index = 0;
        for (int ky = -1; ky <= 1; ++ky)
        {
          for (int kx = -1; kx <= 1; ++kx)
          {
            blurred += CroppedDepthAt(y + ky, x + kx) * kGaussianKernel[kernel_index++];
          }
        }
        const float clipped = std::clamp(blurred, kMinDistance, kMaxDistance);
        processed_depth_[y * kOutputWidth + x] = clipped / kMaxDistance;
      }
    }
  }

  void WriteDepthFile()
  {
    const std::filesystem::path output_path(param::config.pie_depth_binary_path);
    const auto tmp_path = output_path.string() + ".tmp";

    std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
    {
      WarnWriteFailure("failed to open temp PIE depth file");
      return;
    }
    file.write(
        reinterpret_cast<const char *>(processed_depth_.data()),
        static_cast<std::streamsize>(processed_depth_.size() * sizeof(float)));
    file.close();

    std::error_code ec;
    std::filesystem::rename(tmp_path, output_path, ec);
    if (ec)
    {
      WarnWriteFailure("failed to publish PIE depth file: " + ec.message());
    }
  }

  void PublishDisplayDepth()
  {
    std::lock_guard<std::mutex> lock(display_mutex_);
    display_depth_ = processed_depth_;
    has_display_depth_ = true;
  }

  void WarnWriteFailure(const std::string &message)
  {
    if (!warned_write_failure_)
    {
      std::cerr << "PIE depth output: " << message << "\n";
      warned_write_failure_ = true;
    }
  }

  const mjModel *model_ = nullptr;
  int camera_id_ = -1;
  int base_body_id_ = -1;
  int step_counter_ = 0;
  bool warned_write_failure_ = false;
  std::vector<float> raw_depth_;
  std::vector<float> processed_depth_;
  mutable std::mutex display_mutex_;
  std::vector<float> display_depth_;
  bool has_display_depth_ = false;
};

} // namespace pie_depth
