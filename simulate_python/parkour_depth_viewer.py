#!/usr/bin/env python3
"""Preview the Parkour depth tensor written by unitree_mujoco.

The controller consumes a normalized float32 tensor with shape [87, 58].
This helper shows the same tensor as a colorized depth image.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def read_depth(path: Path, height: int, width: int) -> np.ndarray | None:
    expected_bytes = height * width * np.dtype(np.float32).itemsize
    try:
        if path.stat().st_size != expected_bytes:
            return None
        data = np.fromfile(path, dtype=np.float32)
    except FileNotFoundError:
        return None
    if data.size != height * width:
        return None
    return data.reshape(height, width)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/tmp/parkour_depth_image_float32.bin")
    parser.add_argument("--height", type=int, default=87)
    parser.add_argument("--width", type=int, default=58)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--save-path", default="/tmp/parkour_depth_preview.png")
    args = parser.parse_args()

    path = Path(args.path)
    period = 1.0 / max(args.fps, 1.0)

    try:
        import cv2

        use_cv2 = True
    except Exception:
        cv2 = None
        use_cv2 = False

    print(f"Watching {path}")
    print("Depth tensor is normalized: meters = (value + 0.5) * max_depth")
    if not use_cv2:
        print("OpenCV is unavailable; saving preview PNG snapshots instead.")

    highgui_available = use_cv2
    frame_id = 0
    while True:
        depth_norm = read_depth(path, args.height, args.width)
        if depth_norm is None:
            time.sleep(period)
            continue

        depth_m = np.clip(depth_norm + 0.5, 0.0, 1.0) * args.max_depth
        image_u8 = ((1.0 - np.clip(depth_m / args.max_depth, 0.0, 1.0)) * 255.0).astype(np.uint8)

        if use_cv2:
            color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
            color = cv2.resize(
                color,
                (args.width * args.scale, args.height * args.scale),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(
                color,
                f"min {depth_m.min():.2f}m  max {depth_m.max():.2f}m",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if highgui_available:
                try:
                    cv2.imshow("parkour_depth_camera", color)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                except cv2.error:
                    highgui_available = False
                    print(f"OpenCV HighGUI is unavailable; writing {args.save_path} instead.")

            if not highgui_available and frame_id % max(1, int(args.fps)) == 0:
                cv2.imwrite(args.save_path, color)
        else:
            if frame_id % max(1, int(args.fps)) == 0:
                np.savetxt(args.save_path + ".txt", depth_m, fmt="%.3f")

        frame_id += 1
        time.sleep(period)

    if use_cv2:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
