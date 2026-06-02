"""
capture_rgbd.py
===============
Chạy bằng python.sh của Isaac Sim (headless).

Cách chạy:
    /isaac-sim/python.sh scripts/capture_rgbd.py
    /isaac-sim/python.sh scripts/capture_rgbd.py --scene /path/to/scene.usd --frames 5

Script này:
1. Khởi động Isaac Sim headless
2. Load scene + setup physics
3. Chạy N frames để scene ổn định
4. Capture RGB + depth từ camera
5. Lưu ra: sample_rgb.png, sample_depth.npy, sample_depth_preview.png,
           depth_sanity_report.md
"""

import argparse
import os
import sys
import numpy as np


# ╔══════════════════════════════════════════════════════════════╗
# ║  SỬA NẾU CẦN                                               ║
# ╚══════════════════════════════════════════════════════════════╝
CAMERA_PATH = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
WIDTH, HEIGHT = 640, 480
DEFAULT_SCENE = "/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/assets/resources/Collected_Task4/SubUSDs/2_small_warehouse2.usd"
DEFAULT_OUTPUT = "lab_outputs/perception"
# ══════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(description="Capture RGB+depth from Isaac Sim")
    parser.add_argument("--scene",   default=DEFAULT_SCENE)
    parser.add_argument("--output",  default=DEFAULT_OUTPUT)
    parser.add_argument("--camera",  default=CAMERA_PATH)
    parser.add_argument("--frames",  type=int, default=30,
                        help="Warmup frames before capture (default 30)")
    parser.add_argument("--width",   type=int, default=WIDTH)
    parser.add_argument("--height",  type=int, default=HEIGHT)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    p = lambda name: os.path.join(args.output, name)

    print("=" * 60)
    print("  Isaac Sim RGB-D Capture")
    print(f"  Scene : {args.scene}")
    print(f"  Camera: {args.camera}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    # ── 1. Launch Isaac Sim headless ──
    print("\n[1/5] Starting Isaac Sim headless...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    import omni.usd
    import omni.kit.app
    import omni.replicator.core as rep

    # ── 2. Load scene ──
    print(f"[2/5] Loading scene...")
    result = omni.usd.get_context().open_stage(args.scene)
    if not result:
        print(f"[ERROR] Cannot open: {args.scene}")
        app.close()
        sys.exit(1)
    print(f"      OK: {args.scene}")

    # ── 3. Setup physics + warmup ──
    print(f"[3/5] Running {args.frames} warmup frames...")
    from omni.isaac.core import World
    world = World()
    world.reset()

    for i in range(args.frames):
        world.step(render=True)
        if i % 10 == 0:
            print(f"      frame {i}/{args.frames}")

    # ── 4. Setup annotators ──
    print("[4/5] Setting up annotators...")
    rp = rep.create.render_product(args.camera, (args.width, args.height))

    # RGB
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])

    # Depth — distance_to_image_plane = Z-depth (dùng cho pixel-to-3D)
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annot.attach([rp])

    # Chạy thêm vài frame để annotators warm up
    for _ in range(5):
        world.step(render=True)
        app.update()

    # ── 5. Capture ──
    print("[5/5] Capturing RGB + depth...")
    import omni.syntheticdata

    rgb_data   = rgb_annot.get_data()
    depth_data = depth_annot.get_data()

    print(f"      RGB:   shape={rgb_data.shape}, dtype={rgb_data.dtype}")
    print(f"      Depth: shape={depth_data.shape}, dtype={depth_data.dtype}")

    # ── Save RGB ──
    import cv2

    if rgb_data is None or rgb_data.size == 0:
        print("[ERROR] RGB data is empty!")
    else:
        # Replicator trả về RGBA (H,W,4) uint8
        if rgb_data.ndim == 3 and rgb_data.shape[2] == 4:
            bgr = rgb_data[:, :, :3][:, :, ::-1]   # RGBA → BGR
        elif rgb_data.ndim == 3 and rgb_data.shape[2] == 3:
            bgr = rgb_data[:, :, ::-1]              # RGB → BGR
        else:
            bgr = rgb_data

        cv2.imwrite(p("sample_rgb.png"), bgr)
        print(f"      Saved: sample_rgb.png")

    # ── Save depth ──
    if depth_data is None or depth_data.size == 0:
        print("[ERROR] Depth data is empty!")
    else:
        np.save(p("sample_depth.npy"), depth_data)
        print(f"      Saved: sample_depth.npy")

        # Depth preview (colormap)
        valid = np.isfinite(depth_data) & (depth_data > 0)
        if valid.any():
            d_min, d_max = depth_data[valid].min(), depth_data[valid].max()
            if d_max > d_min:
                normed = np.clip((depth_data - d_min) / (d_max - d_min), 0, 1)
                preview_u8 = (normed * 255).astype(np.uint8)
            else:
                preview_u8 = np.zeros_like(depth_data, dtype=np.uint8)
            preview_u8[~valid] = 0
            preview_color = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
        else:
            preview_color = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        cv2.imwrite(p("sample_depth_preview.png"), preview_color)
        print(f"      Saved: sample_depth_preview.png")

        # ── Depth sanity ──
        print(f"\n{'~'*45}")
        print("  DEPTH SANITY")
        print(f"{'~'*45}")
        print(f"  shape:       {depth_data.shape}")
        print(f"  dtype:       {depth_data.dtype}")
        print(f"  valid_ratio: {valid.mean():.4f}")

        if valid.any():
            vals = depth_data[valid]
            med = float(np.median(vals))
            print(f"  min:         {vals.min():.4f}")
            print(f"  median:      {med:.4f}")
            print(f"  max:         {vals.max():.4f}")
            unit = "meter" if med < 10 else "millimeter"
            print(f"  guessed_unit: {unit}")

            # Write depth_sanity_report.md
            with open(p("depth_sanity_report.md"), "w") as f:
                f.write("# Depth Sanity Report\n\n")
                f.write(f"- **Camera**: {args.camera}\n")
                f.write(f"- **Resolution**: {args.width}x{args.height}\n")
                f.write(f"- **Shape**: {list(depth_data.shape)}\n")
                f.write(f"- **Dtype**: {str(depth_data.dtype)}\n")
                f.write(f"- **Valid ratio**: {valid.mean():.4f}\n")
                f.write(f"- **Min**: {vals.min():.4f}\n")
                f.write(f"- **Median**: {med:.4f}\n")
                f.write(f"- **Max**: {vals.max():.4f}\n")
                f.write(f"- **Guessed unit**: `{unit}`\n\n")
                f.write(f"## Kết luận\n\n")
                f.write(f"Điền `depth_unit: \"{unit}\"` vào YAML.\n")
                if unit == "millimeter":
                    f.write("Chia depth cho 1000 trước khi dùng pixel_to_camera_point.\n")
            print(f"  Saved: depth_sanity_report.md")
        else:
            print("  [WARNING] No valid depth pixels!")
            print("  → Kiểm tra: camera có nhìn thấy scene không?")
            print("  → Thử tăng --frames để scene load hoàn chỉnh hơn")

    print(f"\n{'='*60}")
    print(f"  CHECKLIST")
    print(f"{'='*60}")
    print(f"  [ ] Mở sample_rgb.png  — thấy bàn và 4 workpieces?")
    print(f"  [ ] Mở sample_depth_preview.png — vật thể nổi rõ?")
    print(f"  [ ] valid_ratio > 0.8?")
    print(f"  Output dir: {args.output}")
    print(f"{'='*60}")

    app.close()


if __name__ == "__main__":
    main()
