"""
extract_camera_params.py
========================
Chạy bằng python.sh của Isaac Sim (headless — không cần mở GUI).

Cách chạy:
    /isaac-sim/python.sh scripts/extract_camera_params.py
    /isaac-sim/python.sh scripts/extract_camera_params.py --scene /path/to/scene.usd

Tìm python.sh:
    find ~ -name "python.sh" 2>/dev/null | grep isaac

Script này:
1. Khởi động Isaac Sim headless
2. Load scene Task 1
3. Đọc camera intrinsics + world transforms
4. Tính T_base_camera
5. In block YAML sẵn sàng copy-paste
6. Lưu ra file output_camera_params.yaml
"""

import argparse
import math
import sys
import os
import numpy as np


# ╔══════════════════════════════════════════════════════════════╗
# ║  SỬA CÁC GIÁ TRỊ NÀY NẾU CẦN                              ║
# ╚══════════════════════════════════════════════════════════════╝
CAMERA_PATH = "/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
ROBOT_BASE_PATH = "/Root/Ref_Xform/Ref/base_link"
WIDTH, HEIGHT = 640, 480

# Đường dẫn scene — override bằng --scene argument
DEFAULT_SCENE_PATH = "/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/assets/resources/Collected_Task4/SubUSDs/2_small_warehouse2.usd"
OUTPUT_FILE = "output_camera_params.yaml"
# ══════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract camera params from Isaac Sim scene"
    )
    parser.add_argument(
        "--scene", default=DEFAULT_SCENE_PATH,
        help="Path to scene USD file"
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help="Output YAML file path"
    )
    parser.add_argument(
        "--camera", default=CAMERA_PATH,
        help="Camera prim path"
    )
    parser.add_argument(
        "--base", default=ROBOT_BASE_PATH,
        help="Robot base prim path"
    )
    # parse_known_args vì Isaac Sim có thể thêm args riêng
    args, _ = parser.parse_known_args()
    return args


# ── Isaac Sim headless startup ────────────────────────────────
def launch_isaac_sim():
    """Khởi động Isaac Sim ở chế độ headless (không cần GPU render)."""
    from isaacsim import SimulationApp

    CONFIG = {
        "headless": True,
        "renderer": "RayTracedLighting",  # hoặc "PathTracing"
    }
    app = SimulationApp(CONFIG)
    return app


# ── USD helpers ───────────────────────────────────────────────
def load_scene(scene_path: str):
    """Mở USD file và trả về stage."""
    import omni.usd
    from pxr import Usd

    # Thử load qua omni.usd context
    result = omni.usd.get_context().open_stage(scene_path)
    if not result:
        print(f"[ERROR] Không load được scene: {scene_path}")
        print("  → Kiểm tra đường dẫn, hoặc dùng --scene /absolute/path/to/scene.usd")
        sys.exit(1)

    stage = omni.usd.get_context().get_stage()
    print(f"[OK] Loaded scene: {scene_path}")
    return stage


def get_world_transform_4x4(stage, prim_path: str) -> np.ndarray | None:
    """Lấy 4x4 world transform matrix (numpy) từ prim path."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    xformable = UsdGeom.Xformable(prim)
    world_mat = xformable.ComputeLocalToWorldTransform(0)
    # Gf.Matrix4d là row-major → transpose để ra numpy column-major convention
    return np.array(world_mat).T


def get_world_pose(stage, prim_path: str):
    """Trả về (position [x,y,z], quaternion_xyzw) của prim trong world frame."""
    from pxr import UsdGeom, Gf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None

    xformable = UsdGeom.Xformable(prim)
    world_mat = xformable.ComputeLocalToWorldTransform(0)
    tf = Gf.Transform(world_mat)

    pos = tf.GetTranslation()
    rot = tf.GetRotation().GetQuat()
    w = rot.GetReal()
    img = rot.GetImaginary()

    position = [float(pos[0]), float(pos[1]), float(pos[2])]
    quat_xyzw = [float(img[0]), float(img[1]), float(img[2]), float(w)]
    return position, quat_xyzw


def list_all_cameras(stage) -> list[str]:
    """Liệt kê tất cả camera prim trong scene."""
    from pxr import UsdGeom
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cameras.append(str(prim.GetPath()))
    return cameras


# ── Main extraction logic ─────────────────────────────────────
def extract_params(stage, camera_path: str, base_path: str,
                   width: int, height: int) -> dict:
    """Core extraction: intrinsics + extrinsics."""
    from pxr import UsdGeom

    result = {}

    # ── 1. List cameras ──
    all_cams = list_all_cameras(stage)
    result["all_cameras"] = all_cams

    # ── 2. Camera intrinsics ──
    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid():
        print(f"\n[ERROR] Camera not found: {camera_path}")
        print("Available cameras:")
        for c in all_cams:
            print(f"  {c}")
        result["intrinsics_ok"] = False
        return result

    focal_length = cam_prim.GetAttribute("focalLength").Get()
    horiz_aperture = cam_prim.GetAttribute("horizontalAperture").Get()
    vert_aperture = cam_prim.GetAttribute("verticalAperture").Get()
    clip_range = cam_prim.GetAttribute("clippingRange").Get()

    fx = (width * focal_length) / horiz_aperture
    fy = (height * focal_length) / vert_aperture
    cx = width / 2.0
    cy = height / 2.0
    fov_h = math.degrees(2 * math.atan(horiz_aperture / (2 * focal_length)))

    result["usd_attrs"] = {
        "focalLength": focal_length,
        "horizontalAperture": horiz_aperture,
        "verticalAperture": vert_aperture,
        "clippingRange": list(clip_range) if clip_range else [0.01, 1000.0],
    }
    result["intrinsics"] = {
        "fx": round(fx, 4),
        "fy": round(fy, 4),
        "cx": cx,
        "cy": cy,
        "fov_h_deg": round(fov_h, 2),
    }
    result["intrinsics_ok"] = True

    # ── 3. Camera world pose ──
    cam_pos, cam_quat = get_world_pose(stage, camera_path)
    result["camera_world_pose"] = {
        "position_m": cam_pos,
        "quaternion_xyzw": cam_quat,
    }

    # ── 4. T_base_camera ──
    T_world_camera = get_world_transform_4x4(stage, camera_path)
    T_world_base = get_world_transform_4x4(stage, base_path)

    if T_world_camera is None:
        print(f"[ERROR] Cannot get camera transform: {camera_path}")
        result["T_base_camera"] = None
        return result

    if T_world_base is None:
        print(f"[ERROR] Cannot get base transform: {base_path}")
        print("Trying to find base prim...")
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            if "base" in name or "robot" in name:
                print(f"  Candidate: {prim.GetPath()}")
        result["T_base_camera"] = None
        return result

    T_base_camera = np.linalg.inv(T_world_base) @ T_world_camera
    det_R = np.linalg.det(T_base_camera[:3, :3])

    result["T_base_camera"] = T_base_camera.tolist()
    result["T_sanity"] = {
        "det_R": round(float(det_R), 6),
        "translation": T_base_camera[:3, 3].tolist(),
        "distance_m": round(float(np.linalg.norm(T_base_camera[:3, 3])), 4),
        "ok": abs(det_R - 1.0) < 1e-3,
    }

    return result


def print_results(result: dict, camera_path: str, base_path: str,
                  width: int, height: int):
    """Print extraction results to console."""
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  HRC2026 — CAMERA PARAMETER EXTRACTOR")
    print(f"  Camera: {camera_path.split('/')[-1]}")
    print(SEP)

    # Cameras
    print(f"\n[1] Cameras trong scene:")
    for c in result.get("all_cameras", []):
        marker = "  <-- DUNG" if c == camera_path else ""
        print(f"    {c}{marker}")

    if not result.get("intrinsics_ok"):
        return

    # USD attrs
    attrs = result["usd_attrs"]
    print(f"\n[2] USD Camera Attributes:")
    print(f"    focalLength        = {attrs['focalLength']}")
    print(f"    horizontalAperture = {attrs['horizontalAperture']}")
    print(f"    verticalAperture   = {attrs['verticalAperture']}")
    print(f"    clippingRange      = {attrs['clippingRange']}")

    # Intrinsics
    intr = result["intrinsics"]
    print(f"\n[3] INTRINSICS:")
    print(f"    fx = {intr['fx']}")
    print(f"    fy = {intr['fy']}")
    print(f"    cx = {intr['cx']}")
    print(f"    cy = {intr['cy']}")
    print(f"    FOV ngang = {intr['fov_h_deg']} deg")

    # Pose
    pose = result.get("camera_world_pose", {})
    print(f"\n[4] Camera world pose:")
    print(f"    position_m:      {pose.get('position_m')}")
    print(f"    quaternion_xyzw: {pose.get('quaternion_xyzw')}")

    # T_base_camera
    T = result.get("T_base_camera")
    sanity = result.get("T_sanity", {})
    print(f"\n[5] T_base_camera:")
    if T:
        for row in T:
            print(f"    - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}]")
        ok = "OK" if sanity.get("ok") else "WARN"
        print(f"    det(R) = {sanity.get('det_R')}  [{ok}]")
        print(f"    translation = {[round(x,4) for x in sanity.get('translation', [])]}")
        print(f"    camera-to-base distance = {sanity.get('distance_m')} m")
        if sanity.get("distance_m", 0) > 3.0:
            print(f"    WARNING: distance > 3m — kiem tra lai scale hoac prim path!")

    # YAML block
    intr = result["intrinsics"]
    T = result.get("T_base_camera", [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    pose = result.get("camera_world_pose", {})
    attrs = result["usd_attrs"]

    print(f"\n{SEP}")
    print("  COPY-PASTE VAO task1_perception.yaml")
    print(SEP)
    print(f"""
camera:
  name: "head_stereo_left"
  prim_path: "{camera_path}"
  rgb_resolution: [{width}, {height}]
  depth_resolution: [{width}, {height}]
  depth_unit: "meter"

intrinsics:
  fx: {intr['fx']}
  fy: {intr['fy']}
  cx: {intr['cx']}
  cy: {intr['cy']}

extrinsics:
  T_base_camera_source: "computed_from_world_transforms"
  robot_base_prim: "{base_path}"
  T_base_camera:""")
    for row in T:
        print(f"    - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}]")

    print(f"\n{SEP}")
    print("  COPY-PASTE VAO camera.yaml")
    print(SEP)
    print(f"""
cameras:
  head_stereo_left:
    prim_path: "{camera_path}"
    rgb:
      width: {width}
      height: {height}
    depth:
      unit: "meter"
      min_range: {attrs['clippingRange'][0]}
      max_range: {attrs['clippingRange'][1]}
    intrinsics:
      fx: {intr['fx']}
      fy: {intr['fy']}
      cx: {intr['cx']}
      cy: {intr['cy']}
    extrinsic:
      position_m: {pose.get('position_m')}
      quaternion_xyzw: {pose.get('quaternion_xyzw')}""")

    print(f"\n{SEP}")


def save_yaml(result: dict, output_path: str, camera_path: str,
              base_path: str, width: int, height: int):
    """Save results to YAML file."""
    import json

    intr = result.get("intrinsics", {})
    T = result.get("T_base_camera", [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    pose = result.get("camera_world_pose", {})
    attrs = result.get("usd_attrs", {})

    lines = [
        "# Auto-generated by extract_camera_params.py",
        "# Copy relevant sections into task1_perception.yaml and camera.yaml",
        "",
        "camera:",
        f'  name: "head_stereo_left"',
        f'  prim_path: "{camera_path}"',
        f"  rgb_resolution: [{width}, {height}]",
        f"  depth_resolution: [{width}, {height}]",
        '  depth_unit: "meter"',
        "",
        "intrinsics:",
        f"  fx: {intr.get('fx')}",
        f"  fy: {intr.get('fy')}",
        f"  cx: {intr.get('cx')}",
        f"  cy: {intr.get('cy')}",
        "",
        "extrinsics:",
        '  T_base_camera_source: "computed_from_world_transforms"',
        f'  robot_base_prim: "{base_path}"',
        "  T_base_camera:",
    ]
    for row in T:
        lines.append(
            f"    - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}]"
        )

    lines += [
        "",
        "# Sanity",
        f"# det(R) = {result.get('T_sanity', {}).get('det_R')}",
        f"# camera_world_position = {pose.get('position_m')}",
        f"# camera_world_quaternion_xyzw = {pose.get('quaternion_xyzw')}",
        "",
        "# All cameras found in scene:",
    ]
    for c in result.get("all_cameras", []):
        lines.append(f"# {c}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[SAVED] {output_path}")


# ── Entry point ───────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 65)
    print("  Isaac Sim Camera Parameter Extractor")
    print("  Dung python.sh cua Isaac Sim")
    print("=" * 65)
    print(f"  Scene: {args.scene}")
    print(f"  Camera: {args.camera}")
    print(f"  Base:   {args.base}")
    print(f"  Res:    {WIDTH}x{HEIGHT}")

    # Launch Isaac Sim headless
    print("\n[1/3] Starting Isaac Sim headless...")
    app = launch_isaac_sim()

    # Import after app starts
    import omni.usd

    # Load scene
    print("[2/3] Loading scene...")
    stage = load_scene(args.scene)

    # Wait a tick for scene to settle
    import omni.kit.app
    app.update()

    # Extract params
    print("[3/3] Extracting camera parameters...")
    result = extract_params(
        stage, args.camera, args.base,
        WIDTH, HEIGHT
    )

    # Print and save
    print_results(result, args.camera, args.base, WIDTH, HEIGHT)
    save_yaml(result, args.output, args.camera, args.base, WIDTH, HEIGHT)

    print(f"\nXong! Ket qua luu tai: {args.output}")

    # Shutdown
    app.close()


if __name__ == "__main__":
    main()
