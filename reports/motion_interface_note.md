# Motion Interface Technical Notes

**Tài liệu kỹ thuật**: Quy ước hệ quy chiếu, API điều khiển, và chuỗi waypoints 8-bước cho bài toán Pick & Place.

---

## 1. Quy Ước Hệ Quy Chiếu (Coordinate Frame Convention)

### 1.1 Hệ Quy Chiếu Sử Dụng

**Pipeline sử dụng World Frame (Robot Base Frame)** để toàn bộ dữ liệu được đồng bộ với Person 2.

```
World Frame (Robot Base Frame):
    Z (lên)
    |
    +---- X (phía trước)
   /
  Y (trái)

Gốc World Frame: Nằm tại mặt đất (Z=0). Robot đặt tại X=0.7, Y=-0.2, Z=0.9
```

### 1.2 Frame Definitions

| Frame | ID | Description |
|-------|----|----|
| **Robot Base** | `base_link` | Gốc tại chân robot, Z hướng lên |
| **Left EE (End-Effector)** | `L_sixforce_link` | 6-axis force/torque sensor trên wrist trái |
| **Right EE** | `R_sixforce_link` | 6-axis force/torque sensor trên wrist phải |
| **Torso** | `torso_link` | Mốc chính trên thân robot |
| **TCP Offset Vector** | `[dx, dy, dz]` | Offset từ EE frame (`L_sixforce_link`/`R_sixforce_link`) đến điểm contact thực tế của gripper fingers. Load từ config: `grasp_cfg["tcp_offset"]`. Dùng để IK tính chính xác vị trí grasping khi chạm vật |
### 1.3 Transformation Pipeline

```
Object in World Frame  [x_w, y_w, z_w]
              ↓
        CoordinateTransform
              ↓
Object in Base Frame   [x_base, y_base, z_base]
              ↓
        DualArmIK.solve_ik()
              ↓
    Joint Angles       [q1, q2, ..., q14]
              ↓
    RobotArticulation.execute()
              ↓
    Isaac Sim Actuators
```

**Key Point**: Mọi target pose cho IK solver phải được biểu diễn trong **Robot Base Frame**:
```python
target_pose = [x, y, z, roll, pitch, yaw]  # All in base frame
```

---

## 2. Danh Sách API Điều Khiển (Control API Listing)

### 2.1 High-Level Task API

#### `execute_pick_and_place(obj_pose, bin_pose, side='right')`
**Mục đích**: Thực hiện task nhặt vật từ vị trí `obj_pose` đặt vào bin tại `bin_pose`.

**Input**:
- `obj_pose`: [x, y, z, r, p, y] - Vị trí vật trong World Frame (meters & radians)
- `bin_pose`: [x, y, z, r, p, y] - Vị trí bin trong World Frame
- `side`: 'left' hoặc 'right' - Chọn cánh tay nào

**Returns**: `PrimitiveEvent`
```python
{
    "status": "success" | "failed",
    "step_completed": "grasp" | "lift" | "place" | ...,
    "gripper_state": "open" | "close" | "holding",
    "ee_pose": [x, y, z, r, p, y],  # Current EE pose
    "timestamp": float
}
```

**Example**:
```python
result = execute_pick_and_place(
    obj_pose=[0.3, 0.2, 0.8, 0, 0, 0],
    bin_pose=[0.5, 0.3, 0.9, 0, 0, 0],
    side='right'
)
if result.status == "success":
    print(f"Grasped at step: {result.step_completed}")
```

### 2.2 Mid-Level Arm Control API

#### `robot.move_arm_to_target(side, target_pose, duration=3.0)`
**Mục đích**: Di chuyển cánh tay đến vị trí/hướng mục tiêu sử dụng IK.

**Input**:
- `side`: ArmSide.LEFT | ArmSide.RIGHT | ArmSide.BOTH
- `target_pose`: [x, y, z, r, p, y] trong Robot Base Frame
- `duration`: Thời gian thực hiện (seconds)

**Returns**: Boolean (success/failure)

**Implementation**: 
```python
def move_arm_to_target(self, side, target_pose, duration=3.0):
    # Convert to IKAction
    action = IKAction(side=side, target_pose=target_pose, duration=duration)
    
    # Validate
    valid, msg = ActionValidator.validate_ik_action(action)
    if not valid:
        return False
    
    # Solve IK
    ik_result = self.ik_solver.solve_ik(target_pose, side=side)
    if ik_result is None:
        return False
    
    # Execute
    self.set_joint_positions(ik_result, duration)
    return True
```

#### `robot.move_arm_with_joints(side, positions, duration=2.0)`
**Mục đích**: Di chuyển cánh tay đến vị trí khớp cụ thể (joint-level).

**Input**:
- `side`: ArmSide.LEFT | ArmSide.RIGHT | ArmSide.BOTH
- `positions`: List của 7 hoặc 14 joint angles (radians)
- `duration`: Thời gian thực hiện (seconds)

**Returns**: Boolean (success/failure)

**Implementation**:
```python
def move_arm_with_joints(self, side, positions, duration=2.0):
    action = JointAction(side=side, positions=positions, duration=duration)
    
    valid, msg = ActionValidator.validate_joint_action(action)
    if not valid:
        return False
    
    cmd = RobotCommand(
        command_id=f"joint_move_{self.cmd_counter}",
        timestamp=self.world.current_time,
        arm_action=action
    )
    self.set_joint_positions(positions, duration)
    return True
```

### 2.3 Low-Level Gripper Control API

#### `robot.open_gripper(side, duration=1.0)`
**Mục đích**: Mở gripper.

**Input**:
- `side`: ArmSide.LEFT | ArmSide.RIGHT | ArmSide.BOTH
- `duration`: Thời gian thực hiện (seconds)

**Returns**: Boolean (success/failure)

**Implementation**:
```python
def open_gripper(self, side, duration=1.0):
    action = GripperAction(
        side=side, 
        state=GripperState.OPEN, 
        effort_mode=True,  # Torque-based (robust)
        duration=duration
    )
    return self._execute_gripper_action(action)
```

#### `robot.close_gripper(side, width=0.01, duration=1.0)`
**Mục đích**: Đóng gripper với độ rộng hoặc lực cụ thể.

**Input**:
- `side`: ArmSide.LEFT | ArmSide.RIGHT | ArmSide.BOTH
- `width`: Độ mở gripper tính bằng joint position (radians, default = 0.01 for close)
- `duration`: Thời gian thực hiện (seconds)

**Returns**: Boolean (success/failure)

**Implementation**:
```python
def close_gripper(self, side, width=0.01, duration=1.0):
    action = GripperAction(
        side=side,
        state=GripperState.CLOSE,
        effort_mode=True,  # Torque-based for better grasp
        duration=duration
    )
    return self._execute_gripper_action(action)
```

### 2.4 State Query API

#### `robot.get_ee_pose(side)`
**Mục đích**: Lấy vị trí/hướng hiện tại của end-effector.

**Returns**: [x, y, z, r, p, y] trong Robot Base Frame (hoặc None nếu lỗi)

**Implementation**:
```python
def get_ee_pose(self, side):
    if side == ArmSide.LEFT:
        frame_id = self.ik_solver.left_ee_id
    else:
        frame_id = self.ik_solver.right_ee_id
    
    pin.forwardKinematics(self.ik_solver.model, self.ik_solver.data, self.ik_solver.q)
    se3 = self.ik_solver.data.oMf[frame_id]
    return self.ik_solver.se3_to_xyzrpy(se3)
```

#### `robot.get_gripper_state(side)`
**Mục đích**: Lấy trạng thái gripper hiện tại.

**Returns**: `{'state': 'open' | 'close' | 'holding', 'width': float}`

---

## 3. Chuỗi Waypoints 8-Bước (Pick & Place Sequence)

### 3.1 Tổng Quát

Bài toán Pick & Place được phân thành **8 bước hình học** để đảm bảo an toàn, tránh va chạm, và grasping chắc chắn:

```
[Start]
  ↓ 
[1] Pre-grasp
  ↓
[2] Grasp
  ↓
[3] Lift
  ↓
[4] Move to Bin
  ↓
[5] Pre-place
  ↓
[6] Place
  ↓
[7] Release
  ↓
[8] Retreat
  ↓
[End]
```

### 3.2 Chi Tiết Từng Bước

#### **Step 1: Pre-Grasp** (Chuẩn Bị Grasping)
**Mục đích**: Di chuyển EE gần vật, gripper mở sẵn, chuẩn bị để grasping an toàn.

**Geometry**:
- **Input**: Object pose $T_{obj} = [x, y, z, r, p, y]$
- **Pre-grasp offset**: Dịch theo Z để tránh va chạm: $\Delta z = -0.10m$ (phía trên vật)
- **Pre-grasp target**: $T_{pre} = [x, y, z + 0.10, r, p, y]$
- **Gripper state**: OPEN

**Constraints**:
- EE phải ở trên vật (Z > Z_obj)
- Gripper width max (GRIPPER_OPEN_WIDTH = -0.0215 rad)
- Approach trajectory phải smooth (không có singularities)

**Example**:
```python
obj_pose = [0.3, 0.2, 0.8, 0, 0, 0]
pre_grasp = [obj_pose[0], obj_pose[1], obj_pose[2] + 0.10, 0, 0, 0]

# Move to pre-grasp
success = robot.move_arm_to_target(ArmSide.RIGHT, pre_grasp, duration=2.0)
robot.open_gripper(ArmSide.RIGHT, duration=0.5)
```

---

#### **Step 2: Grasp** (Nhặt Vật)
**Mục đích**: Cập nhật EE pose để tiếp xúc với vật, sau đó đóng gripper.

**Geometry**:
- **Grasp approach**: Di chuyển EE xuống vật một chút $\Delta z = -0.02m$
- **Grasp target**: $T_{grasp} = [x, y, z + 0.02, r, p, y]$
- **Gripper state**: CLOSE

**Constraints**:
- EE phải chạm vật mà không đâm sâu quá (z_offset = 0.02m)
- Gripper phải đủ lực để nắm vật (GRIPPER_CLOSE_TAU = 100 N·m)
- Hãy đợi gripper settle (0.5-1s) trước khi nâng

**Example**:
```python
grasp_target = [obj_pose[0], obj_pose[1], obj_pose[2] + 0.02, 0, 0, 0]

# Move to contact
success = robot.move_arm_to_target(ArmSide.RIGHT, grasp_target, duration=1.0)

# Close gripper with force
success = robot.close_gripper(ArmSide.RIGHT, width=0.01, duration=1.0)
time.sleep(0.5)  # Let gripper settle
```

---

#### **Step 3: Lift** (Nâng Vật)
**Mục đích**: Nâng vật khỏi mặt bàn/conveyor belt, chuẩn bị để di chuyển.

**Geometry**:
- **Lift target**: Tăng Z lên 0.15m so với vật ban đầu
- $T_{lift} = [x, y, z_{obj} + 0.15, r, p, y]$
- **Gripper state**: HOLD (giữ lực grasping)

**Constraints**:
- Phải nâng đủ cao để không va chạm với bàn/conveyor (Z > 0.15m)
- Tốc độ nâng phải chậm (duration = 1.5-2.0s) để tránh động lực quá lớn
- Gripper giữ lực không thay đổi

**Example**:
```python
lift_target = [obj_pose[0], obj_pose[1], obj_pose[2] + 0.15, 0, 0, 0]

# Lift with slow duration
success = robot.move_arm_to_target(ArmSide.RIGHT, lift_target, duration=2.0)
# Gripper stays CLOSE (maintaining grasp)
```

---

#### **Step 4: Move to Bin** (Di Chuyển Đến Bin)
**Mục đích**: Di chuyển vật từ vị trí hiện tại đến bin, giữ nguyên height để tránh va chạm.

**Geometry**:
- **Bin pose**: $T_{bin} = [x_{bin}, y_{bin}, z_{bin}, r, p, y]$
- **Move target**: Giữ nguyên độ cao $z_{lift}$
- $T_{move} = [x_{bin}, y_{bin}, 0.15, r, p, y]$
- **Gripper state**: HOLD

**Constraints**:
- Phải kiểm tra collision với objects khác
- Duración: 3-4 giây (chuyển động chậm, an toàn)
- Không thay đổi gripper state

**Example**:
```python
bin_pose = [0.5, 0.3, 0.9, 0, 0, 0]
move_target = [bin_pose[0], bin_pose[1], obj_pose[2] + 0.15, 0, 0, 0]

# Move to bin location
success = robot.move_arm_to_target(ArmSide.RIGHT, move_target, duration=3.0)
```

---

#### **Step 5: Pre-Place** (Chuẩn Bị Đặt Vật)
**Mục đích**: Hạ EE gần vị trí đặt trong bin, kiểm tra xác nhận, chuẩn bị xả.

**Geometry**:
- **Pre-place target**: Hạ xuống 0.05m so với vị trí cuối cùng
- $T_{pre\_place} = [x_{bin}, y_{bin}, z_{bin} + 0.05, r, p, y]$
- **Gripper state**: HOLD (đặt vật nhẹ nhàng)

**Constraints**:
- EE phải đủ cao để nhìn vào trong bin (vision check)
- Hạ từ từ (duration = 1.5-2.0s)
- Không xả gripper cho đến khi xác nhận vị trí

**Example**:
```python
pre_place_target = [bin_pose[0], bin_pose[1], bin_pose[2] + 0.05, 0, 0, 0]

# Lower carefully to bin
success = robot.move_arm_to_target(ArmSide.RIGHT, pre_place_target, duration=2.0)
```

---

#### **Step 6: Place** (Đặt Vật)**
**Mục đích**: Hạ EE xuống dáy bin, đặt vật xuống.

**Geometry**:
- **Place target**: Hạ xuống dáy bin
- $T_{place} = [x_{bin}, y_{bin}, z_{bin}, r, p, y]$
- **Gripper state**: HOLD (giữ lực cho đến giây cuối)

**Constraints**:
- Phải chạm dáy bin (z = z_bin)
- Hạ từ từ để tránh bounce/quay của vật (duration = 1.0-1.5s)
- Kiểm tra force/torque sensor để phát hiện contact

**Example**:
```python
place_target = [bin_pose[0], bin_pose[1], bin_pose[2], 0, 0, 0]

# Lower to bin bottom
success = robot.move_arm_to_target(ArmSide.RIGHT, place_target, duration=1.5)

# Check F/T sensor
wrench = robot.get_sixforce_reading(ArmSide.RIGHT)
if wrench['force'][2] > threshold:  # Z-direction force
    print("Contact detected - place successful")
```

---

#### **Step 7: Release** (Xả Vật)
**Mục đích**: Mở gripper, thả vật xuống bin hoàn toàn.

**Geometry**:
- **Release**: Giữ nguyên EE pose, chỉ mở gripper
- $T_{release} = T_{place}$ (không thay đổi vị trí)
- **Gripper state**: OPEN

**Constraints**:
- Mở gripper từ từ (duration = 0.5-1.0s) để tránh rơi vật đột ngột
- Phải xác nhận gripper mở (feedback từ joint effort)
- Vật phải nằm yên trong bin trước khi retract

**Example**:
```python
# Open gripper to release
success = robot.open_gripper(ArmSide.RIGHT, duration=1.0)
time.sleep(0.5)  # Let object settle

# Verify gripper open
gripper_state = robot.get_gripper_state(ArmSide.RIGHT)
assert gripper_state['state'] == 'open'
```

---

#### **Step 8: Retreat** (Rút Tay)**
**Mục đích**: Rút EE khỏi bin, trở về safe position, chuẩn bị cho task tiếp theo.

**Geometry**:
- **Retreat target**: Nâng EE lên cao 0.20m so với vị trí ban đầu, dịch sang một bên
- $T_{retreat} = [x_{bin} - 0.10, y_{bin}, z_{bin} + 0.20, 0, 0, 0]$
- **Gripper state**: OPEN (giữ mở)

**Constraints**:
- Phải rút đủ cao để không va chạm với bin edge (Z > 0.20m)
- Dịch sang một bên (dX = -0.10m) để tránh va chạm khi retract
- Tốc độ rút có thể nhanh hơn (duration = 1.5-2.0s)
- Trở về safe position (home) nếu cần task tiếp theo

**Example**:
```python
retreat_target = [bin_pose[0] - 0.10, bin_pose[1], bin_pose[2] + 0.20, 0, 0, 0]

# Retract quickly
success = robot.move_arm_to_target(ArmSide.RIGHT, retreat_target, duration=1.5)

# Optional: Return to home
robot.move_arm_to_target(ArmSide.RIGHT, home_pose, duration=2.0)
```

---

### 3.3 Tóm Tắt Chuỗi Waypoints

| Step | Z Offset | Duration | Gripper | Key Constraint |
|------|----------|----------|---------|-----------------|
| 1. Pre-grasp | +0.10m | 2.0s | OPEN | Avoid collision |
| 2. Grasp | +0.02m | 1.0s | CLOSE | Contact detection |
| 3. Lift | +0.15m | 2.0s | HOLD | Clear table height |
| 4. Move | +0.15m | 3.0s | HOLD | Trajectory safety |
| 5. Pre-place | +0.05m | 2.0s | HOLD | Vision check |
| 6. Place | 0.00m | 1.5s | HOLD | Bottom contact |
| 7. Release | 0.00m | 1.0s | OPEN | Verify open |
| 8. Retreat | +0.20m | 1.5s | OPEN | Clear workspace |

### 3.4 Kiểm Chứng Hình Học

**Geometry Verification**:
1. ✓ Mỗi bước có offset Z rõ ràng (tránh va chạm)
2. ✓ Gripper state được quản lý (mở → đóng → mở)
3. ✓ Duration phù hợp với từng loại movement (slow nâng/hạ, nhanh di chuyển)
4. ✓ Chuỗi từng bước logic (pre → grasp → move → place → retreat)
5. ✓ Sử dụng offset dịch ngang ở step 8 để tránh collision khi retract

---

## Summary

- **Hệ quy chiếu**: Robot Base Frame (đồng bộ với Person 2)
- **API**: Joint level, IK level, Gripper level
- **Waypoints**: 8 bước với geometry rõ ràng, tránh va chạm, grasping an toàn

