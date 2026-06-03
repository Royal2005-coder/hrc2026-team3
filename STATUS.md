# Tình hình dự án HRC2026 Team3

> Cập nhật lần cuối: 2026-06-03 | Branch: `bao/planner_support`

---

## 1. Tổng quan

**Task**: Pick-and-place — robot humanoid S2 dual-arm gắp 4 vật (PartA/PartB) trải ngẫu nhiên trên bàn rồi đặt vào thùng.

**Môi trường**: Isaac Sim (Omniverse), chạy trong Docker container.

**Có 2 chế độ chạy độc lập**:

| Chế độ | Entry point | Mô tả |
|--------|-------------|-------|
| Planner | `src/baseline_source/main_fixed.py` | IK + FSM truyền thống |
| IL (Imitation Learning) | `run_il_policy.py` | LSTM model học từ demos |

---

## 2. Cấu trúc code quan trọng

```
src/
  baseline_source/
    main_fixed.py          ← entry point planner mode
    SceneBuilder.py        ← build scene, scatter vật ngẫu nhiên
    RobotArticulation.py   ← điều khiển robot, IK solver (Pinocchio)
    grasp_planner.py       ← tính grasp target (chỉ dùng bởi main.py cũ)
    coordinate_utils.py    ← chuyển đổi world ↔ robot base frame
  task1/
    motion.py              ← FSM pick-place (S1_PREGRASP → S7_RETREAT)
    primitive_spec_task1.yaml
robot_arm_training/
  task1/
    inference_isaac.py     ← ILPolicyRunner, state vector 38-dim
    train.py / config.py   ← training setup
    checkpoints/lstm/best.pt
run_il_policy.py           ← entry point IL mode
configs/Part_Sorting.yaml  ← cấu hình scene, robot, scatter area
data/task1/part_sorting_sample.csv   ← dữ liệu training
```

---

## 3. Các bug đã fix (đã commit + push)

### Fix 1 — `grasp_planner.py` · commit `e1d8221`

**Bug**: `current_world += 0.15` chạy trước khi check `None` → crash khi prim không tồn tại; đồng thời cộng 0.15 vào cả 3 trục x,y,z thay vì chỉ z.

```python
# Trước (sai)
current_world += 0.15
if current_world is None:
    return

# Sau (đúng)
if current_world is None:
    return
current_world[2] += 0.15  # chỉ offset Z
```

---

### Fix 2 — `src/task1/motion.py` · commit `e1d8221`

**Bug**: `reset_robot_state_full()` teleport robot về tư thế ban đầu trước **mỗi** plan → cánh tay bị giật về vị trí gốc giữa các lần gắp thay vì chuyển động liên tục.

**Fix**: Chỉ teleport ở plan đầu tiên hoặc khi retry sau lỗi. Các plan tiếp theo chỉ sync IK nhẹ (không teleport khớp).

```python
# Trước: reset_robot_state_full() gọi EVERY plan
# Sau:
if is_first_plan or is_recovering:
    reset_robot_state_full(robot, world, verbose=True)
else:
    # chỉ sync IK solver, không teleport khớp
    robot.ik_solver.sync_joint_positions(joints['names'], positions)
```

---

### Fix 3 — `run_il_policy.py` · commit `91f8e38`

**Bug**: Khi LSTM hội tụ (MAE → ~0.0002 rad) với gripper đã đóng nhưng không gắp được gì, arm đứng yên đến hết 500 steps — lãng phí thời gian sim.

**Fix**: Stall detection — nếu MAE < 0.002 rad liên tục 60 steps trong khi gripper=C → break episode sớm.

---

## 4. Vấn đề còn tồn tại

### 4.1 IL mode không gắp được vật ← VẤN ĐỀ CHÍNH

**Triệu chứng trong log**:
```
step=  60 | right arm MAE=0.0039 rad | right gripper=C   ← đến pose cố định, đóng gripper
step= 120 | right arm MAE=0.0006 rad | right gripper=C   ← đứng yên, không gắp được gì
step= 480 | right arm MAE=0.0002 rad | right gripper=C   ← vẫn đứng yên
```

**Nguyên nhân**: Dữ liệu training quá ít.

| Chỉ số | Giá trị | Nhận xét |
|--------|---------|----------|
| Training episodes | 91 | Cần 300–500+ |
| Train loss / Val loss | 0.017 / 0.042 | Overfitting nhẹ (gap 2.5×) |
| Checkpoint epoch (local) | 59 | Config `NUM_EPOCHS=300`, chưa train đủ |
| Checkpoint epoch (container) | 286 | Khác với local — cần đồng bộ |

**Cơ chế lỗi**: Model học được shape của trajectory (chuỗi góc khớp) nhưng **không học được mapping vị trí vật → vị trí tay cần đến**. Arm di chuyển về một pose cố định từ training, đóng gripper ở đó, bất kể vật đang ở đâu.

Về lý thuyết, scatter area `(x=[0.45,0.90], y=[0.05,0.40])` nằm trong phạm vi training data `(obj0_x=[0.27,1.23])` — không phải out-of-distribution. Vấn đề là **91 episodes không đủ** để model học được sự phụ thuộc vị trí vật → vị trí tay.

**Hướng giải quyết (chưa implement)**:

1. **Data augmentation từ CSV hiện tại** ← khuyến nghị làm trước
   - Dùng IK solver (Pinocchio, đã có sẵn trong `DualArmIK.py`) để shift vị trí vật trong các demos hiện có, tính lại joint angles tương ứng
   - Từ 91 demos thật → sinh thêm ~900 demos giả → tăng 10×
   - File liên quan: `src/baseline_source/DualArmIK.py`, `data/task1/part_sorting_sample.csv`

2. **Thu thập thêm demos bằng teleoperation**
   - Cần ít nhất 300–500 episodes thành công với vị trí vật đa dạng
   - Retrain từ đầu sau khi có đủ data

3. **Lọc dữ liệu bẩn trong CSV**
   - CSV hiện tại có `obj0_x` min = -3.999 (vật ở vị trí bất thường, có thể đã rơi)
   - Loại bỏ các frame có `obj_x < 0.2` hoặc `obj_z < 0.9`

---

### 4.2 Planner mode — tcp_offset chưa được calibrate

**Vị trí**: `configs/Part_Sorting.yaml` dòng 54

```yaml
tcp_offset: [0.0, 0.0, 0.0]   # hiện tại là zero
```

Khi `tcp_offset[2] = 0`, code tự dùng mặc định `tcp_z = 0.13m` (khoảng cách từ wrist đến đầu ngón). Nếu con số này sai với robot thực, gripper sẽ dừng cao/thấp hơn vật thực tế.

**Cách kiểm tra**: Chạy planner và tìm log:
```
[FSM] z_down approach: tcp_z=0.130m
[S2 CHECK] wrist z_base: actual=X.XXX target=X.XXX err=+0.0XXm
```

Nếu `err` lớn hơn 2–3cm → cần đo lại và set `tcp_offset: [0.0, 0.0, MEASURED_VALUE]`.

---

## 5. State vector IL model (38-dim)

```
Indices  Ý nghĩa
[0:7]    R arm joints — 7 khớp tay phải (radian)
[7:9]    R finger joints — 2 ngón tay phải (radian)
[9]      right gripper control  (-1 = open, +1 = close)
[10:38]  4 objects × 7 = [x, y, z, qx, qy, qz, qw]  (world frame)
```

**Action output (10-dim)**: 7 arm joints + 2 fingers + 1 gripper — chỉ tay phải, tay trái cố định.

---

## 6. Việc cần làm tiếp theo

| Ưu tiên | Việc | File liên quan |
|---------|------|----------------|
| 🔴 Cao | Viết code data augmentation bằng IK | `src/baseline_source/DualArmIK.py` |
| 🔴 Cao | Retrain LSTM sau khi có data aug | `robot_arm_training/task1/train.py` |
| 🟡 Trung bình | Calibrate `tcp_offset` cho planner | `configs/Part_Sorting.yaml` dòng 54 |
| 🟡 Trung bình | Test planner mode sau fix #1 và #2 | `src/baseline_source/main_fixed.py` |
| 🟢 Thấp | Đồng bộ checkpoint epoch 286 từ container về local | `robot_arm_training/task1/checkpoints/` |
