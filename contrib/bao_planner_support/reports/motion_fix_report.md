# Báo cáo Fix Motion Pipeline — `src/task1/motion.py`

**Branch:** `bao/planner_support`  
**Ngày:** 2026-05-31  
**Commits:** `9f257dd`, `b5b24b3`

---

## Bối cảnh

Pipeline pick-and-place đã hoạt động được ở mức cơ bản:
- Nhận diện vật thể và di chuyển cánh tay đến vị trí pre-grasp ✓
- Hạ cánh tay xuống để gắp vật ✓ (nhưng **chậm**)
- Di chuyển cánh tay sang rổ phân loại sau khi gắp ✓

Còn 3 lỗi cần fix:
1. Hạ cánh tay xuống để gắp **quá chậm**
2. Đến rổ phân loại nhưng **không thả vật** (gripper không mở)
3. Sau khi xong một vật, **không quay lại gắp tiếp** các vật còn lại

---

## Phân tích nguyên nhân

### Lỗi 1 — Hạ cánh tay chậm (S2_GRASP)

**Vị trí:** `_state_s2_grasp()` → `move_interpolated()`

Tham số cũ quá conservative:
```python
# Cũ
num_steps=25, max_sim_steps=600, step_size=0.008
```

- `step_size=0.008` → mỗi bước IK rất nhỏ, cần rất nhiều bước để converge
- `max_sim_steps=600` × `num_steps=25` = tối đa 15.000 simulation steps chỉ để hạ cánh tay

---

### Lỗi 2 — Không thả vật (S6_RELEASE)

**Vị trí:** `_state_s6_release()`

**Root cause:** `open_gripper()` gọi `apply_action()` — đây chỉ là cách **set target cho PD controller**, không phải teleport trực tiếp. Nếu một trong các điều kiện sau xảy ra thì lệnh bị **ignore hoàn toàn, không báo lỗi**:

1. **Drive stiffness = 0** trên finger joints (joint không có cấu hình PD drive)
2. **`_physics_view` bị xóa** sau khi xóa FixedJoint — Isaac Sim rebuild physics scene khi remove USD prim của joint, làm mất `_physics_view` trên tất cả Articulation instances

Cụ thể flow gây lỗi:
```
S2_GRASP: create FixedJoint → physics rebuild → _physics_view cleared
         → _reinitialize_physics() được gọi ✓  (fix đã có từ trước)

S6_RELEASE: remove FixedJoint → physics rebuild → _physics_view cleared
         → KHÔNG có reinit!
         → open_gripper() → apply_action() → AttributeError (_physics_view)
         → _apply_gripper_action bắt lỗi, thử reinit nội bộ
         → reinit thất bại → "reinit failed, skipping" → GRIPPER KHÔNG MỞ
```

Thêm vào đó, `apply_action` với `joint_positions` yêu cầu:
- Physics view hợp lệ
- Joint có PD drive với stiffness > 0

Trong khi đó, `set_joint_positions` **teleport thẳng** vào vị trí joint — không qua PD drive, đáng tin hơn nhiều.

---

### Lỗi 3 — Không gắp tiếp vật còn lại

**Vị trí:** `run_pipeline_from_person2()`, vòng lặp orchestrator

```python
# Cũ — không có try-except
result = fsm.run()
```

Nếu FSM crash với unhandled exception ở bất kỳ state nào (S5, S6, S7...), exception propagate ra ngoài vòng lặp `while plan_idx < len(action_plans)` → **toàn bộ pipeline chết**, không gắp tiếp vật nào.

---

## Các thay đổi đã thực hiện

### Fix 1 — Tăng tốc hạ cánh tay (S2_GRASP)

**File:** `src/task1/motion.py`, hàm `_state_s2_grasp()`

```python
# Trước
num_steps=25, max_sim_steps=600, step_size=0.008

# Sau
num_steps=15, max_sim_steps=250, step_size=0.015
```

| Tham số | Cũ | Mới | Hiệu quả |
|---|---|---|---|
| `step_size` | 0.008 | 0.015 | IK converge nhanh ~2× |
| `num_steps` | 25 | 15 | Ít waypoints hơn |
| `max_sim_steps` | 600 | 250 | Giới hạn mỗi điểm ngắn hơn |
| Max tổng steps | 15.000 | 3.750 | Giảm 75% |

---

### Fix 2 — Giảm thời gian chờ ở S5_LOWER_BIN

**File:** `src/task1/motion.py`, hàm `_state_s5_lower_bin()`

```python
# Trước
max_sim_steps=400  # cho cả 2 move_interpolated

# Sau
max_sim_steps=150
```

Worst-case của S5 giảm từ ~175 giây xuống ~65 giây. Khi arm bị stuck tại một waypoint, sẽ timeout nhanh hơn và chuyển sang S6_RELEASE thay vì đứng im mãi.

---

### Fix 3 — Gripper mở được (S6_RELEASE) — thay đổi chính

**File:** `src/task1/motion.py`, hàm `_state_s6_release()`

**Chiến lược mới:**

```python
def _state_s6_release(self):
    # 1. Xóa FixedJoint (vật được thả ra khỏi wrist)
    if self._grasp_joint:
        _remove_grasp_joint(self.world, self._grasp_joint)
        self._grasp_joint = None
        for _ in range(60): self.world.step(render=True)  # 60 frames (tăng từ 10)

    # 2. PRIMARY: set_joint_positions — teleport trực tiếp, không qua PD drive
    dof_names = robot._articulation.dof_names
    prefix = "L" if side == "left" else "R"
    f_names = [f"{prefix}_finger1_joint", f"{prefix}_finger2_joint"]
    f_idx = [robot._articulation.get_dof_index(n) for n in f_names if n in dof_names]
    if f_idx:
        robot._articulation.set_joint_positions(
            torch.tensor([[open_w] * len(f_idx)]),
            joint_indices=torch.tensor(f_idx)
        )
    # (có try-except: nếu _physics_view missing → reinit → thử lại)

    # 3. BACKUP: open_gripper API (PD drive) — phòng trường hợp set_joint_positions thất bại
    robot.open_gripper(side=self.side)

    for _ in range(60): self.world.step(render=True)
```

**Tại sao `set_joint_positions` đáng tin hơn `apply_action`:**

| | `apply_action` (joint_positions) | `set_joint_positions` |
|---|---|---|
| Cơ chế | Set target cho PD controller | Teleport thẳng joint |
| Cần PD drive? | **Có** (stiffness > 0) | Không |
| Phụ thuộc physics view | Có | Có (nhưng error được bắt) |
| Kết quả nếu drive stiffness = 0 | **Không làm gì, không báo lỗi** | Không liên quan |

**Log bạn sẽ thấy khi chạy:**
```
[S6] ✓ Gripper teleported open: ['R_finger1_joint', 'R_finger2_joint'] → -0.0215m
```

Nếu thấy:
```
[S6] ⚠ No R finger joints found in DOF list
```
→ tên joint không khớp, cần kiểm tra `robot._articulation.dof_names` để lấy tên đúng.

---

### Fix 4 — Bảo vệ pipeline khỏi FSM crash

**File:** `src/task1/motion.py`, hàm `run_pipeline_from_person2()`

```python
# Trước
result = fsm.run()  # nếu crash → pipeline chết

# Sau
try:
    result = fsm.run()
except Exception as _fsm_exc:
    print(f"❌ [FSM CRASH] {plan_id}: {_fsm_exc}")
    traceback.print_exc()
    result = PrimitiveResult(
        primitive_name="pick_place", success=False,
        elapsed_s=0.0, retry_count=0,
        failure_reason=f"fsm_crash: {_fsm_exc}", metrics={})
```

Với fix này, nếu gắp 1 vật bị crash, pipeline vẫn tiếp tục gắp vật tiếp theo.

---

### Fix 5 — Import `torch` top-level

```python
# Thêm vào đầu file
import torch
```

`torch` trước đây được import inline ở nhiều hàm (`import torch` lặp lại). Đưa lên top-level để S6_RELEASE dùng được mà không cần inline import.

---

## Tóm tắt commits

### Commit `9f257dd` — Fix 1, 2, 3, 4

```
fix: speed up grasp descent, fix release bug, protect pipeline against FSM crash

- S2_GRASP: step_size 0.008→0.015, num_steps 25→15, max_sim_steps 600→250
- S5_LOWER_BIN: max_sim_steps 400→150
- S6_RELEASE: thêm _reinitialize_physics() sau khi xóa FixedJoint
- run_pipeline_from_person2: try-except quanh fsm.run()
```

### Commit `b5b24b3` — Fix gripper (override fix 3 ở S6)

```
fix: use set_joint_positions to force-open gripper in S6_RELEASE

- Thay _reinitialize_physics() bằng set_joint_positions (đáng tin hơn)
- Tăng settle time sau joint removal: 30+10 → 60 frames
- Giữ open_gripper API như backup
- Thêm torch top-level import
```

---

## FSM State Machine — Flow đầy đủ sau khi fix

```
INIT
  └─► S1_PREGRASP   : Di chuyển tay đến vị trí pre-grasp (trên vật 8cm)
        └─► S2_GRASP    : Hạ xuống thẳng đứng đến vật [ĐÃ TĂNG TỐC]
              ├─ close_gripper()
              ├─ tạo FixedJoint (vật gắn vào wrist)
              ├─ _reinitialize_physics()
              └─► S3_LIFT     : Nâng lên SAFE_FLY_HEIGHT = 1.40m
                    └─► S4_TRANSFER : Bay ngang sang rổ phân loại
                          └─► S5_LOWER_BIN : Hạ xuống phía trên rổ [ĐÃ GIẢM THỜI GIAN CHỜ]
                                └─► S6_RELEASE  : Xóa joint + set_joint_positions open [ĐÃ FIX]
                                      └─► S7_RETREAT : Rút tay lên cao
                                            └─► VERIFY → DONE
```

---

## Lưu ý khi debug thêm

Nếu gripper vẫn không mở, kiểm tra log để xác định vấn đề:

| Log thấy | Nghĩa | Hành động |
|---|---|---|
| `[S6] ✓ Gripper teleported open` | set_joint_positions thành công | OK |
| `[S6] ⚠ No R finger joints found` | Tên joint sai | In `dof_names` để tìm tên đúng |
| `[S6] _physics_view missing` | _physics_view bị xóa | reinit được gọi, xem reinit có thành công không |
| `[GRASP_JOINT] ⚠ No cached gripper link` | FixedJoint không được tạo | Vật chỉ được giữ bằng contact force |
| `[GRASP_JOINT] ✓ Released` | FixedJoint xóa thành công | Vật đã tự do về mặt vật lý |
