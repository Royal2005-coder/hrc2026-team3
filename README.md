# HRC2026 Team 3

Global Humanoid Robot Challenge 2026 — Vòng Giả Lập (Simulation Round)
Robot Walker S2 | NVIDIA Isaac Sim 5.1 | UBTECH Competition
Đội 3 — Trại Huấn Luyện UEL x BA x AILab

## 1. Khởi Động Nhanh (Quick Start)

### Bước 1: Mỗi lần truy cập vào workspace

```bash
bash /home/ubuntu/auto_setup.sh
```

Ý nghĩa: Thiết lập môi trường workspace và đồng bộ các đường dẫn/tài nguyên cần thiết cho team member.

### Bước 2: Vào thư mục dự án và cập nhật code mới nhất

```bash
cd /home/ubuntu/hrc2026_team3
git pull origin main
```

### Bước 3: Chuyển sang nhánh cá nhân (Ví dụ với Tài)

```bash
git checkout tai/perception
```

### Bước 4: Đồng bộ nhánh cá nhân với nhánh main

```bash
git merge main
```

### Bước 5: Chạy mô phỏng Isaac Sim

```bash
/isaac-sim/python.sh scripts/run_task1.py
```

## 2. Thành Viên Đội (Team Members)

| Vai Trò | Người Phụ Trách | Nhánh (Branch) | Nhiệm Vụ Chính | Trách Nhiệm Chuyên Môn |
| --- | --- | --- | --- | --- |
| PM / Quản Lý Dự Án | Gia | pm/integration | Quản lý repo, kịch bản tích hợp, thiết lập quy trình. Duyệt (Approve) PR. | Đảm bảo kiến trúc dự án, README, tài liệu chuẩn chỉnh. |
| Người 1 (N1): Camera/RGB-D + Perception | Tài | tai/perception | Trong scene hiện tại có những vật nào, thuộc loại gì, nằm ở đâu và có đủ tin cậy để gắp không? | Xử lý Camera RGB-D, nhận diện/phân loại cấu kiện, ước lượng tư thế (pose), đánh giá độ tin cậy. |
| Người 2 (N2): Transform + Planner/FSM | Thư | thu/planner | Robot nên gắp vật nào trước, đặt vào bin nào, và đang ở trạng thái (state) nào của task? | Đóng vai trò là Điều phối kỹ thuật (Technical Coordinator). Máy trạng thái (FSM), chọn vật thể, ánh xạ vị trí thùng (bin mapping). |
| Người 2 (N2) - Hỗ trợ: Planner Support | Bảo | bao/planner_support | Hỗ trợ Thư thiết kế logic FSM và xử lý các hàm biến đổi tọa độ. | Viết các hàm tiện ích (transform_utils.py), tham gia xây dựng/tối ưu Planner logic cùng Thư. |
| Người 3 (N3): Motion Primitive + Pick-Place | Vinh | vinh/motion | Với object pose và bin pose đã có, làm sao robot gắp vật lên và đặt đúng bin mà không rơi, không va chạm? | Nội suy quỹ đạo (waypoints), gắp-đặt (Pick-place), điều khiển lực nắm (gripper), tránh va chạm. |
| Người 4 (N4): Evaluation + Debug + Ablation | Gia | gia/evaluation | Hệ thống Task 1 có thật sự tốt lên không, lỗi chính là gì, và bản (version) nào nên giữ lại? | Thiết kế schema log, hệ thống đánh giá (eval.py), verify_in_bin, phân loại lỗi (taxonomy), chạy batch eval, phân tích dữ liệu (Pandas/CSV). Viết report, README cuối sprint. |

### Người Đánh Giá (Reviewers - Duyệt Pull Request)

| Người Đánh Giá | Vai Trò | Quyền Hạn |
| --- | --- | --- |
| Gia (PM) | Quản lý Dự án | Đánh giá code (Review code), duyệt (approve) và gộp (merge) PR vào nhánh main |
| Anh Hoàng Anh | Cố vấn/Quản trị (Mentor) | Đánh giá kiến trúc hệ thống, phê duyệt các thay đổi mang tính cốt lõi |
| Thầy Hải | Giảng viên/Cố vấn | Đánh giá bước cuối, phê duyệt mã nguồn trước khi nộp (submission-ready code) |

Lưu ý quan trọng: Mỗi PR bắt buộc được review bởi Thầy Hải và Mentor Hoàng Anh, sau đó PM Gia chạy eval.py để đánh giá và approve trước khi merge.

## 3. Cấu Trúc Dự Án (Project Structure)

```
hrc2026_team3/
|-- README.md
|-- .gitignore
|
|-- src/
|   |-- task1/                   # Mã nguồn do đội phát triển
|   |   |-- perception.py        # [Tài]   Nhận diện + phân loại
|   |   |-- planner.py           # [Thư+Bảo] Máy trạng thái FSM + ánh xạ thùng
|   |   |-- motion.py            # [Vinh]  Gắp-đặt + điều khiển tay gắp
|   |   |-- eval.py              # [Gia]   Đánh giá kết quả (Evaluator)
|   |   |-- logger.py            # [Gia]   Ghi log có cấu trúc
|   |   |-- camera_utils.py      # [Tài]   Tiện ích xử lý camera
|   |   |-- transform_utils.py   # [Thư+Bảo] Biến đổi tọa độ
|   |   +-- task1_runner.py      # [Gia]   Kịch bản chạy tích hợp
|   |
|   +-- baseline_source/         # Mã nguồn gốc tham khảo (không thay đổi)
|       |-- SceneBuilder.py      # Dựng môi trường (49KB)
|       |-- RobotArticulation.py # Điều khiển robot
|       |-- DualArmIK.py         # Động học ngược Pinocchio
|       |-- grasp_planner.py     # Tính toán vị trí gắp
|       |-- coordinate_utils.py  # Biến đổi tọa độ Không gian thực - Pinocchio
|       |-- DataLogger.py        # Ghi log định dạng CSV + HDF5
|       |-- config_loader.py     # Tải cấu hình YAML
|       |-- main_fixed.py        # Kịch bản chạy chính đã sửa đổi (ThanhTai)
|       +-- isaac_sim_robot_interface.py
|
|-- configs/                     # Cấu hình YAML cho từng Task
|   |-- Part_Sorting.yaml        # Task 1 (CHÍNH)
|   |-- Conveyor_Sorting.yaml    # Task 2
|   |-- Foam_Inlaying.yaml       # Task 3
|   +-- Packing_Box.yaml         # Task 4
|
|-- scripts/                     # Các kịch bản chạy lệnh (Run scripts)
|-- logs/task1/                  # Nơi lưu episode_log.csv, step_log.jsonl
|-- reports/                     # Nơi lưu eval_summary.csv, failure_cases.jsonl, ablation_task1.csv
|-- lab_outputs/                 # Kết quả bàn giao (Deliverables)
|-- tests/                       # Unit tests
|-- docs/                        # Tài liệu hướng dẫn (failure_taxonomy_task1.yaml)
|-- assets -> symlink            # Mô hình USD/URDF (đường dẫn ảo)
|-- lerobot -> symlink           # Chính sách ACT/Pi0 (đường dẫn ảo)
|-- Ubtech_sim_ref -> symlink    # Mã nguồn giả lập gốc (đường dẫn ảo)
```

## 4. Quy Trình Git và Các Nguyên Tắc (Git Workflow & Rules)

### 4.1 Quy Ước Đặt Tên Nhánh (Branch Naming Convention)

Cú pháp: <tên_thành_viên>/<module>-<mô-tả-ngắn-gọn>

Ví dụ:

- tai/perception-detect-workpieces
- thu/planner-fsm-state-machine
- bao/planner-transform-utils
- vinh/motion-pick-place-v1
- gia/evaluation-episode-logger
- pm/integration-run-script

### 4.2 Định Dạng Thông Điệp Commit (Commit Message Format)

Cú pháp: <loại_thay_đổi>(<phạm_vi>): <mô_tả_ngắn_gọn>

| Loại (Type) | Trường hợp sử dụng | Ví dụ |
| --- | --- | --- |
| feat | Thêm tính năng mới | feat(perception): detect 4 workpieces |
| fix | Sửa lỗi (bug) | fix(motion): gripper close timing |
| docs | Cập nhật tài liệu | docs(readme): add pipeline diagram |
| test | Thêm mã kiểm thử | test(eval): verify_in_bin unit test |
| refactor | Cải thiện cấu trúc mã (không thay đổi tính năng) | refactor(planner): simplify FSM |
| wip | Lưu tạm tiến độ (chưa hoàn thành) | wip: saving progress before leaving |

### 4.3 Quy Trình Làm Việc Hàng Ngày (Daily Workflow)

KHI VÀO workspace:

```bash
bash /home/ubuntu/auto_setup.sh
cd /home/ubuntu/hrc2026_team3
git pull origin main
git checkout <nhánh-của-mình>
git merge main
```

Chỉ nhận báo cáo khi có PR theo lịch đăng ký sheet.

TRONG LÚC LÀM VIỆC (thường xuyên kiểm tra):

```bash
git add -A
git commit -m "<loại>(<phạm_vi>): <mô_tả>"
```

TRƯỚC KHI RỜI ĐI (BẮT BUỘC):

```bash
git add -A
git commit -m "wip: saving progress"
git push origin <nhánh-của-mình>
```


### 4.4 Quy Trình Tạo Yêu Cầu Gộp Mã (Pull Request - PR)

Bước 1: Hoàn thành tính năng trên nhánh cá nhân

```bash
git add -A
git commit -m "feat(perception): detect + classify 4 workpieces"
git push origin tai/perception
```

Bước 2: Tạo PR trên GitHub

- Truy cập https://github.com/Royal2005-coder/hrc2026-team3
- Chọn tab "Pull requests" -> "New pull request".
- Base: main <- Compare: tai/perception.
- Tiêu đề (Title): "feat(perception): detect + classify 4 workpieces".
- Mô tả (Description): Ghi rõ những thay đổi, các tệp kết quả đầu ra, và các bài kiểm thử đã chạy.
- Người đánh giá (Assign reviewers): bắt buộc assign Thầy Hải, Mentor Hoàng Anh và PM Gia.

Bước 3: Thông báo trên Zalo

[PR] Tài tạo PR #<số> — feat(perception): detect workpieces. Link: <link PR>. Xin review: @Gia @Hoang Anh

Bước 4: Xem xét và đánh giá (Review)

- Thứ tự review: Thầy Hải (Teacher) -> Mentor Hoàng Anh -> PM Gia.
- Thầy Hải và Mentor Hoàng Anh review và xác nhận đạt chuẩn yêu cầu.
- PM Gia chạy eval.py, ghi log và đánh giá kết quả tự động, sau đó approve.

Bước 5: Chỉnh sửa theo phản hồi (Nếu có)

```bash
git add -A
git commit -m "fix(perception): address PR feedback"
git push origin tai/perception
```

Bước 6: Phê duyệt và Gộp mã (Approve + Merge)

PM Gia thực hiện gộp (merge) PR trên GitHub.
Thông báo trên Zalo: [MERGED] PR #<số> đã merge vào main

Bước 7: Các thành viên cập nhật lại nhánh của mình

```bash
git checkout main && git pull
git checkout <nhánh-mình> && git merge main
```

### 4.5 Danh Sách Kiểm Tra Trước Khi Tạo PR (PR Checklist)

(Người tạo PR tự kiểm tra)

- [ ] Mã nguồn chạy ổn định, không bị lỗi (crash).
- [ ] Có đầy đủ tệp kết quả/chứng cứ đính kèm (file log, hình ảnh, CSV, JSON).
- [ ] Thông điệp commit tuân thủ đúng định dạng quy định.
- [ ] Không tự ý thay đổi file của thành viên khác (trừ khi đã thống nhất trước).
- [ ] Không gắn cứng đường dẫn (hard-code path) trong mã nguồn (Sử dụng đường dẫn trong file configs/).
- [ ] Đã kiểm thử thành công trên môi trường workspace chung.

## 5. Quy Tắc Giao Tiếp Trên Nhóm Zalo

### 5.1 Cú Pháp Tin Nhắn

| Tag | Khi Nào Sử Dụng | Cú Pháp | Ví Dụ |
| --- | --- | --- | --- |
| [PR] | Thành viên tạo PR và thông báo | [PR] <Tên> tạo PR #<số> — <mô_tả>. Xin review: @<tên> | [PR] Tài tạo PR #1 — detect workpieces. @Gia review |
| [MERGED] | PM thông báo PR đã được gộp | [MERGED] PR #<số> đã merge vào main | [MERGED] PR #1 đã merge. Mọi người pull main |
| [LỖI] | Thành viên gặp lỗi cần hỗ trợ | [LỖI] <Tên> gặp lỗi <mô_tả>, cần help | [LỖI] Vinh gặp lỗi Isaac Sim crash khi pick-place |
| [HỎI] | Thành viên thắc mắc chung | [HỎI] <Câu_hỏi> | [HỎI] Camera nào dùng cho Task 1? |
| [BLOCK] | Thành viên bị tắc nghẽn, cần kết quả từ người khác | [BLOCK] <Tên> bị chặn bởi <việc>, cần @<tên> | [BLOCK] Thư cần perception output từ @Tài |

### 5.2 Quy Tắc Sử Dụng Workspace

- Trước khi VÀO: Luôn hỏi trên nhóm Zalo để kiểm tra xem có ai đang sử dụng hay không.
- KHÔNG dừng (stop) workspace khi chưa hỏi ý kiến cả đội.
- Nếu gặp lỗi: Chụp màn hình log lỗi và thông báo ngay lập tức. KHÔNG tự ý xóa file hệ thống hoặc file của người khác.

## 6. Thông Số Kỹ Thuật Task 1 — Desktop Sorting

### 6.1 Mục Tiêu (Goal)

Robot thực hiện gắp 4 cấu kiện (2 loại A + 2 loại B) và đặt vào đúng vị trí thùng (bin) tương ứng.

### 6.2 Cấu Hình Robot Walker S2

- Tổng cộng 41 khớp (joints) | 16 bậc tự do (DOF) đang kích hoạt (2 khớp hông + 2x7 khớp tay).
- Điểm tác động cuối (End-effector): L_sixforce_link / R_sixforce_link.
- Tay gắp (Gripper): Các khớp ngón tay điều khiển với thông số — Mở (open)=-0.0215, Đóng (close)=0.01, Lực (torque)=100.
- Động học ngược (IK): Thư viện Pinocchio [x,y,z,roll,pitch,yaw] so với hệ tọa độ gốc của robot.
- Tọa độ (Coordinate): Dùng hàm world_to_robot() thông qua điểm neo torso_link.

### 6.3 Bố Trí Môi Trường (Scene Layout)

| Đối Tượng | Vị Trí (Position) | Ghi Chú |
| --- | --- | --- |
| Robot | [0.70, -0.20, 0.90] | Hướng về phía +Y (xoay 90 độ) |
| Bàn (Table) | [0.75, 0.30, 0.50] | Chiều cao mặt bàn $z \approx 1.0m$ |
| Hộp/Thùng (Box/Bins) | [1.20, 0.30, 1.05] | Vị trí cố định, không thay đổi |
| Khu Vực Cấu Kiện | Tâm điểm [0.75, 0.28, 1.04] | Phạm vi rải: $x \in [0.50, 0.80]$, $y \in [0.10, 0.30]$ |

### 6.4 Phân Loại Cấu Kiện (Part Variants)

- Cấu kiện A (Part A): Màu đồng/vàng nguyên bản (ori_color) kết hợp màu đỏ.
- Cấu kiện B (Part B): Màu xanh dương kết hợp màu nguyên bản (ori_color).

### 6.5 Hệ Thống Camera

| Tên Camera | Vị Trí Gắn | Chức Năng / Ghi Chú |
| --- | --- | --- |
| head_stereo_L/R | head_pitch_link | Ứng viên cho xử lý RGB-D |
| head_fisheye_L/R | head_pitch_link | Camera góc rộng (Wide FOV) |
| waist_front_cam | Gốc s2_v1 (root) | Camera trước cố định |
| back_rear_cam | waist_pitch_link | Không sử dụng trong Task 1 |

## 7. Luồng Xử Lý Chính (Pipeline)

```
observe()
  -> perception.detect_parts(rgb, depth)
     -> [{object_id, class_id, confidence, pose_base, grasp_hint}]
  -> planner.select_next_object(objects, handled_ids)
  -> planner.map_class_to_bin(class_id)
  -> motion.pick_place(object_pose, bin_pose, grasp_hint)
     -> {success, duration_s, retry_count, failure_reason}
  -> evaluator.verify_in_bin(object_id, class_id)
  -> logger.log_step(...)
  -> Lặp lại cho đến khi xử lý xong 4 cấu kiện hoặc hết thời gian (timeout).
```

## 8. Tiêu Chí Đánh Giá (Metrics)

| Tiêu Chí | Đạt Yêu Cầu Cơ Bản (MVP) | Xuất Sắc (Good) |
| --- | --- | --- |
| Thành công toàn nhiệm vụ (4/4 vào đúng thùng) | $\ge 50\%$ | $\ge 95\%$ |
| Tỷ lệ thả đúng thùng (correct_bin_rate) | $\ge 75\%$ | $\ge 98\%$ |
| Số lần thả sai thùng (wrong_bin_count) | $\le 10$ | $0$ |
| Số lần làm rơi cấu kiện (drop_count) | $\le 10$ | $0$ |
| Số lần va chạm (collision_count) | $0-10$ | $0$ |
| Số lần thử lại (retry_count) | $\le 2$ | $\le 1$ |

## 9. Các Đường Dẫn Quan Trọng (Key Paths)

| Mục | Đường Dẫn (Path) |
| --- | --- |
| Kho lưu trữ của đội (Được lưu vĩnh viễn) | /home/ubuntu/hrc2026_team3/ |
| Mã nguồn Baseline (Chỉ đọc) | /workspace/GlobalHumanoidRobotChallenge_2026_Baseline/ |
| Kịch bản khởi động Isaac Sim | /isaac-sim/python.sh |
| Mô hình Robot (URDF) | assets/resources/s2.urdf |
| Cấu hình Task 1 | configs/Part_Sorting.yaml |
| Kịch bản kiểm tra tự động | bash /home/ubuntu/verify.sh |
| Kịch bản thiết lập ban đầu | bash /home/ubuntu/auto_setup.sh |
| Thư mục sao lưu sửa đổi Baseline | /home/ubuntu/backup_baseline_mods/ |

## 10. Tóm Tắt Kỷ Luật (Rules Summary)

- KHÔNG lập trình trực tiếp trên nhánh main. Phải tạo PR và chờ được review duyệt mới gộp mã.
- LUÔN commit và push lên Git TRƯỚC KHI rời khỏi workspace.
- LUÔN cập nhật (pull) code mới nhất SAU KHI vào workspace.
- CHỈ làm việc trong thư mục /home/ubuntu/hrc2026_team3/.
- KHÔNG thay đổi các file trong /workspace/ (do sẽ bị mất dữ liệu khi khởi động lại).
- KHÔNG tắt (stop) workspace khi chưa có sự đồng ý của cả đội.
- Tuân thủ quy tắc thông báo trên Zalo: [PR], [MERGED], [LỖI], [HỎI], [BLOCK].
- Mọi PR bắt buộc phải có approval từ mentor, teacher và đảm bảo chạy test PM trước khi merge.
- Ghi chú commit (Commit message) phải đúng định dạng: type(scope): mô tả.
- KHÔNG tự ý sửa đổi file do thành viên khác phụ trách nếu chưa thảo luận trước.

## 11. Liên Kết Hữu Ích (Links)

| Tài Nguyên | Liên Kết (URL) |
| --- | --- |
| Team GitHub Repository | https://github.com/Royal2005-coder/hrc2026-team3 |
| Mã Nguồn UBTECH Baseline | https://github.com/UBTECH-Robot/GlobalHumanoidRobotChallenge_2026_Baseline |
| Tài Sản Cuộc Thi (Assets) | https://huggingface.co/UBTECH-Robotics/challenge2026_assets |
| Tài Liệu Isaac Sim | https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ |
| Tài Liệu LeRobot | https://huggingface.co/docs/lerobot |

Cập nhật: 18/05/2026