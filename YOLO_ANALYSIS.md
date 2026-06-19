# Phân tích tính khả thi của YOLO trong bài toán Part Sorting

## Bối cảnh

Cuộc thi HRC 2026 sử dụng robot Walker S2 trong môi trường mô phỏng Isaac Sim.
Task **Part Sorting** yêu cầu robot nhận biết vị trí các vật thể trên bàn và di chuyển chúng
vào đúng vị trí. Dataset do BTC cung cấp gồm 756 episode dạng video từ 4 camera
(head_left, head_right, wrist_left, wrist_right) kèm theo joint positions và object poses.

---

## Lý do YOLO không phù hợp cho bài toán này

### 1. YOLO chỉ cho tọa độ 2D — robot cần 3D

YOLO detect vật thể và trả về **bounding box trên ảnh** (pixel x, y, width, height).
Để robot biết cần đưa tay đến đâu, nó cần **tọa độ 3D trong không gian thực** (đơn vị mét,
trong hệ tọa độ của robot hoặc world frame).

Chuyển từ 2D pixel sang 3D world coordinate đòi hỏi:
- **Camera calibration** chính xác (intrinsics + extrinsics cho từng camera)
- **Depth information** — hoặc depth camera, hoặc stereo vision (dùng 2 camera để tính)
- **Giải bài toán PnP** (Perspective-n-Point) để tìm pose 3D từ 2D correspondences

Trong khi đó, Isaac Sim đã cung cấp **ground-truth 3D pose** của từng vật thể
(x, y, z, qx, qy, qz, qw) trực tiếp từ physics engine, chính xác tuyệt đối, không cần
bất kỳ bước tính toán nào. YOLO không thể đạt được độ chính xác này.

---

### 2. Dataset đã có sẵn thông tin mà YOLO cần tái tạo

State vector trong dataset BTC đã bao gồm:

```
obj0_x, obj0_y, obj0_z, obj0_qx, obj0_qy, obj0_qz, obj0_qw
obj1_x, obj1_y, obj1_z, obj1_qx, obj1_qy, obj1_qz, obj1_qw
obj2_x, obj2_y, obj2_z, obj2_qx, obj2_qy, obj2_qz, obj2_qw
obj3_x, obj3_y, obj3_z, obj3_qx, obj3_qy, obj3_qz, obj3_qw
```

28 giá trị này mô tả đầy đủ vị trí và hướng của 4 vật thể trong không gian 3D,
được lấy trực tiếp từ simulator — không có sai số.

Dùng YOLO để cố gắng tái tạo lại thông tin đã có sẵn này là **thêm việc mà không thêm giá trị**,
thậm chí còn kém chính xác hơn.

---

### 3. YOLO không nhận ra các vật thể trong Part Sorting

YOLO các phiên bản phổ biến (YOLOv8, YOLOv11) được pre-train trên tập COCO
(80 class: người, xe, ghế, ...) hoặc ImageNet. Các part trong cuộc thi HRC 2026
**không có trong tập train của bất kỳ YOLO nào**.

Để YOLO hoạt động được, cần phải:
1. Thu thập ảnh của đúng các part đó trong môi trường Isaac Sim
2. Label tay từng vật thể trong từng frame (hoặc dùng synthetic labeling)
3. Fine-tune YOLO trên tập data mới này
4. Validate độ chính xác trước khi tích hợp vào pipeline

Đây là một dự án riêng biệt, tốn nhiều công sức, trước khi có thể bắt đầu làm
bất cứ điều gì liên quan đến điều khiển robot.

---

### 4. Tay robot che khuất vật tại thời điểm quan trọng nhất

Trong quá trình **grasp** (kẹp vật), tay robot di chuyển xuống và **che khuất vật thể**
hoàn toàn khỏi góc nhìn camera. Đây chính xác là thời điểm mà hệ thống cần biết
vị trí vật thể nhất để điều chỉnh góc kẹp.

Khi bị che khuất:
- YOLO mất khả năng detect → không có tọa độ
- Phải dùng tracking (giữ lại vị trí cũ) → tích lũy sai số
- Isaac Sim vẫn cho đúng vị trí vật dù bị che khuất hoàn toàn

---

### 5. Nhiều vật thể cùng loại — cần instance tracking, không chỉ detection

Trong Part Sorting, nhiều part có thể **cùng loại và cùng hình dạng**. YOLO chuẩn
chỉ phân loại class (loại vật), không phân biệt được instance (vật nào là vật nào).

Ví dụ: nếu có 3 cái bánh răng giống nhau trên bàn, YOLO detect được 3 cái bánh răng
nhưng không theo dõi được "cái nào đang ở đâu" qua các frame. Cần thêm **instance
segmentation** (YOLOv8-seg trở lên) và **multi-object tracking** (ByteTrack, BotSORT)
— tăng đáng kể độ phức tạp và latency.

---

### 6. Độ trễ không phù hợp với điều khiển real-time

Hệ thống chạy ở **30fps** với 4 camera đồng thời. Để đưa YOLO vào vòng lặp điều khiển:

- Mỗi frame từ 4 camera đều cần inference → 4 lần YOLO/frame
- YOLOv8n (nhanh nhất): ~5ms/image trên GPU → 20ms cho 4 camera
- Cộng thêm 2D→3D conversion, tracking, coordinate transform
- Tổng latency có thể vượt quá budget 33ms/frame (30fps)

Và nếu chỉ chạy trên 1-2 camera để tiết kiệm thời gian, độ chính xác 3D giảm thêm
do mất thông tin stereo.

---

### 7. Vấn đề gốc rễ của dataset không liên quan đến object detection

Lý do model (SmolVLA) không hoạt động tốt là:

- **Tay trái không di chuyển** trong toàn bộ 756 episode (standard deviation ≈ 0.001)
- **MEAN_STD normalization** khiến model không thể predict các góc joint cực đoan
  cần thiết để hạ tay xuống bàn (cần output -4.43σ, gần như không thể)

YOLO không giải quyết được bất kỳ vấn đề nào trong số này. Fix thực sự là:
- Đổi sang **MIN_MAX normalization** (đã áp dụng cho ACT)
- Thu thập thêm demo với workspace coverage rộng hơn

---

## Tổng kết

| Tiêu chí | Isaac Sim (hiện tại) | YOLO |
|---|---|---|
| Độ chính xác vị trí 3D | Tuyệt đối (ground-truth) | Thấp (cần 2D→3D, có sai số) |
| Hoạt động khi bị che khuất | Có | Không |
| Cần training riêng | Không | Có (tốn thêm thời gian) |
| Phù hợp real-time 30fps | Có | Có thể, nhưng thêm latency |
| Giải quyết vấn đề arm movement | Không | Không |
| Độ phức tạp tích hợp | Thấp | Rất cao |

**Kết luận**: Trong môi trường Isaac Sim với dataset đã có ground-truth object poses,
dùng YOLO để tạo hoặc cải thiện dataset là **không khả thi về mặt hiệu quả** —
công sức bỏ ra lớn hơn nhiều so với lợi ích thu được, và kết quả vẫn kém hơn
dữ liệu simulator đã có sẵn.

YOLO chỉ thực sự phù hợp khi triển khai trên **robot thực** (real robot) không có
simulator, nơi không có ground-truth object poses và cần dùng camera để ước lượng.
