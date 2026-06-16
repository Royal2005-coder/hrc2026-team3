# SmolVLA Training — Part Sorting (2026-06-14)

## Quyết định chọn model

BTC docs (Model Training section, docs/4) liệt kê 5 policy: ACT, Diffusion, PI0, PI0.5, SmolVLA.
Lý do chọn **SmolVLA** thay vì ACT/PI0.5:

- Môi trường GUI **random vị trí vật thể trên bàn mỗi lần chạy**, và dataset hiện có (756-1000 episodes,
  single task) **không phủ hết không gian random** này.
- **ACT** = behavior cloning từ scratch (resnet18), không có prior về định vị vật → dễ "bắt chước không
  tới nơi" khi vật ở vị trí chưa thấy trong demo (generalization "Limited" theo BTC).
- **PI0.5** generalization "Excellent" nhưng full fine-tune backbone 2B+ trên dataset nhỏ (756 ep) với
  100K steps ≈ ~22 epoch → rủi ro overfit cao, có thể "xoá" luôn prior pretrained.
- **SmolVLA**: `freeze_vision_encoder=True`, `train_expert_only=True` (mặc định trong code) → vision
  encoder pretrained giữ nguyên khả năng định vị vật ở vị trí lạ, chỉ action expert (~100M/450M params)
  được train trên data của mình → ít overfit hơn, generalization "Excellent" theo BTC.

## Dataset

`Part_Sorting/part_sorting_long_756_episode/` (756 episodes, 580,608 frames, ~768 frames/episode,
robot_type=walker_s2_sim, codebase v3.0).

## Setup môi trường đã sửa (venv `/home/team3/Isaac-GR00T/.venv`)

Venv này trước đó dùng cho GR00T training, chưa có deps cho `src/lerobot/scripts/lerobot_train.py`
của repo baseline. Đã cài/sửa (qua `uv pip install --python /home/team3/Isaac-GR00T/.venv/bin/python ...`):

- Cài thêm: `draccus`, `pyserial`, `jsonlines`, `rerun-sdk`, `deepdiff`, `gymnasium`, `diffusers`,
  `opencv-python-headless`, `einops`, `imageio[ffmpeg]`, `termcolor`, `num2words`, `safetensors`
- Gỡ `deepspeed` (lỗi compile triton vì thiếu `python3-dev`, không cần cho training 1 GPU)
- Nâng `transformers` 4.51.3 → 4.57.6 (repo yêu cầu `transformers>=5.3.0` cho extra `smolvla`,
  4.57.6 đã có `transformers.masking_utils` cần cho PI0/SmolVLA và vẫn `import gr00t` OK)
- Nâng `datasets` 3.6.0 → 4.8.5 (repo yêu cầu `>=4.0.0`, cần cho feature type `"List"` trong
  meta/episodes parquet của dataset v3.0)
- Numpy được nâng 1.26.4 → 2.4.6 theo dependency resolution (đã verify torch + gr00t vẫn OK)

**Lưu ý**: các thay đổi trên là chung cho cả venv `Isaac-GR00T/.venv` — nếu resume GR00T training
sau này (kế hoạch 60-90K steps) mà gặp lỗi do version mới, có thể cần kiểm tra lại.

## Lệnh training đang chạy

```bash
cd /home/team3/GlobalHumanoidRobotChallenge_2026_Baseline
/home/team3/Isaac-GR00T/.venv/bin/python -m src.lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=team3/part_sorting_long_756_episode \
  --dataset.root=Part_Sorting/part_sorting_long_756_episode \
  --dataset.video_backend=pyav \
  --output_dir=outputs/train/smolvla_part_sorting_long \
  --batch_size=64 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --log_freq=200 \
  --wandb.enable=false
```

- Output/checkpoints: `outputs/train/smolvla_part_sorting_long/checkpoints/`
- Log file: `/tmp/smolvla_train.log`
- ~2.6s/step, batch=64 → **30,000 steps ≈ 21-22 giờ** (~3.3 epoch trên 756 episodes)
- Checkpoint mới mỗi 5000 steps (~3.6 giờ)
- GPU: ~24.3GB/49GB dùng, 100% utilization

## Theo dõi log

```bash
# Xem realtime toàn bộ log
tail -f /tmp/smolvla_train.log

# Chỉ xem loss/step (cập nhật mỗi 200 step)
grep -o "step:[0-9]* smpl:[0-9]* ep:[0-9]* epch:[0-9.]* loss:[0-9.]* grdn:[0-9.]* lr:[^ ]*" /tmp/smolvla_train.log | tail -20

# Kiểm tra GPU
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

## Bước tiếp theo

- Sau mỗi checkpoint (5000 steps), load vào GUI eval (Isaac Sim, random object position) để theo dõi
  **eval performance theo random layout** — không chỉ train loss — vì mục tiêu là chọn checkpoint
  generalize tốt nhất, tránh checkpoint cuối bị overfit vào 756 demo.
- Nếu loss giảm nhưng eval-with-random-position không cải thiện/giảm → dừng sớm, chọn checkpoint trước đó.

## Kết quả training (HOÀN THÀNH 2026-06-15 22:24)

Training chạy đủ 30,000 steps trong **~23h6m** (03:08 → 22:24, 2026-06-15), exit code 0, "End of training".

Loss theo checkpoint (epch = epoch trên 580,608 frames):

| step | epoch | loss   | grad_norm | lr      |
|------|-------|--------|-----------|---------|
| 5K   | 0.55  | 0.061-0.089 | ~1.0  | 9.4e-05 |
| 10K  | 1.10  | 0.038-0.054 | ~0.5  | 7.6e-05 |
| 15K  | 1.65  | 0.028-0.045 | ~0.45 | 5.2e-05 |
| 20K  | 2.20  | 0.023-0.029 | ~0.39 | 2.7e-05 |
| 25K  | 2.76  | 0.019-0.022 | ~0.33 | 9.3e-06 |
| 30K  | 3.31  | 0.020-0.023 | ~0.32 | 2.5e-06 |

Loss **plateau từ ~step 20K (epoch 2.2) trở đi** — không còn cải thiện đáng kể, dao động quanh 0.02-0.03.
=> Checkpoint 20000-25000 có khả năng generalize tốt tương đương 30000 nhưng ít rủi ro overfit hơn.

Checkpoints có sẵn tại `outputs/train/smolvla_part_sorting_long/checkpoints/`:
`005000/ 010000/ 015000/ 020000/ 025000/ 030000/` (và symlink `last -> 030000`).

## Eval trong GUI (random object position) — chưa làm

Đây là bước quan trọng nhất còn lại: load từng checkpoint (gợi ý ưu tiên 020000, 025000, 030000) vào
Isaac Sim GUI, chạy nhiều lần với vị trí vật random, so sánh tỉ lệ thành công. Chọn checkpoint
generalize tốt nhất, không nhất thiết là 030000 (loss thấp nhất nhưng epoch cao nhất = rủi ro overfit
cao nhất, theo phân tích "khả năng học bắt chước không tới nơi" ở trên).

## Debug eval (2026-06-16) — các vấn đề đã gặp và fix

### Fix 1: `config.json` thiếu key `"type"`
Draccus 0.11.6 yêu cầu key `"type": "smolvla"` ở đầu `config.json` để dispatch đúng subclass.
Checkpoint được save ra không có key này. Đã fix thủ công cho tất cả 6 checkpoints:
```bash
for step in 005000 010000 015000 020000 025000 030000; do
  python3 -c "
import json; f='outputs/train/smolvla_part_sorting_long/checkpoints/${step}/pretrained_model/config.json'
d=json.load(open(f)); d={'type':'smolvla',**d}; json.dump(d,open(f,'w'),indent=4)
"
done
```

### Fix 2: Script phải chạy bên trong container `ghrc_2026:v0`
`walkers2sim.py` import `from isaacsim import SimulationApp` — module này chỉ có trong Isaac Sim
Python (`/isaac-sim/kit/python/bin/python3`). Không thể chạy từ host venv `Isaac-GR00T/.venv`.

Cách chạy đúng: `docker run ghrc_2026:v0 /isaac-sim/python.sh src/lerobot/scripts/lerobot_record.py ...`
(truyền script file, KHÔNG dùng `-m` vì Isaac Sim's kit framework chiếm mất flag đó → segfault)

### Fix 3: Mount assets đúng chỗ
Task config `Ubtech_sim/config/Part_Sorting.yaml` resolve scene USD tới `assets/resources/...`
(relative từ project root). Nhưng USD files thực tế nằm ở `/opt/hrc2026/assets/resources/`.
Phải mount: `-v /opt/hrc2026/assets/resources:/workspace/assets/resources:ro`

### Fix 4: HuggingFace cache
Mount đúng path: `-v /home/team3/.cache/huggingface:/root/.cache/huggingface`
(KHÔNG dùng `/root/.cache/...` trên host vì máy chạy với user `team3`, không phải `root`)

### Fix 5: Streaming mode — gốc rễ và giải pháp đã verify (2026-06-16)

**Root cause**: `ghrc_2026:v0` image có ENTRYPOINT là `/isaac-sim/runheadless.sh` (không phải bash).
Khi `docker run ghrc_2026:v0 <cmd>`, `<cmd>` được truyền như args cho `runheadless.sh`, script này
gọi `isaac-sim.streaming.sh ... "$@"` → Isaac Sim luôn khởi động ở "Full Streaming App" mode
(chờ WebSocket client), Python script không bao giờ chạy được.

**Giải pháp đã verify HOẠT ĐỘNG**: Dùng container `eval_smolvla` với `sleep infinity` + `docker exec`.
Flow:
1. Extract xauth cho display `:1` (Xvfb đang chạy trên host)
2. `docker run -d ... sleep infinity` — container giữ sống, không qua `runheadless.sh`
3. `docker exec -d` script vào container — chạy `/isaac-sim/python.sh script.py` trực tiếp
4. Kit framework detect DISPLAY=:1 + XAUTHORITY → dùng `isaacsim.exp.base.python.kit` (KHÔNG phải streaming)
5. "app ready" sau 10s (vs 98s trong streaming mode)
6. Khi prompt "调整好按Enter键开始录制" xuất hiện → gửi Enter từ host bằng sendkey binary

**Key insight**: Khi XAUTHORITY đúng + DISPLAY hợp lệ, SimulationApp tự chọn base Python profile.
KHÔNG cần `--robot.headless=true`.

### Fix 6: Assets mount path khác (workspace nested deeper)

Container `eval_smolvla` mount workspace tại `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline`
(khác command cũ dùng `/workspace`). Path resolution trong YAML:
`Ubtech_sim/config/../../assets/resources/` → `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/assets/resources/`

Folder `assets/resources/` trong workspace rỗng (không có Collected_Task4). Cần mount:
`-v /opt/hrc2026/assets/resources:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/assets/resources:ro`

### sendkey — compile thủ công (không có x11-dev headers)
```bash
# Tạo /tmp/sendkey.c (không cần headers, declare prototypes thủ công):
cat > /tmp/sendkey.c << 'EOF'
typedef unsigned long KeySym;
typedef unsigned char KeyCode;
typedef unsigned long XID;
typedef XID Window;
typedef struct _XDisplay Display;
typedef int Bool;
#define None 0L
#define True 1
#define False 0
Display *XOpenDisplay(const char *display_name);
void XCloseDisplay(Display *d);
KeySym XStringToKeysym(const char *string);
KeyCode XKeysymToKeycode(Display *d, KeySym keysym);
int XFlush(Display *d);
int XSync(Display *d, Bool discard);
int XTestFakeKeyEvent(Display *d, unsigned int keycode, Bool is_press, unsigned long delay);
int main(int argc, char *argv[]) {
    const char *key = argc > 1 ? argv[1] : "Return";
    Display *d = XOpenDisplay(0);
    if (!d) return 1;
    KeySym sym = XStringToKeysym(key);
    KeyCode code = XKeysymToKeycode(d, sym);
    XTestFakeKeyEvent(d, code, True, 0);
    XTestFakeKeyEvent(d, code, False, 0);
    XFlush(d); XSync(d, False); XCloseDisplay(d);
    return 0;
}
EOF
gcc /tmp/sendkey.c /usr/lib/x86_64-linux-gnu/libX11.so.6 /usr/lib/x86_64-linux-gnu/libXtst.so.6 -o /tmp/sendkey
```

### Command eval ĐÚNG và ĐÃ VERIFY (2026-06-16 04:00) — CHẠY THÀNH CÔNG:

```bash
# 1. Chuẩn bị (1 lần)
xauth extract /tmp/.docker.xauth :1 && chmod 644 /tmp/.docker.xauth
xhost +local:docker

# 2. Khởi động container
docker run -d \
  --name eval_smolvla \
  --privileged --network host --user root --gpus all --shm-size=8g \
  -e DISPLAY=:1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e PYTHONPATH=/workspace/GlobalHumanoidRobotChallenge_2026_Baseline \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/.docker.xauth:/tmp/.docker.xauth:ro \
  -v /home/team3/GlobalHumanoidRobotChallenge_2026_Baseline:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline:rw \
  -v /opt/hrc2026/assets/resources:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/assets/resources:ro \
  -v /home/team3/.cache/isaac_sim_container/cache/kit:/root/.cache/kit:rw \
  -v /home/team3/.cache/isaac_sim_container/cache/ov:/root/.cache/ov:rw \
  -v /home/team3/.cache/isaac_sim_container/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v /home/team3/.cache/isaac_sim_container/cache/computecache:/root/.cache/nvidia/ComputeCache:rw \
  -v /home/team3/.cache/isaac_sim_container/data:/root/.local/share/ov/data:rw \
  -v /home/team3/.cache/huggingface:/root/.cache/huggingface:rw \
  -w /workspace/GlobalHumanoidRobotChallenge_2026_Baseline \
  ghrc_2026:v0 \
  sleep infinity

# 3. Chạy eval checkpoint 020000 (đổi 020000→025000/030000 cho các ckpt khác)
docker exec -d eval_smolvla bash -c '
/isaac-sim/python.sh src/lerobot/scripts/lerobot_record.py \
  --robot.type=walker_s2_sim \
  --task=Part_Sorting \
  --policy.path=outputs/train/smolvla_part_sorting_long/checkpoints/020000/pretrained_model \
  --dataset.repo_id=team3/eval_smolvla_20k \
  --dataset.single_task="Part Sorting" \
  --dataset.num_episodes=10 \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=100000000 \
  --dataset.num_image_writer_processes=4 \
  --dataset.root=datasets/eval/smolvla_20k \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.video=true \
  --play_sounds=false \
  > outputs/eval/smolvla_20k.log 2>&1
'

# 4. Đợi ~20-25s cho "调整好按Enter键开始录制" xuất hiện trong log, rồi gửi Enter:
DISPLAY=:1 XAUTHORITY=/tmp/.docker.xauth /tmp/sendkey Return

# 5. Monitor
tail -f outputs/eval/smolvla_20k.log | tr '\r' '\n' | grep -v "Warning\|carb\|HTTP"
```

**Kết quả eval checkpoint 020000 (đang chạy 2026-06-16 04:00+)**:
- Isaac Sim khởi động với `isaacsim.exp.base.python.kit` (10s, KHÔNG streaming)
- SimulationApp created → Scene USD loaded → World running → 4 cameras init → DualArmIK init → PASS
- SmolVLA inference chạy ở **~10 FPS** (vs GR00T 5-6 FPS)
- GPU: 9.1GB / 49GB (18.5%), 37% utilization
- 10 episodes đang chạy (timelimit 100 giây/episode ≈ 16-20 phút tổng)
- Dataset output: `datasets/eval/smolvla_20k_20260616_035628/`
- Log: `outputs/eval/smolvla_20k.log`

**Sau khi 020000 xong**: chạy tiếp 025000 và 030000 với cùng command
(đổi `checkpoints/020000` → `025000` / `030000` và `smolvla_20k` → `smolvla_25k` / `smolvla_30k`).
