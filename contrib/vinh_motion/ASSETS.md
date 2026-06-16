# HRC 2026 Team 3 — Assets & Downloads

All large files (checkpoints, datasets) are hosted on Hugging Face instead of Git.

## Repositories

| Type | URL |
|------|-----|
| Model checkpoints | https://huggingface.co/vinhnguyen02092005/hrc2026-outputs |
| Datasets | https://huggingface.co/datasets/vinhnguyen02092005/hrc2026-datasets |

---

## Download

> Requires `huggingface-cli` or `hf`. Install via:
> ```bash
> pip install huggingface_hub[cli]
> ```
> Then login once:
> ```bash
> hf auth login
> ```

### Download all checkpoints (~9 GB)

```bash
hf download vinhnguyen02092005/hrc2026-outputs \
    --local-dir outputs
```

### Download only the latest checkpoint (step 030000, ~1.5 GB)

```bash
hf download vinhnguyen02092005/hrc2026-outputs \
    --include "outputs/train/smolvla_part_sorting_long/checkpoints/030000/*" \
    --local-dir .
```

### Download training datasets (~9.7 GB)

```bash
hf download vinhnguyen02092005/hrc2026-datasets \
    --repo-type dataset \
    --local-dir .
```

### Download only Part_Sorting (long episodes, ~7.3 GB)

```bash
hf download vinhnguyen02092005/hrc2026-datasets \
    --repo-type dataset \
    --include "Part_Sorting/part_sorting_long_756_episode/*" \
    --local-dir .
```

### Download only Part_Sorting (short episodes, ~2.4 GB)

```bash
hf download vinhnguyen02092005/hrc2026-datasets \
    --repo-type dataset \
    --include "Part_Sorting/part_sorting_short_1000_episode/*" \
    --local-dir .
```

---

## Checkpoint Structure

```
outputs/
└── train/
    └── smolvla_part_sorting_long/
        └── checkpoints/
            ├── 005000/   # 1.5 GB
            ├── 010000/   # 1.5 GB
            ├── 015000/   # 1.5 GB
            ├── 020000/   # 1.5 GB
            ├── 025000/   # 1.5 GB
            └── 030000/   # 1.5 GB  ← latest
                ├── pretrained_model/
                └── training_state/
```

## Dataset Structure

```
Part_Sorting/
├── part_sorting_long_756_episode/    # 756 episodes, 7.3 GB
│   ├── data/
│   ├── meta/
│   └── videos/
└── part_sorting_short_1000_episode/  # 1000 episodes, 2.4 GB
    ├── data/
    ├── meta/
    └── videos/

datasets/
└── eval/                             # eval rollout recordings per model version
    ├── snap_v2/
    ├── ema_v3/
    └── ...
```

---

## Run (Isaac Sim + Docker)

### 1. Start the container

```bash
# With GUI (requires monitor connected)
./run.sh

# Headless — no monitor needed
./run.sh --headless
```

`run.sh` starts the Docker container and drops you into a bash shell inside it.
All commands below are run **inside the container**.

### 2. Open Isaac Sim GUI (simulation viewer)

```bash
# From inside the container, at the workspace root:
python Ubtech_sim/main.py
```

This opens the Isaac Sim window with the Part Sorting scene (`headless: False`).
Make sure you started the container with `./run.sh` (not `--headless`) and have a display connected.

### 3. Train inside the container

```bash
python src/lerobot/scripts/lerobot_train.py \
    --config-name smolvla_part_sorting \
    hydra.run.dir=outputs/train/smolvla_part_sorting_long
```

Add `env.headless=true` to train without opening the Isaac Sim window:

```bash
python src/lerobot/scripts/lerobot_train.py \
    --config-name smolvla_part_sorting \
    hydra.run.dir=outputs/train/smolvla_part_sorting_long \
    env.headless=true
```

### 4. Run eval with GUI — xem robot chạy thực tế trong Isaac Sim

Eval dùng `lerobot_record.py` với `/isaac-sim/python.sh` bên trong container. Chạy từ **ngoài container** (host):

**Bước 1 — Chuẩn bị X11 (1 lần):**
```bash
xauth extract /tmp/.docker.xauth :1 && chmod 644 /tmp/.docker.xauth
xhost +local:docker
```

**Bước 2 — Khởi động container với GUI:**
> Nếu container `eval_smolvla` đang chạy sẵn, bỏ qua bước này và chạy thẳng Bước 3.
> Để tạo lại từ đầu: `docker rm -f eval_smolvla`

```bash
docker rm -f eval_smolvla 2>/dev/null; docker run -d \
  --name eval_smolvla \
  --privileged --network host --user root --gpus all --shm-size=8g \
  -e DISPLAY=:1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e PYTHONPATH=/workspace/GlobalHumanoidRobotChallenge_2026_Baseline \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/.docker.xauth:/tmp/.docker.xauth:ro \
  -v /home/team3/hrc2026-team3:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline:rw \
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
```

**Bước 3 — Chạy eval checkpoint (đổi `030000` sang checkpoint muốn test):**
```bash
docker exec -d eval_smolvla bash -c '
/isaac-sim/python.sh src/lerobot/scripts/lerobot_record.py \
  --robot.type=walker_s2_sim \
  --task=Part_Sorting \
  --policy.path=outputs/train/smolvla_part_sorting_long/checkpoints/025000/pretrained_model \
  --dataset.repo_id=team3/eval_smolvla_25k \
  --dataset.single_task="Part Sorting" \
  --dataset.num_episodes=10 \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=100 \
  --dataset.num_image_writer_processes=4 \
  --dataset.root=datasets/eval/smolvla_25k \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.video=true \
  --play_sounds=false \
  > outputs/eval/smolvla_25k.log 2>&1
'
```

**Bước 4 — Đợi ~20-25s rồi gửi Enter để bắt đầu record:**
```bash
# Build sendkey tool (1 lần)
cat > /tmp/sendkey.c << 'EOF'
#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>
int main(int argc, char**argv){
    Display*d=XOpenDisplay(NULL); if(!d)return 1;
    KeyCode code=XKeysymToKeycode(d,XStringToKeysym(argv[1]));
    XTestFakeKeyEvent(d,code,True,0); XTestFakeKeyEvent(d,code,False,0);
    XFlush(d); XSync(d,False); XCloseDisplay(d); return 0;
}
EOF
gcc /tmp/sendkey.c /usr/lib/x86_64-linux-gnu/libX11.so.6 /usr/lib/x86_64-linux-gnu/libXtst.so.6 -o /tmp/sendkey

# Gửi Enter
DISPLAY=:1 XAUTHORITY=/tmp/.docker.xauth /tmp/sendkey Return
```

**Bước 5 — Theo dõi log:**
```bash
tail -f outputs/eval/smolvla_25k.log | tr '\r' '\n' | grep -v "Warning\|carb\|HTTP"
```

> Isaac Sim mất ~20-30s khởi động, sau đó GUI mở và robot bắt đầu chạy tự động.
> Kết quả (videos + metrics) lưu tại `datasets/eval/smolvla_30k/`.

---

## Upload (team use)

To push updated files back to Hugging Face:

```bash
HF=/home/team3/Isaac-GR00T/.venv/bin/hf

# Push checkpoints
$HF upload vinhnguyen02092005/hrc2026-outputs ./outputs outputs \
    --commit-message "Update checkpoints"

# Push datasets
$HF upload --repo-type dataset vinhnguyen02092005/hrc2026-datasets \
    ./Part_Sorting Part_Sorting --commit-message "Update Part_Sorting"

$HF upload --repo-type dataset vinhnguyen02092005/hrc2026-datasets \
    ./datasets datasets --commit-message "Update eval datasets"
```
