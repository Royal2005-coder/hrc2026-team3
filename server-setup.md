# Server Setup — Cài Ollama, Qwen2.5-3B & Chuẩn bị Fine-tune

> Hướng dẫn cài đặt trên máy server **không có quyền sudo**.
> Cấu hình server: GPU ~7.5 GB VRAM · RAM 24 GB · Disk 300 GB

---

## Mục lục

1. [Kiểm tra môi trường server](#1-kiểm-tra-môi-trường-server)
2. [Cài Ollama không cần sudo](#2-cài-ollama-không-cần-sudo)
3. [Tải model Qwen2.5-3B vào Ollama](#3-tải-model-qwen25-3b-vào-ollama)
4. [Cài thư viện fine-tune](#4-cài-thư-viện-fine-tune)
5. [Tải Qwen2.5-3B-Instruct từ HuggingFace](#5-tải-qwen25-3b-instruct-từ-huggingface)
6. [Pull dataset từ GitHub](#6-pull-dataset-từ-github)
7. [Kiểm tra toàn bộ trước khi fine-tune](#7-kiểm-tra-toàn-bộ-trước-khi-fine-tune)

---

## 1. Kiểm tra môi trường server

Chạy lần lượt các lệnh sau để xác nhận server đủ điều kiện.

### 1.1 Kiểm tra GPU

```bash
nvidia-smi
```

Ghi lại thông tin:
- Tên GPU: _______________
- CUDA Version: _______________
- Driver Version: _______________

### 1.2 Kiểm tra CUDA qua Python

```bash
python3 -c "
import torch
print('PyTorch  :', torch.__version__)
print('CUDA     :', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU      :', p.name)
    print('VRAM     :', round(p.total_memory / 1e9, 1), 'GB')
"
```

Kết quả mong đợi:
```
PyTorch  : 2.x.x+cuXXX
CUDA     : True
GPU      : NVIDIA ...
VRAM     : 7.5 GB
```

> ⚠️ Nếu `CUDA: False` → CUDA chưa được add vào PATH, liên hệ admin server.

### 1.3 Kiểm tra RAM và disk

```bash
free -h && df -h ~
```

### 1.4 Kiểm tra Python và pip

```bash
python3 --version && pip --version
```

---

## 2. Cài Ollama không cần sudo

### 2.1 Tải binary về thư mục user

```bash
mkdir -p ~/.local/bin
curl -fsSL https://ollama.com/download/ollama-linux-amd64 -o ~/.local/bin/ollama
chmod +x ~/.local/bin/ollama
```

### 2.2 Thêm vào PATH

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Kiểm tra
ollama --version
```

### 2.3 Chạy Ollama serve trong tmux

Vì không có sudo nên không tạo được systemd service — dùng `tmux` để giữ process chạy ngầm:

```bash
# Tạo session mới
tmux new -s ollama

# Trong session, chạy:
ollama serve

# Nhấn Ctrl+B rồi D để detach (process vẫn chạy ngầm)
```

### 2.4 Kiểm tra Ollama đang chạy

```bash
curl http://localhost:11434
# Kết quả mong đợi: Ollama is running
```

---

## 3. Tải model Qwen2.5-3B vào Ollama

```bash
ollama pull qwen2.5:3b
```

Ollama tự detect GPU và load lên VRAM khi chạy inference.

```bash
# Kiểm tra
ollama list

# Test thử
ollama run qwen2.5:3b "Xin chào!"
```

---

## 4. Cài thư viện fine-tune

Tất cả cài vào user space — không cần sudo.

### 4.1 Cài thư viện

```bash
pip install \
    transformers \
    peft \
    trl \
    accelerate \
    datasets \
    bitsandbytes \
    huggingface_hub \
    scipy \
    sentencepiece \
    protobuf
```

> **Lưu ý:** `bitsandbytes` hoạt động được vì server có GPU CUDA — đây là thư viện cốt lõi cho phép QLoRA 4-bit quantization khi fine-tune.

### 4.2 Kiểm tra sau khi cài

```bash
python3 -c "
import transformers, peft, trl, accelerate, bitsandbytes, datasets
import torch

print('transformers :', transformers.__version__)
print('peft         :', peft.__version__)
print('trl          :', trl.__version__)
print('bitsandbytes :', bitsandbytes.__version__)
print('CUDA ready   :', torch.cuda.is_available())
print()
print('✅ Sẵn sàng fine-tune!' if torch.cuda.is_available() else '❌ Chưa có CUDA')
"
```

---

## 5. Tải Qwen2.5-3B-Instruct từ HuggingFace

Model HuggingFace format dùng để fine-tune — khác với model GGUF trong Ollama dùng để inference.

### 5.1 Đăng nhập HuggingFace (nếu cần)

```bash
# Qwen2.5 là model public, thường không cần token
# Nếu bị yêu cầu thì chạy lệnh này và nhập token
huggingface-cli login
```

### 5.2 Tải model (~6.5 GB)

```bash
python3 -c "
from huggingface_hub import snapshot_download

print('Đang tải Qwen/Qwen2.5-3B-Instruct (~6.5 GB)...')
snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct',
    local_dir='./qwen2.5-3b-base',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.pt', '*.bin', 'original/*'],
)
print('✅ Tải xong! Lưu tại ./qwen2.5-3b-base')
"
```

### 5.3 Kiểm tra sau khi tải

```bash
ls ./qwen2.5-3b-base/
# Phải thấy: config.json, tokenizer.json, model-*.safetensors, ...

du -sh ./qwen2.5-3b-base/
# Kích thước khoảng ~6–7 GB
```

---

## 6. Pull dataset từ GitHub

```bash
# Clone repo về server
git clone <your-github-repo-url> ./Cadebot

# Hoặc nếu đã có repo, chỉ pull mới nhất
cd ./Cadebot && git pull
```

### Kiểm tra dataset

```bash
ls ./Cadebot/dataset/
# Phải thấy: train.jsonl  val.jsonl  menu_knowledge.json

wc -l ./Cadebot/dataset/train.jsonl
# Kết quả mong đợi: 144

wc -l ./Cadebot/dataset/val.jsonl
# Kết quả mong đợi: 26
```

---

## 7. Kiểm tra toàn bộ trước khi fine-tune

Chạy script tổng hợp dưới đây — nếu tất cả ✅ thì sẵn sàng fine-tune:

```bash
python3 - << 'EOF'
import os, torch

print("=" * 55)
print("  KIỂM TRA MÔI TRƯỜNG FINE-TUNE")
print("=" * 55)

results = []

# 1. GPU + CUDA
print("\n[1] GPU & CUDA")
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    p = torch.cuda.get_device_properties(0)
    vram = round(p.total_memory / 1e9, 1)
    vram_free = round(torch.cuda.mem_get_info()[0] / 1e9, 1)
    print(f"    GPU       : {p.name}")
    print(f"    VRAM tổng : {vram} GB")
    print(f"    VRAM free : {vram_free} GB")
    vram_ok = vram_free >= 4.5
    print(f"    Đủ VRAM   : {'✅' if vram_ok else '❌ cần ít nhất 4.5 GB'}")
    results.append(("GPU + CUDA", cuda_ok and vram_ok))
else:
    print("    CUDA : ❌ không khả dụng")
    results.append(("GPU + CUDA", False))

# 2. Thư viện
print("\n[2] Thư viện fine-tune")
libs = {
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "accelerate": "accelerate",
}
libs_ok = True
for name, mod in libs.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"    {name:<15}: ✅ {ver}")
    except ImportError:
        print(f"    {name:<15}: ❌ chưa cài")
        libs_ok = False
results.append(("Thư viện", libs_ok))

# 3. Ollama
print("\n[3] Ollama")
import subprocess
r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
ollama_ok = r.returncode == 0
print(f"    ollama list : {'✅' if ollama_ok else '❌ ollama chưa chạy'}")
if ollama_ok:
    has_qwen = "qwen2.5:3b" in r.stdout
    print(f"    qwen2.5:3b  : {'✅' if has_qwen else '❌ chưa pull'}")
    results.append(("Ollama + model", has_qwen))
else:
    results.append(("Ollama + model", False))

# 4. Model HuggingFace
print("\n[4] Model HuggingFace (để fine-tune)")
model_path = "./qwen2.5-3b-base"
model_ok = os.path.isfile(f"{model_path}/config.json")
if model_ok:
    size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(model_path) for f in fs
    )
    print(f"    Đường dẫn  : {model_path}")
    print(f"    Kích thước  : {round(size/1e9, 1)} GB  ✅")
else:
    print(f"    {model_path} : ❌ chưa tải")
results.append(("Model HuggingFace", model_ok))

# 5. Dataset
print("\n[5] Dataset")
train_path = "./Cadebot/dataset/train.jsonl"
val_path   = "./Cadebot/dataset/val.jsonl"
dataset_ok = True
for label, path in [("train", train_path), ("val", val_path)]:
    if os.path.isfile(path):
        with open(path) as f:
            n = sum(1 for _ in f)
        print(f"    {label:<6}: {n} mẫu  ✅")
    else:
        print(f"    {label:<6}: ❌ không tìm thấy ({path})")
        dataset_ok = False
results.append(("Dataset", dataset_ok))

# Tổng kết
print("\n" + "=" * 55)
all_ok = all(ok for _, ok in results)
for name, ok in results:
    print(f"  {'✅' if ok else '❌'} {name}")
print("=" * 55)
if all_ok:
    print("  🎉 Tất cả sẵn sàng — có thể bắt đầu fine-tune!")
else:
    print("  ⚠️  Hoàn thành các mục ❌ trước khi fine-tune.")
print("=" * 55)
EOF
```

---

## Checklist tổng hợp

| # | Hạng mục | Lệnh kiểm tra nhanh | Trạng thái |
|---|----------|-------------------|-----------|
| 1 | GPU nhận diện được | `nvidia-smi` | ⬜ |
| 2 | PyTorch thấy CUDA | `python3 -c "import torch; print(torch.cuda.is_available())"` | ⬜ |
| 3 | Ollama cài xong | `ollama --version` | ⬜ |
| 4 | Ollama serve đang chạy | `curl localhost:11434` | ⬜ |
| 5 | Qwen2.5-3B trong Ollama | `ollama list` | ⬜ |
| 6 | Thư viện fine-tune đủ | `python3 -c "import peft, trl, bitsandbytes"` | ⬜ |
| 7 | Model HuggingFace tải xong | `ls ./qwen2.5-3b-base/config.json` | ⬜ |
| 8 | Dataset pull từ GitHub xong | `wc -l ./Cadebot/dataset/train.jsonl` | ⬜ |
| 9 | Script kiểm tra tổng hợp pass | Tất cả ✅ | ⬜ |

**→ Tick đủ 9 mục → sẵn sàng chạy fine-tune.**

---

## Ghi chú lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|-----------|
| `CUDA: False` sau khi `import torch` | CUDA chưa trong PATH | Kiểm tra `echo $LD_LIBRARY_PATH`, hỏi admin |
| `ollama: command not found` | PATH chưa reload | Chạy `source ~/.bashrc` |
| `bitsandbytes: CUDA error` | Version không khớp | `pip install bitsandbytes --upgrade` |
| `curl: (7) Connection refused` | Ollama chưa serve | Vào tmux: `tmux a -t ollama`, chạy lại `ollama serve` |
| `401 Unauthorized` khi tải HuggingFace | Cần token | `huggingface-cli login` |
| Tải model bị ngắt giữa chừng | Mất kết nối | Chạy lại lệnh tải — `snapshot_download` tự resume |
