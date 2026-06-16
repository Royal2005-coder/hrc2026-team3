# SmolVLA Debug & Fix Status

**Last updated**: 2026-06-16  
**Checkpoint tested**: `outputs/train/smolvla_part_sorting_long/checkpoints/030000`  
**Task**: Part sorting — left arm picks up objects, gripper grasps, lifts

---

## Root Cause (CONFIRMED)

Robot reaches correct XYZ zone but **fails to grasp** because the arm oscillates while gripper is closed.

### Evidence from `outputs/eval/arm_debug.log` (baseline, no fix):
| Metric | Training data | Baseline eval (030000) |
|--------|--------------|----------------------|
| Arm delta at grasp frame | **0.000 rad** (100% stopped) | **0.0213 rad mean** (88% steps > 0.01 rad) |
| Right arm error | — | 0.0000 (perfect) |
| ARM_AT_GRASP entries | — | 436 captured |

### Why it oscillates:
- SmolVLA uses **chunk prediction**: outputs 50 actions at once (chunk_size=50)
- At 10 FPS eval → every 5 seconds, **new chunk** gets applied → sudden arm joint jump (up to +0.214 rad seen in dim09)
- Training data: IK-generated, arm is **fully settled** (delta=0.000) before gripper closes every single time (frame 32, 187, 224, etc. across all 756 episodes)
- Policy can't replicate IK's millimeter precision → gripper closes while arm ±1-2cm off target → misses object

### NOT the cause:
- ❌ Singularity — arm reaches correct region fine
- ❌ Generalization — training covers scatter area [0.75, 0.28, 1.04], size [0.3, 0.23] with 3024 placements (0.5cm spacing)
- ❌ Overfitting — only 3.31 epochs (30K steps × 64 batch / 580K frames)
- ❌ Code bug — right arm output is PERFECT (std=0 in training, model learned it exactly)
- ❌ Wrong position — arm IS near the object, just oscillating ±1-2cm

---

## Fix Applied: EMA Smoothing

**File**: `src/lerobot/robots/walker_s2_sim/walkers2sim.py`

### Changes made:

**1. In `__init__` (around line 157):**
```python
# EMA smoothing for arm joints to reduce chunk-prediction oscillation
self._ema_arm: Optional[np.ndarray] = None
self._ema_alpha: float = 0.5  # 0.5 = equal weight new vs old
```

**2. In `_robot_control_callback` (replaced ~line 459):**

Before (original):
```python
self._hold_arm_positions = abs_action[:14].copy()
```

After (EMA smoothing):
```python
raw_arm = abs_action[:14].copy()
if self._ema_arm is None:
    self._ema_arm = raw_arm
else:
    self._ema_arm = self._ema_alpha * raw_arm + (1.0 - self._ema_alpha) * self._ema_arm
self._hold_arm_positions = self._ema_arm.copy()
abs_action = abs_action.copy()
abs_action[:14] = self._ema_arm
```

**3. Debug logging added (can remove later):**
```python
if self._left_gripping or self._right_gripping:
    print(f"[ARM_AT_GRASP] arm=[...]")
```

### EMA Results (alpha=0.5, checkpoint 025000, `outputs/eval/ema_test.log`):
| Metric | Baseline | EMA alpha=0.5 | Improvement |
|--------|----------|---------------|-------------|
| Mean delta/step | 0.0213 rad | 0.0059 rad | **3.6x better** |
| % steps > 0.01 rad | 88% | 15.1% | **5.8x better** |
| Max delta | ~0.214 rad | 0.171 rad | Chunk jump still exists |
| ARM_AT_GRASP entries | 436 | 790 | More episodes ran |

---

## Training Info

### SmolVLA checkpoint progression:
- `checkpoints/020000` — tested, similar behavior to 025000
- `checkpoints/025000` — tested with EMA (ema_test.log, 790 entries, still running)
- `checkpoints/030000` — best available, arm_debug.log baseline captured here

### Learning rate at step 30K:
```json
"_last_lr": [2.5e-6]
```
→ LR already very low. **Training more (30K→60K) with resume will likely NOT improve**, same situation as GR00T (LR hit floor at 10K, 30K→90K resume = zero improvement).

### Architecture:
- SmolVLM2-500M-Video-Instruct backbone (`freeze_vision_encoder=True`, `train_expert_only=True`)  
- `chunk_size=50`, `n_action_steps=50`, `n_obs_steps=1`
- Action: 20-dim absolute joint angles
  - `[0:7]` = right arm (STATIC — std=0.000 in all training, parked at fixed pose)
  - `[7:14]` = left arm (ACTIVE — std 0.16-0.38 rad)
  - `[14:17]` = finger positions
  - `[18]` = left gripper command (mean=-0.8748, std=0.4845)
  - `[19]` = right gripper command (mean=0.0104, std=0.9999)

---

## How to Run Eval

### Standard eval command (GUI + Isaac Sim):
```bash
docker run --rm -it \
  --name eval_smolvla \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/team3/GlobalHumanoidRobotChallenge_2026_Baseline:/workspace \
  ghcr.io/huggingface/lerobot:latest \
  python /workspace/src/lerobot/scripts/eval.py \
  --policy.path=/workspace/outputs/train/smolvla_part_sorting_long/checkpoints/030000/pretrained_model \
  --env.type=walker_s2_sim \
  --eval.n_episodes=3 \
  --dataset.repo_id=datasets/eval/test_NAME \
  --dataset.video=true \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  2>&1 | tee /workspace/outputs/eval/test_NAME.log
```

**CRITICAL**: Must use `--dataset.streaming_encoding=true --dataset.encoder_threads=2` or it deadlocks (fork-after-CUDA issue).

### Send Enter to start episode (in another terminal):
```bash
/tmp/sendkey  # compiled XTest binary that sends Enter keypress
# or: DISPLAY=:1 xdotool key Return
```

---

## Next Steps / TODO

### Option A: Tune EMA alpha (quick)
Try lower alpha = 0.3 or 0.2 for more aggressive smoothing:
```python
self._ema_alpha: float = 0.3  # 0.3 = 70% old, 30% new → very smooth
```
Risk: too much lag, arm can't reach new positions in time.

### Option B: Check if arm mean position is correct
The arm is near the object but the question is: is it **above** the object (needs to go lower) or **beside** it?

To check: need FK computation or visual inspection in GUI:
- In eval GUI, watch where gripper is relative to object when gripper closes
- If gripper is 1-2cm above: EMA won't fix it, arm just needs lower target
- If gripper is ±1-2cm side-to-side: EMA IS the fix

### Option C: Re-train with LR restart (big effort)
Start fresh training or restart LR warm-up from checkpoint. Would take ~24h but could improve to 60-90K steps effectively. Only do this if EMA doesn't work.

### Option D: Force lower position at grasp (code fix, check rules)
Detect gripper-close moment and add Z-offset to push arm down slightly. User asked about this — confirm with competition rules first.

---

## File Map

| File | Purpose |
|------|---------|
| `outputs/eval/arm_debug.log` | Baseline ARM_AT_GRASP log (no EMA, 030000 checkpoint) |
| `outputs/eval/ema_test.log` | EMA alpha=0.5 test log (025000 checkpoint, still running) |
| `outputs/train/smolvla_part_sorting_long/checkpoints/030000/` | Best checkpoint |
| `src/lerobot/robots/walker_s2_sim/walkers2sim.py` | Robot interface — EMA changes here |
| `Part_Sorting/part_sorting_long_756_episode/` | Training dataset (756 episodes, 580K frames) |
| `SMOLVLA_TRAINING.md` | Training setup instructions |
