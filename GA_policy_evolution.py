"""
GA tiến hóa trực tiếp trọng số mạng Policy của Part Sorting Long, dùng chung
1 simulation context (parallel-env) thay vì PPO.

Hạ tầng tham chiếu:
  - robot_arm_training/part_sorting_long_rl/env_cfg.py:145  -> scene.num_envs (512 hoặc 4096)
  - robot_arm_training/part_sorting_long_rl/agent_cfg.py    -> POLICY_NET_ARCH, obs/act space
  - robot_arm_training/part_sorting_long_rl/train.py        -> class Policy (kiến trúc mạng)

Ý tưởng: KHÔNG mở 100 simulation context riêng. Thay vào đó mở đúng 1 context với
NUM_ENVS env (vd 4096), chia đều cho POP_SIZE=100 cá thể (mỗi cá thể ~41 env), mỗi
cá thể dùng dải env riêng để rollout bằng bộ trọng số của chính nó, rồi cộng reward
trong dải đó lại thành fitness.

Cấu trúc gen mỗi cá thể giống hệt GA.py (ma trận pop_size x npar x ngene), chỉ khác
ở chỗ npar = TỔNG SỐ THAM SỐ của mạng Policy (mỗi "hàng" gen mã hoá đúng 1 trọng số
bằng 10 chữ số 0-9, ghép lại rồi nội suy về khoảng min_max).
"""

from __future__ import annotations

import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=300, suppress=True)

# ============================================================
# 0. HẠ TẦNG PARALLEL-ENV DÙNG CHUNG
# ============================================================
NUM_ENVS = 4096   # env_cfg.py:145 (PartSortingEnvCfg.scene.num_envs) — đổi thành 512 nếu chạy debug
POP_SIZE = 100    # số cá thể GA chạy song song trong CÙNG 1 simulation context

# Kiến trúc mạng Policy lấy đúng từ agent_cfg.py / train.py (không đổi để fitness so sánh được)
OBS_DIM = 52              # PartSortingEnvCfg.observation_space
ACT_DIM = 10              # PartSortingEnvCfg.action_space
HIDDEN  = [256, 128, 64]  # agent_cfg.POLICY_NET_ARCH


def split_envs(num_envs: int, pop_size: int) -> list[np.ndarray]:
    """Chia đều num_envs cho pop_size cá thể, mỗi cá thể nhận 1 dải env_id liên tiếp.

    VD: 4096 / 100 -> 96 cá thể nhận 41 env, 4 cá thể nhận 40 env (rải đều phần dư),
    tổng vẫn khớp NUM_ENVS nên không có env nào "thừa" hay bị bỏ sót.
    """
    base, rem = divmod(num_envs, pop_size)
    sizes = np.full(pop_size, base, dtype=int)
    sizes[:rem] += 1
    bounds = np.concatenate(([0], np.cumsum(sizes)))
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(pop_size)]


def count_policy_params(obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: list[int] = HIDDEN) -> int:
    """Tổng số tham số (weight + bias) của mạng Policy trong train.py:
    [Linear -> ELU] x len(hidden)  ->  mean_layer (Linear)  ->  log_std (Parameter)."""
    total, in_dim = 0, obs_dim
    for out_dim in hidden:
        total += in_dim * out_dim + out_dim   # nn.Linear(in_dim, out_dim): weight + bias
        in_dim = out_dim
    total += in_dim * act_dim + act_dim        # mean_layer
    total += act_dim                            # log_std (nn.Parameter, shape = (act_dim,))
    return total


# ============================================================
# 1. THAM SỐ GA (tương tự GA.py, nhưng npar = tổng số tham số mô hình)
# ============================================================
pop_size = POP_SIZE
npar     = count_policy_params()   # mỗi cá thể "mang" toàn bộ bộ trọng số mạng Policy (= 55_380)
ngene    = 10                      # 10 đoạn gen / 1 trọng số, mỗi đoạn là 1 chữ số 0-9
min_max  = [-0.2, 0.2]             # khoảng giá trị của 1 trọng số sau khi ghép 10 đoạn gen
                                   # (thử thu hẹp từ [-1, 1] -> [-0.2, 0.2]: gần thang trọng số init
                                   # mặc định của PyTorch hơn, giảm bão hoà Tanh ở các lớp ẩn — đồng thời
                                   # giảm biên độ "nhảy" mỗi khi 1 chữ số gen bị đột biến, từ ~0.2 xuống ~0.04)
mutation_rate = 0.03               # đổi min_max thành [-10, 10] hoặc tuỳ chỉnh nếu cần biên độ lớn hơn


# ============================================================
# 2. KHỞI TẠO — mỗi cá thể là khối (npar x ngene), pop là (pop_size x npar x ngene)
# ============================================================
def pop_init(pop_size: int, npar: int, ngene: int) -> np.ndarray:
    return np.random.randint(0, 10, (pop_size, npar, ngene), dtype=np.uint8)


# ============================================================
# 3. GIẢI MÃ — ghép 10 chữ số (0-9) thành 1 số thập phân, nội suy về [min_max[0], min_max[1]]
# ============================================================
def decode(p: np.ndarray) -> np.ndarray:
    powers = 10.0 ** np.arange(ngene - 1, -1, -1, dtype=np.float64)
    decimals = np.sum(p * powers, axis=2)              # (pop_size, npar)
    max_val = (10 ** ngene) - 1
    val = min_max[0] + (decimals / max_val) * (min_max[1] - min_max[0])
    return val


def genome_to_state_dict(decoded_vector: np.ndarray,
                         obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                         hidden: list[int] = HIDDEN) -> dict[str, np.ndarray]:
    """Cắt 1 vector trọng số đã giải mã (length == npar) thành state_dict khớp 1-1
    với class Policy trong train.py (net.{0,2,4}.{weight,bias}, mean_layer.*, log_std).

    Dùng để nạp trực tiếp vào model: model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
    """
    assert decoded_vector.shape == (npar,)
    sd: dict[str, np.ndarray] = {}
    ptr, in_dim = 0, obs_dim
    for i, out_dim in enumerate(hidden):
        layer_idx = 2 * i  # nn.Sequential xen kẽ Linear/ELU -> Linear nằm ở chỉ số chẵn
        w = in_dim * out_dim
        sd[f"net.{layer_idx}.weight"] = decoded_vector[ptr:ptr + w].reshape(out_dim, in_dim); ptr += w
        sd[f"net.{layer_idx}.bias"]   = decoded_vector[ptr:ptr + out_dim].copy();             ptr += out_dim
        in_dim = out_dim

    w = in_dim * act_dim
    sd["mean_layer.weight"] = decoded_vector[ptr:ptr + w].reshape(act_dim, in_dim); ptr += w
    sd["mean_layer.bias"]   = decoded_vector[ptr:ptr + act_dim].copy();             ptr += act_dim
    sd["log_std"]           = decoded_vector[ptr:ptr + act_dim].copy();             ptr += act_dim

    assert ptr == npar
    return sd


# ============================================================
# 4. CHỌN LỌC (giống GA.py — bốc nguyên khối npar x ngene của từng cá thể)
# ============================================================
def selection(pop: np.ndarray, fitness, c) -> np.ndarray:
    fitness = np.array(fitness, dtype=np.float64)
    fitness = fitness - np.min(fitness) + 1e-6
    probs = fitness / np.sum(fitness)
    indices = np.random.choice(len(pop), size=len(pop), p=probs)
    return pop[indices]


# ============================================================
# 4b. ELITISM — giữ nguyên `n_elite` cá thể tốt nhất của thế hệ hiện tại,
#     ghi đè lên `n_elite` cá thể CUỐI (theo index) của quần thể mới.
#     Đảm bảo nghiệm tốt nhất không bao giờ bị mất qua chọn lọc/lai ghép/đột biến.
# ============================================================
ELITISM_COUNT = 5


def apply_elitism(old_pop: np.ndarray, new_pop: np.ndarray, fitness, n_elite: int = ELITISM_COUNT) -> np.ndarray:
    fitness = np.array(fitness, dtype=np.float64)
    elite_idx = np.argsort(fitness)[-n_elite:]   # n_elite cá thể có fitness cao nhất của thế hệ hiện tại
    new_pop[-n_elite:] = old_pop[elite_idx]      # bỏ n_elite cá thể cuối của quần thể mới, thay bằng elite
    return new_pop


# ============================================================
# 5. LAI GHÉP (duỗi (npar*ngene,) để cắt chéo, cuộn lại thành (npar x ngene))
# ============================================================
def crossover(pop: np.ndarray) -> np.ndarray:
    new_pop = np.copy(pop)
    flat_pop = new_pop.reshape(len(pop), -1)
    total_genes = flat_pop.shape[1]

    for i in range(0, len(flat_pop) - 1, 2):
        pt = np.random.randint(1, total_genes)
        temp1 = np.concatenate((flat_pop[i][:pt], flat_pop[i + 1][pt:]))
        temp2 = np.concatenate((flat_pop[i + 1][:pt], flat_pop[i][pt:]))
        flat_pop[i] = temp1
        flat_pop[i + 1] = temp2

    return flat_pop.reshape(pop.shape)


# ============================================================
# 6. ĐỘT BIẾN (giống GA.py — quét toàn ma trận 3D, thay số 0-9)
# ============================================================
def mutation(pop: np.ndarray, mutation_rate: float) -> np.ndarray:
    flips = np.random.rand(*pop.shape) < mutation_rate
    random_genes = np.random.randint(0, 10, size=pop.shape, dtype=np.uint8)
    pop[flips] = random_genes[flips]
    return pop


# ============================================================
# 7. ĐÁNH GIÁ FITNESS — chạy CHUNG 1 simulation context, mỗi cá thể dùng dải env riêng
#    (khung sườn — điền phần forward/env.step khi chạy trong IsaacLab runtime)
# ============================================================
def evaluate_population(pop: np.ndarray, env, env_raw, policy_model, env_slices: list[np.ndarray],
                        n_steps: int) -> np.ndarray:
    """Đánh giá fitness cho cả quần thể trong 1 lượt rollout chung.

    env          : PartSortingEnv đã khởi tạo với scene.num_envs == NUM_ENVS, BỌC qua wrap_env()
    env_raw      : chính instance PartSortingEnv đó nhưng KHÔNG bọc — cần để gọi thẳng
                   _reset_idx() (xem ghi chú "ép reset toàn bộ" bên dưới)
    policy_model : 1 instance Policy (train.py) dùng làm "khung", nạp lại trọng số
                   của từng cá thể trước khi forward dải env tương ứng
    env_slices   : kết quả split_envs(NUM_ENVS, pop_size) — env_slices[i] là các env_id
                   thuộc về cá thể i

    Mỗi bước mô phỏng:
      1. decoded = decode(pop)                       # (pop_size, npar) trọng số thực
      2. Với mỗi cá thể i: nạp decoded[i] qua genome_to_state_dict() vào policy_model,
         forward CHỈ các quan sát thuộc env_slices[i] -> action cho dải env đó
      3. Ghép action của 100 cá thể thành 1 tensor (NUM_ENVS, ACT_DIM) và env.step() MỘT LẦN
         cho toàn bộ context (đúng tinh thần "share hạ tầng parallel-env")
      4. Cộng dồn reward theo từng dải env_slices[i] qua n_steps bước -> fitness[i]
    """
    import torch  # chỉ cần khi chạy thật trong IsaacLab runtime

    decoded = decode(pop)
    fitness = np.zeros(pop_size, dtype=np.float64)

    # Ép _reset_idx() chạy cho TOÀN BỘ env trước mỗi lượt đánh giá.
    # env.reset() (DirectRLEnv) chỉ thực sự scatter lại 4 vật cho các env đang ở trạng thái
    # "done" — ngay sau khi vừa khởi tạo / vừa reset xong KHÔNG env nào "done", nên các lần
    # gọi env.reset() sau đó gần như no-op: layout (vị trí 4 vật + robot) bị đóng băng y
    # nguyên từ thế hệ đầu tiên (đã xác minh bằng debug_ga_pipeline.py — Test C: 2 lần
    # reset() liên tiếp cho vị trí vật lệch nhau đúng 0.000000m). Hệ quả: tất cả các thế hệ
    # đánh giá quần thể trên CÙNG 1 layout cố định -> phần lớn dải env hội tụ về 1 quỹ đạo
    # không phụ thuộc action (Test B), khiến fitness "đứng yên" giống hệt qua hàng trăm thế
    # hệ bất kể genome. Gọi thẳng _reset_idx() trên env GỐC (chưa bọc wrap_env) cho mọi
    # env_id để đảm bảo mỗi thế hệ thực sự bắt đầu từ 1 layout random mới.
    all_ids = torch.arange(env.num_envs, device=env.device)
    env_raw._reset_idx(all_ids)
    obs, _ = env.reset()
    for _ in range(n_steps):
        # Dùng env.num_envs thực tế (không phải hằng số NUM_ENVS) — cho phép chạy với
        # --num_envs khác (vd 512 khi debug) mà không lệch shape khi env.step()
        actions = torch.zeros((env.num_envs, ACT_DIM), device=env.device)
        for i, ids in enumerate(env_slices):
            sd = genome_to_state_dict(decoded[i])
            policy_model.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32, device=env.device)
                                          for k, v in sd.items()}, strict=True)
            with torch.no_grad():
                ids_t = torch.as_tensor(ids, device=env.device)
                actions[ids_t], _ = policy_model.compute({"states": obs[ids_t]})

        obs, rewards, terminated, truncated, _ = env.step(actions)
        rewards_np = rewards.detach().cpu().numpy()
        for i, ids in enumerate(env_slices):
            fitness[i] += rewards_np[ids].sum()

    return fitness


# ============================================================
# 8. VÒNG LẶP TRAIN — lặp qua các thế hệ: đánh giá -> chọn lọc -> lai ghép -> đột biến
# ============================================================
def run_evolution(env, env_raw, policy_model, num_generations: int = 1000, n_steps: int = 24,
                  log_every: int = 1, save_dir: str | None = None,
                  adaptive_mutation: bool = True,
                  stagnation_patience: int = 20,
                  mutation_rate_base: float = mutation_rate,
                  mutation_rate_spike: float = 0.25,
                  mutation_decay: float = 0.9,
                  elitism_count: int = ELITISM_COUNT):
    """Train GA qua `num_generations` thế hệ trên CHUNG 1 simulation context.

    Mỗi thế hệ:
      1. evaluate_population() — rollout n_steps bước, mỗi cá thể dùng dải env riêng -> fitness
      2. selection -> crossover -> mutation sinh quần thể kế tiếp
    Lưu lại gen tốt nhất mỗi thế hệ (nếu save_dir != None) để có thể nạp lại sau.

    Đột biến thích nghi (adaptive_mutation=True) — chống "kẹt ngưỡng" (stagnation):
      - Đếm số thế hệ liên tiếp mà best_fitness KHÔNG cải thiện (`stagnation_patience`).
      - Khi đếm chạm ngưỡng, "đá" mutation_rate tăng vọt lên `mutation_rate_spike`
        (vd 0.01 -> 0.25) để quần thể thoát khỏi cực trị địa phương.
      - Mỗi thế hệ sau đó, mutation_rate tự GIẢM DẦN (decay theo `mutation_decay`)
        về lại `mutation_rate_base` — tránh phá vỡ các khối gen tốt vừa tìm lại được.
      - Nếu vẫn tiếp tục chững (không cải thiện), bộ đếm đi tiếp và sẽ "đá" lại
        mỗi khi chạm bội số của `stagnation_patience`.

    Elitism (elitism_count > 0):
      - Giữ nguyên `elitism_count` cá thể tốt nhất của thế hệ hiện tại (theo fitness),
        ghi đè lên `elitism_count` cá thể CUỐI (theo index) của quần thể vừa sinh ra
        (sau selection -> crossover -> mutation) — đảm bảo nghiệm tốt nhất không bao
        giờ bị mất do may rủi của chọn lọc/lai ghép/đột biến.
    """
    import os as _os

    pop = pop_init(pop_size, npar, ngene)
    env_slices = split_envs(env.num_envs, pop_size)
    history = []

    cur_mutation_rate = mutation_rate_base
    best_ever = -np.inf
    stagnation_count = 0

    for gen in range(num_generations):
        fitness = evaluate_population(pop, env, env_raw, policy_model, env_slices, n_steps)
        best_idx = int(np.argmax(fitness))
        best_now = float(fitness[best_idx])
        history.append(best_now)

        spike_triggered = False
        if adaptive_mutation:
            if best_now > best_ever + 1e-6:
                best_ever = best_now
                stagnation_count = 0
            else:
                stagnation_count += 1
                # Chạm ngưỡng kiên nhẫn (hoặc bội số của nó nếu vẫn tiếp tục chững) -> đá tăng vọt
                if stagnation_count % stagnation_patience == 0:
                    cur_mutation_rate = mutation_rate_spike
                    spike_triggered = True

            if not spike_triggered:
                # Giảm dần về mức cơ sở mỗi thế hệ — tránh phá vỡ các khối gen tốt vừa tìm lại
                cur_mutation_rate = max(mutation_rate_base, cur_mutation_rate * mutation_decay)
        else:
            cur_mutation_rate = mutation_rate_base

        if gen % log_every == 0:
            tag = "  <-- SPIKE (thoát stagnation)" if spike_triggered else ""
            print(f"[GA] gen {gen:4d}/{num_generations} | best={fitness[best_idx]:.2f} | "
                  f"mean={fitness.mean():.2f} | worst={fitness.min():.2f} | "
                  f"mutation_rate={cur_mutation_rate:.3f} | stagnation={stagnation_count}{tag}")

        if save_dir is not None:
            _os.makedirs(save_dir, exist_ok=True)
            np.save(_os.path.join(save_dir, "best_genome.npy"), pop[best_idx])
            np.save(_os.path.join(save_dir, "fitness_history.npy"), np.array(history))

        new_pop = selection(pop, fitness, 0.5)
        new_pop = crossover(new_pop)
        new_pop = mutation(new_pop, cur_mutation_rate)
        if elitism_count > 0:
            new_pop = apply_elitism(pop, new_pop, fitness, n_elite=elitism_count)
        pop = new_pop

    return pop, history


# ==========================================
# IN MA TRẬN TEST (tương tự GA.py, chạy độc lập không cần IsaacLab)
# ==========================================
if __name__ == "__main__":
    print("0. HẠ TẦNG PARALLEL-ENV:")
    slices = split_envs(NUM_ENVS, pop_size)
    sizes = [len(s) for s in slices]
    print(f"  NUM_ENVS={NUM_ENVS}, POP_SIZE={pop_size} -> {min(sizes)}~{max(sizes)} env/cá thể "
          f"(tổng = {sum(sizes)})")
    print(f"  Cá thể 0 nhận env_id {slices[0][0]}..{slices[0][-1]} ({len(slices[0])} env)")
    print("-" * 50)

    print("1. KHỞI TẠO QUẦN THỂ:")
    pop = pop_init(pop_size, npar, ngene)
    print(f"  npar = tổng tham số mạng Policy ({OBS_DIM}->{HIDDEN}->{ACT_DIM}) = {npar}")
    print(f"  Kích thước quần thể: {pop.shape} => ({pop_size} cá thể, {npar} trọng số, {ngene} đoạn gen)")
    print(f"  Cá thể 0, 5 trọng số đầu:\n{pop[0, :5]}\n")
    print("-" * 50)

    print("2. GIẢI MÃ (giá trị trọng số trong khoảng", min_max, "):")
    val = decode(pop)
    print(f"  Kích thước kết quả: {val.shape} => ({pop_size} cá thể, {npar} giá trị thực)")
    print(f"  Cá thể 0, 5 trọng số đầu (đã giải mã): {np.round(val[0, :5], 6)}")

    sd0 = genome_to_state_dict(val[0])
    print("  state_dict tương ứng (khớp class Policy trong train.py):")
    for k, v in sd0.items():
        print(f"    {k:<18} shape={v.shape}")
    print("-" * 50)

    fitness = np.random.rand(pop_size) * 100  # placeholder — thực tế lấy từ evaluate_population()
    print(f"3. CHỌN LỌC (fitness mẫu, 5 cá thể đầu = {np.round(fitness[:5], 2)}...):")
    pop = selection(pop, fitness, 0.5)
    print(f"  Vẫn giữ nguyên form ({npar} x {ngene}) cho từng cá thể.")

    print("\n4. LAI GHÉP:")
    pop = crossover(pop)
    print(f"  Form sau lai ghép: {pop.shape}")

    print("\n5. ĐỘT BIẾN: (sẵn sàng cho generation tiếp theo)")
    pop = mutation(pop, mutation_rate)
    print(f"  Form sau đột biến: {pop.shape}")
