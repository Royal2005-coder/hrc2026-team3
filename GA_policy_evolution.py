"""
Evolution Strategies (ES kiểu OpenAI-ES, Salimans et al. 2017) cho Policy của
Part Sorting Long, dùng chung 1 simulation context (parallel-env).

Lịch sử: bản trước dùng GA cổ điển — mã hoá mỗi trọng số thành chuỗi 10 chữ số
0-9 rồi giải mã về [-1, 1], cộng thêm selection/crossover/mutation rời rạc.
Sau ~200 thế hệ best/mean/worst gần như đứng yên (best ~ -230, mean ~ -246.9):
quần thể hành xử như random search chứ không tiến hoá, vì:
  1. Ánh xạ kiểu gen -> kiểu hình quá "gồ ghề": đổi 1 chữ số ở hàng cao làm
     trọng số nhảy ~0.2 trên thang [-1, 1] — không có khái niệm "lân cận tốt
     hơn một chút" để leo dốc.
  2. mutation_rate=0.01 nhưng mỗi cá thể có npar*10 ~ 553.800 gen -> hàng nghìn
     trọng số bị random lại mỗi thế hệ; spike thích nghi (0.25 khi stagnation)
     còn random hoá tới ~25% gen — gần như reset lại quần thể theo chu kỳ.
  3. Crossover 1 điểm trên vector phẳng cắt ngang ma trận trọng số ở vị trí
     ngẫu nhiên -> ghép 2 mạng "xa lạ" thành mạng vô nghĩa (competing
     conventions, vấn đề kinh điển của neuroevolution).
  4. Không gian tìm kiếm 55.380 chiều với quần thể 100 cá thể là quá lớn cho
     GA cổ điển + roulette-wheel selection trên fitness có range hẹp & nhiễu.

ES giải quyết đúng các điểm trên bằng cách bỏ hẳn quần thể đa dạng + selection
rời rạc: chỉ giữ MỘT vector trung tâm theta (= trọng số Policy thực, số thực),
mỗi thế hệ sinh các bản sao nhiễu Gaussian quanh theta, đánh giá fitness, rồi
cập nhật theta theo ước lượng natural-gradient (trung bình có trọng số của các
hướng nhiễu theo fitness đã chuẩn hoá). Cách này đã được chứng minh scale tới
hàng triệu tham số (Atari/MuJoCo) — đúng lớp bài toán đang gặp ở đây.

Hạ tầng tham chiếu:
  - robot_arm_training/part_sorting_long_rl/env_cfg.py:145  -> scene.num_envs (512 hoặc 4096)
  - robot_arm_training/part_sorting_long_rl/agent_cfg.py    -> POLICY_NET_ARCH, obs/act space
  - robot_arm_training/part_sorting_long_rl/train.py        -> class Policy (kiến trúc mạng)

Ý tưởng hạ tầng song song (giữ nguyên so với bản GA): KHÔNG mở nhiều simulation
context riêng. Mở đúng 1 context với NUM_ENVS env (vd 4096), chia đều cho
POP_SIZE bản sao nhiễu, mỗi bản sao dùng dải env riêng để rollout bằng bộ
trọng số theta + sigma * epsilon của chính nó, rồi cộng reward trong dải đó
lại thành fitness.
"""

from __future__ import annotations

import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=300, suppress=True)

# ============================================================
# 0. HẠ TẦNG PARALLEL-ENV DÙNG CHUNG
# ============================================================
NUM_ENVS = 4096   # env_cfg.py:145 (PartSortingEnvCfg.scene.num_envs) — đổi thành 512 nếu chạy debug
POP_SIZE = 100    # số bản sao nhiễu chạy song song trong CÙNG 1 simulation context (cần là số chẵn — mirrored sampling)

# Kiến trúc mạng Policy lấy đúng từ agent_cfg.py / train.py (không đổi để fitness so sánh được)
OBS_DIM = 52              # PartSortingEnvCfg.observation_space
ACT_DIM = 10              # PartSortingEnvCfg.action_space
HIDDEN  = [256, 128, 64]  # agent_cfg.POLICY_NET_ARCH


def split_envs(num_envs: int, pop_size: int) -> list[np.ndarray]:
    """Chia đều num_envs cho pop_size bản sao, mỗi bản sao nhận 1 dải env_id liên tiếp.

    VD: 4096 / 100 -> 96 bản sao nhận 41 env, 4 bản sao nhận 40 env (rải đều phần dư),
    tổng vẫn khớp NUM_ENVS nên không có env nào "thừa" hay bị bỏ sót.
    """
    base, rem = divmod(num_envs, pop_size)
    sizes = np.full(pop_size, base, dtype=int)
    sizes[:rem] += 1
    bounds = np.concatenate(([0], np.cumsum(sizes)))
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(pop_size)]


def count_policy_params(obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: list[int] = HIDDEN) -> int:
    """Tổng số tham số (weight + bias) của mạng Policy trong train.py:
    [Linear -> activation] x len(hidden)  ->  mean_layer (Linear)  ->  log_std (Parameter)."""
    total, in_dim = 0, obs_dim
    for out_dim in hidden:
        total += in_dim * out_dim + out_dim   # nn.Linear(in_dim, out_dim): weight + bias
        in_dim = out_dim
    total += in_dim * act_dim + act_dim        # mean_layer
    total += act_dim                            # log_std (nn.Parameter, shape = (act_dim,))
    return total


# ============================================================
# 1. THAM SỐ ES
# ============================================================
pop_size = POP_SIZE
npar     = count_policy_params()   # số chiều của theta = tổng tham số mạng Policy (= 55_380)

# Mặc định cho run_evolution() — có thể override khi gọi:
SIGMA          = 0.05    # độ lệch chuẩn nhiễu Gaussian quanh theta (cùng bậc với độ lớn trọng số init của PyTorch)
LEARNING_RATE  = 0.02    # tốc độ cập nhật theta theo hướng gradient ước lượng
WEIGHT_DECAY   = 0.005   # hệ số L2 lên theta — chống ||theta|| trôi dạt không kiểm soát (khuyến nghị OpenAI-ES)


# ============================================================
# 2. CHUYỂN ĐỔI genome (vector số thực, length == npar) <-> state_dict của Policy
#    Không còn bước "giải mã digit-string" như bản GA cũ — genome CHÍNH LÀ
#    vector trọng số thực, ánh xạ kiểu gen -> kiểu hình do đó MƯỢT (thay đổi nhỏ
#    trong genome -> thay đổi nhỏ trong hành vi mạng), điều kiện cần để ES leo dốc.
# ============================================================
def genome_to_state_dict(genome: np.ndarray,
                         obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                         hidden: list[int] = HIDDEN) -> dict[str, np.ndarray]:
    """Cắt 1 vector trọng số thực (length == npar) thành state_dict khớp 1-1
    với class Policy trong train.py (net.{0,2,4}.{weight,bias}, mean_layer.*, log_std).

    Dùng để nạp trực tiếp vào model: model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
    """
    assert genome.shape == (npar,)
    sd: dict[str, np.ndarray] = {}
    ptr, in_dim = 0, obs_dim
    for i, out_dim in enumerate(hidden):
        layer_idx = 2 * i  # nn.Sequential xen kẽ Linear/activation -> Linear nằm ở chỉ số chẵn
        w = in_dim * out_dim
        sd[f"net.{layer_idx}.weight"] = genome[ptr:ptr + w].reshape(out_dim, in_dim); ptr += w
        sd[f"net.{layer_idx}.bias"]   = genome[ptr:ptr + out_dim].copy();             ptr += out_dim
        in_dim = out_dim

    w = in_dim * act_dim
    sd["mean_layer.weight"] = genome[ptr:ptr + w].reshape(act_dim, in_dim); ptr += w
    sd["mean_layer.bias"]   = genome[ptr:ptr + act_dim].copy();             ptr += act_dim
    sd["log_std"]           = genome[ptr:ptr + act_dim].copy();             ptr += act_dim

    assert ptr == npar
    return sd


def state_dict_to_genome(sd: dict, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                         hidden: list[int] = HIDDEN) -> np.ndarray:
    """Chiều ngược lại của genome_to_state_dict — duỗi state_dict (numpy hoặc
    torch.Tensor) thành 1 vector (npar,) theo ĐÚNG thứ tự đã cắt ở trên.

    Dùng để khởi tạo theta từ chính bộ trọng số init (Kaiming/Xavier) mà
    PyTorch đã gán cho policy_model — tận dụng 1 init scheme hợp lý về độ lớn,
    thay vì sinh theta hoàn toàn ngẫu nhiên.
    """
    def _np(v):
        return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

    parts = []
    in_dim = obs_dim
    for i, out_dim in enumerate(hidden):
        layer_idx = 2 * i
        parts.append(_np(sd[f"net.{layer_idx}.weight"]).reshape(-1))
        parts.append(_np(sd[f"net.{layer_idx}.bias"]).reshape(-1))
        in_dim = out_dim
    parts.append(_np(sd["mean_layer.weight"]).reshape(-1))
    parts.append(_np(sd["mean_layer.bias"]).reshape(-1))
    parts.append(_np(sd["log_std"]).reshape(-1))

    genome = np.concatenate(parts).astype(np.float64)
    assert genome.shape == (npar,)
    return genome


# ============================================================
# 3. ĐÁNH GIÁ FITNESS — chạy CHUNG 1 simulation context, mỗi bản sao nhiễu
#    dùng dải env riêng (giữ nguyên tinh thần "share hạ tầng parallel-env" của bản GA)
# ============================================================
def evaluate_population(pop: np.ndarray, env, policy_model, env_slices: list[np.ndarray],
                        n_steps: int) -> np.ndarray:
    """Đánh giá fitness cho cả lô bản sao nhiễu trong 1 lượt rollout chung.

    pop          : (n, npar) — mỗi hàng là 1 vector trọng số thực (theta + sigma*epsilon)
    env          : PartSortingEnv đã khởi tạo với scene.num_envs == NUM_ENVS
    policy_model : 1 instance Policy (train.py) dùng làm "khung", nạp lại trọng số
                   của từng bản sao trước khi forward dải env tương ứng
    env_slices   : kết quả split_envs(NUM_ENVS, len(pop)) — env_slices[i] là các env_id
                   thuộc về bản sao i

    Mỗi bước mô phỏng:
      1. Với mỗi bản sao i: nạp pop[i] qua genome_to_state_dict() vào policy_model,
         forward CHỈ các quan sát thuộc env_slices[i] -> action cho dải env đó
      2. Ghép action của tất cả bản sao thành 1 tensor (NUM_ENVS, ACT_DIM) và env.step()
         MỘT LẦN cho toàn bộ context (đúng tinh thần "share hạ tầng parallel-env")
      3. Cộng dồn reward theo từng dải env_slices[i] qua n_steps bước -> fitness[i]
    """
    import torch  # chỉ cần khi chạy thật trong IsaacLab runtime

    n = len(pop)
    fitness = np.zeros(n, dtype=np.float64)

    obs, _ = env.reset()
    for _ in range(n_steps):
        # Dùng env.num_envs thực tế (không phải hằng số NUM_ENVS) — cho phép chạy với
        # --num_envs khác (vd 512 khi debug) mà không lệch shape khi env.step()
        actions = torch.zeros((env.num_envs, ACT_DIM), device=env.device)
        for i, ids in enumerate(env_slices):
            sd = genome_to_state_dict(pop[i])
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
# 4. VÒNG LẶP ES — lặp qua các thế hệ:
#      sinh nhiễu quanh theta -> đánh giá -> chuẩn hoá fitness (centered rank) ->
#      ước lượng natural-gradient -> cập nhật theta
# ============================================================
def run_evolution(env, policy_model, num_generations: int = 1000, n_steps: int = 24,
                  log_every: int = 1, save_dir: str | None = None,
                  sigma: float = SIGMA, sigma_decay: float = 1.0, sigma_min: float = 0.01,
                  learning_rate: float = LEARNING_RATE, lr_decay: float = 1.0, lr_min: float = 0.005,
                  weight_decay: float = WEIGHT_DECAY,
                  antithetic: bool = True):
    """Train Policy bằng Evolution Strategies (OpenAI-ES, Salimans et al. 2017)
    qua `num_generations` thế hệ trên CHUNG 1 simulation context.

    Khác với GA cổ điển: không có quần thể đa dạng + selection/crossover/mutation
    rời rạc. Chỉ có MỘT vector trung tâm `theta` (= trọng số Policy thực). Mỗi thế hệ:
      1. Sinh `pop_size` bản sao nhiễu quanh theta:
           epsilon ~ N(0, I),  pop[i] = theta + sigma * epsilon[i]
         (mirrored/antithetic sampling: dùng cả +epsilon và -epsilon để giảm
         phương sai của ước lượng gradient — kỹ thuật chuẩn của OpenAI-ES)
      2. evaluate_population() — rollout n_steps bước, mỗi bản sao dùng dải env
         riêng -> fitness (NHƯ bản GA, hạ tầng song song env giữ nguyên)
      3. Centered rank shaping: chuyển fitness thô (range hẹp, nhiễu do vị trí
         scatter ngẫu nhiên) thành rank chuẩn hoá trong [-0.5, 0.5] — chống
         outlier & làm phẳng thang đo trước khi dùng làm trọng số gradient
      4. Ước lượng natural-gradient:
           g = (1 / (pop_size * sigma)) * sum_i( shaped_fitness[i] * epsilon[i] )
         rồi cập nhật theta += learning_rate * (g - weight_decay * theta)
         (L2 trên theta để ||theta|| không trôi dạt không kiểm soát)

    Vì sao cách này khắc phục đúng vấn đề "best/mean/worst đứng yên" của bản GA cũ:
      - genome là vector số thực -> nhiễu nhỏ tạo thay đổi hành vi nhỏ (mượt),
        khác hẳn digit-encoding (đổi 1 chữ số = nhảy ~0.2 trên thang [-1,1]).
      - Không còn crossover cắt-ghép phá huỷ cấu trúc trọng số, không còn
        spike mutation_rate=0.25 "reset" quần thể theo chu kỳ.
      - Mọi mẫu trong lô đều đóng góp vào hướng cập nhật (không chỉ giữ lại
        "kẻ thắng" rồi bỏ phần còn lại như GA) -> tận dụng dữ liệu tốt hơn và
        trung bình hoá bớt nhiễu môi trường (vị trí scatter vật ngẫu nhiên).

    Lưu lại theta hiện tại mỗi thế hệ (nếu save_dir != None) để có thể nạp lại sau.
    `best_genome.npy` lưu bản sao nhiễu tốt nhất của lô (chỉ mang tính chẩn đoán/
    theo dõi tiến độ) — artefact chính để dùng lại policy là `theta.npy`.
    """
    import os as _os

    assert pop_size % 2 == 0, "pop_size cần là số chẵn để dùng mirrored/antithetic sampling"
    n_pairs = pop_size // 2 if antithetic else pop_size
    n_eval  = 2 * n_pairs if antithetic else n_pairs

    env_slices = split_envs(env.num_envs, n_eval)
    history: list[float] = []

    # Khởi tạo theta từ chính trọng số init (Kaiming/Xavier) mà PyTorch đã gán cho
    # policy_model — tận dụng 1 init scheme có độ lớn hợp lý, thay vì random uniform
    # qua digit như bản GA cũ (vốn làm mọi cá thể ban đầu đã ở rất xa nhau & vô nghĩa).
    init_sd = {k: v for k, v in policy_model.state_dict().items()}
    theta = state_dict_to_genome(init_sd)

    cur_sigma = sigma
    cur_lr = learning_rate

    for gen in range(num_generations):
        eps = np.random.randn(n_pairs, npar)
        noise = np.concatenate([eps, -eps], axis=0) if antithetic else eps   # (n_eval, npar)
        pop = theta[None, :] + cur_sigma * noise                              # (n_eval, npar)

        fitness = evaluate_population(pop, env, policy_model, env_slices, n_steps)
        best_idx = int(np.argmax(fitness))
        best_now = float(fitness[best_idx])
        history.append(best_now)

        # Centered rank shaping (Salimans et al. 2017, mục 3) — fitness thô có range hẹp
        # & nhiễu (phụ thuộc nhiều vào vị trí scatter vật ngẫu nhiên lúc reset), rank
        # transform giúp mọi cập nhật chỉ phụ thuộc THỨ TỰ chứ không phụ thuộc thang đo
        ranks = np.argsort(np.argsort(fitness))
        shaped = ranks / (n_eval - 1) - 0.5   # trong [-0.5, 0.5], tổng ~ 0

        # Ước lượng natural-gradient hướng cải thiện kỳ vọng quanh theta
        grad = (shaped @ noise) / (n_eval * cur_sigma)
        theta = theta + cur_lr * (grad - weight_decay * theta)

        cur_sigma = max(sigma_min, cur_sigma * sigma_decay)
        cur_lr    = max(lr_min,   cur_lr   * lr_decay)

        if gen % log_every == 0:
            print(f"[ES] gen {gen:4d}/{num_generations} | best={fitness[best_idx]:.2f} | "
                  f"mean={fitness.mean():.2f} | worst={fitness.min():.2f} | "
                  f"sigma={cur_sigma:.4f} | lr={cur_lr:.4f} | ||theta||={np.linalg.norm(theta):.2f}")

        if save_dir is not None:
            _os.makedirs(save_dir, exist_ok=True)
            np.save(_os.path.join(save_dir, "theta.npy"), theta)
            np.save(_os.path.join(save_dir, "best_genome.npy"), pop[best_idx])
            np.save(_os.path.join(save_dir, "fitness_history.npy"), np.array(history))

    return theta, history


# ==========================================
# IN MA TRẬN TEST (tương tự bản GA cũ, chạy độc lập không cần IsaacLab)
# ==========================================
if __name__ == "__main__":
    print("0. HẠ TẦNG PARALLEL-ENV:")
    slices = split_envs(NUM_ENVS, pop_size)
    sizes = [len(s) for s in slices]
    print(f"  NUM_ENVS={NUM_ENVS}, POP_SIZE={pop_size} -> {min(sizes)}~{max(sizes)} env/bản sao "
          f"(tổng = {sum(sizes)})")
    print(f"  Bản sao 0 nhận env_id {slices[0][0]}..{slices[0][-1]} ({len(slices[0])} env)")
    print("-" * 50)

    print("1. THAM SỐ MẠNG POLICY:")
    print(f"  npar = tổng tham số mạng Policy ({OBS_DIM}->{HIDDEN}->{ACT_DIM}) = {npar}")
    print(f"  sigma={SIGMA}, learning_rate={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}")
    print("-" * 50)

    print("2. SINH QUẦN THỂ THỬ NGHIỆM QUANH theta (mirrored/antithetic sampling):")
    theta = np.random.randn(npar) * 0.1
    n_pairs = pop_size // 2
    eps = np.random.randn(n_pairs, npar)
    noise = np.concatenate([eps, -eps], axis=0)
    pop = theta[None, :] + SIGMA * noise
    print(f"  Kích thước lô thử nghiệm: {pop.shape} => ({pop.shape[0]} bản sao, {npar} trọng số thực)")
    print(f"  Bản sao 0, 5 trọng số đầu: {np.round(pop[0, :5], 6)}")

    sd0 = genome_to_state_dict(pop[0])
    print("  state_dict tương ứng (khớp class Policy trong train.py):")
    for k, v in sd0.items():
        print(f"    {k:<18} shape={v.shape}")

    genome_back = state_dict_to_genome({k: v for k, v in sd0.items()})
    print(f"  Kiểm tra round-trip genome -> state_dict -> genome: khớp = {np.allclose(genome_back, pop[0])}")
    print("-" * 50)

    print("3. CENTERED RANK SHAPING (fitness mẫu, 5 bản sao đầu = ...):")
    fitness = np.random.rand(pop_size) * 100
    print(f"  fitness thô (5 đầu) = {np.round(fitness[:5], 2)}")
    ranks = np.argsort(np.argsort(fitness))
    shaped = ranks / (pop_size - 1) - 0.5
    print(f"  rank chuẩn hoá (5 đầu) = {np.round(shaped[:5], 3)}  (trong [-0.5, 0.5], tổng ~ 0)")
    print("-" * 50)

    print("4. ƯỚC LƯỢNG NATURAL-GRADIENT & CẬP NHẬT theta:")
    grad = (shaped @ noise) / (pop_size * SIGMA)
    new_theta = theta + LEARNING_RATE * (grad - WEIGHT_DECAY * theta)
    print(f"  ||grad||      = {np.linalg.norm(grad):.6f}")
    print(f"  ||theta||     = {np.linalg.norm(theta):.4f} -> ||theta_new|| = {np.linalg.norm(new_theta):.4f}")
