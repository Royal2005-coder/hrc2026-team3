import numpy as np 

np.set_printoptions(threshold=np.inf, linewidth=300, suppress=True)

pop_size = 4
npar = 3       # 3 hàng (3 Trọng số)
ngene = 10     # 10 cột (10 chữ số 0-9)
min_max = [-10, 10]
mutation_rate = 0.01

# 1. KHỞI TẠO (Đã sửa thành cấu trúc 3D: pop_size x npar x ngene)
def pop_init(pop_size, npar, ngene):
    # Trả về ma trận đúng form: 4 cá thể, mỗi cá thể 3 hàng x 10 cột
    return np.random.randint(0, 10, (pop_size, npar, ngene))

# 2. GIẢI MÃ (Không cần reshape nữa vì đầu vào đã là 3D chuẩn form)
def decode(p):
    powers = 10 ** np.arange(ngene - 1, -1, -1, dtype=np.float64) 
    
    # Tính tổng theo trục ngang (axis=2 tức là gộp 10 cột lại)
    decimals = np.sum(p * powers, axis=2) 
    
    max_val = (10 ** ngene) - 1
    val = min_max[0] + (decimals / max_val) * (min_max[1] - min_max[0])
    return val

# 3. CHỌN LỌC (Vẫn bốc nguyên khối 3x10 của con chim)
def selection(pop, fitness, c):
    fitness = np.array(fitness)
    fitness = fitness - np.min(fitness) + 1e-6 
    probs = fitness / np.sum(fitness) 
    indices = np.random.choice(len(pop), size=len(pop), p=probs)
    return pop[indices]

# 4. LAI GHÉP (Tạm duỗi thẳng ra để cắt chéo, sau đó cuộn lại thành 3x10)
def crossover(pop):
    new_pop = np.copy(pop)
    # Duỗi mỗi con chim thành 1 hàng 30 số để dễ cắt điểm ngẫu nhiên
    flat_pop = new_pop.reshape(len(pop), -1) 
    total_genes = flat_pop.shape[1]
    
    for i in range(0, len(flat_pop) - 1, 2):
        pt = np.random.randint(1, total_genes) 
        temp1 = np.concatenate((flat_pop[i][:pt], flat_pop[i+1][pt:]))
        temp2 = np.concatenate((flat_pop[i+1][:pt], flat_pop[i][pt:]))
        flat_pop[i] = temp1
        flat_pop[i+1] = temp2
        
    # Cuộn lại về hình dáng khối 3x10 ban đầu
    return flat_pop.reshape(pop.shape)

# 5. ĐỘT BIẾN (Quét qua ma trận 3D, thay số 0-9)
def mutation(pop, mutation_rate):
    flips = np.random.rand(*pop.shape) < mutation_rate
    random_genes = np.random.randint(0, 10, size=pop.shape)
    pop[flips] = random_genes[flips] 
    return pop

# ==========================================
# IN MA TRẬN TEST
# ==========================================

print("1. KHỞI TẠO QUẦN THỂ:")
pop = pop_init(pop_size, npar, ngene)
print(f"Kích thước hệ thống: {pop.shape} => (4 chim, 3 hàng, 10 cột)")
print("-" * 50)
for i, bird in enumerate(pop):
    print(f"Chim {i+1}:\n{bird}\n")

print("-" * 50)
print("2. GIẢI MÃ:")
val = decode(pop)
print(f"Kích thước kết quả: {val.shape} => (4 chim, 3 số thập phân)")
for i, v in enumerate(val):
    print(f"Bộ não chim {i+1}: {np.round(v, 6)}")
print("-" * 50)

fitness = [10, 2, 8, 1] 
print(f"3. CHỌN LỌC (Fitness = {fitness}):")
pop = selection(pop, fitness, 0.5)
print("Cấu trúc vẫn giữ nguyên form 3x10 cho từng cá thể (Bị ẩn log để đỡ dài).")

print("\n4. LAI GHÉP: (Đã xử lý giữ nguyên form 3x10 sau khi lai)")
pop = crossover(pop)
print(pop)

print("\n5. ĐỘT BIẾN: (Sẵn sàng đem đi chạy Gen mới)")
pop = mutation(pop, mutation_rate)
print(pop)