# Автори: Тал Майк, Піддубна Марія, Дмитренко Владислав, студенти груп КІ-33 та КІ-31.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Параметри
N = 100  # кількість точок
g_star = 3  # задана кількість кластерів
np.random.seed(42)

# Генеруємо N точок в R2 так, щоб вони утворювали віддалені скупчення
X, true_labels = make_blobs(
    n_samples=N,
    centers=g_star,
    cluster_std=1.5,
    center_box=(-10, 10),
    random_state=42,
)

print(f"Згенеровано {N} точок з {g_star} кластерами")
print(f"Координати перших 5 точок:")
for i in range(5):
    print(f"Точка {i+1}: ({X[i,0]:.2f}, {X[i,1]:.2f})")


def generate_membership_matrix(points, num_clusters):
    """Генерує матрицю розбиття U"""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_

    # Обчислюємо відстані від точок до центрів
    U = np.zeros((num_clusters, len(points)))

    for j in range(len(points)):
        distances = []
        for k in range(num_clusters):
            dist = np.sqrt(np.sum((points[j] - centers[k]) ** 2))
            distances.append(dist)

        # Конвертуємо відстані в коефіцієнти належності
        for k in range(num_clusters):
            if distances[k] == 0:
                U[k, j] = 1.0
            else:
                sum_part = 0
                for i in range(num_clusters):
                    if distances[i] > 0:
                        sum_part += (distances[k] / distances[i]) ** 2
                U[k, j] = (
                    1.0 / sum_part if sum_part > 0 else 1.0 / num_clusters
                )

    # Нормалізуємо щоб сума = 1
    for j in range(len(points)):
        col_sum = np.sum(U[:, j])
        if col_sum > 0:
            U[:, j] = U[:, j] / col_sum

    return U


def add_noise_to_matrix(U, noise_level):
    """Додає шум до матриці"""
    U_noisy = U.copy()
    noise = np.random.uniform(-noise_level, noise_level, U.shape)
    U_noisy += noise
    U_noisy = np.clip(U_noisy, 0, 1)  # обмежуємо [0,1]

    # Перенормалізуємо
    for j in range(U.shape[1]):
        col_sum = np.sum(U_noisy[:, j])
        if col_sum > 0:
            U_noisy[:, j] = U_noisy[:, j] / col_sum

    return U_noisy


def calculate_PC(U):
    """PC = сума всіх u^2 / N"""
    return np.sum(U**2) / U.shape[1]


def calculate_CI(U):
    """CI = (g*PC - 1) / (g - 1)"""
    g = U.shape[0]
    if g == 1:
        return 1.0
    PC = calculate_PC(U)
    CI = (g * PC - 1) / (g - 1)
    return CI


def print_matrix(U, title, max_show=8):
    """Друкує матрицю"""
    print(f"\n{title}")
    print("=" * 50)
    g, N = U.shape
    print(f"Розмір: {g} кластерів x {N} точок")

    show_points = min(max_show, N)
    print(f"\nПерші {show_points} точок:")

    # Заголовок
    header = "Кластер "
    for j in range(show_points):
        header += f"   Т{j+1:02d} "
    print(header)
    print("-" * len(header))

    # Рядки матриці
    for k in range(g):
        row = f"   {k+1}    "
        for j in range(show_points):
            row += f" {U[k,j]:5.3f}"
        print(row)

    print("-" * len(header))

    # Сума по стовпцях
    sum_row = "Сума    "
    for j in range(show_points):
        col_sum = np.sum(U[:, j])
        sum_row += f" {col_sum:5.3f}"
    print(sum_row)

    # Статистика
    print(f"\nСтатистика матриці:")
    print(f"Середнє значення: {np.mean(U):.4f}")
    print(f"Мін: {np.min(U):.4f}, Макс: {np.max(U):.4f}")


# 1. Еталонна кластеризація U*
print("\n" + "=" * 60)
print("ГЕНЕРАЦІЯ МАТРИЦЬ РОЗБИТТЯ")
print("=" * 60)

U_star = generate_membership_matrix(X, g_star)
print_matrix(U_star, "ЕТАЛОННА МАТРИЦЯ U*")

PC_star = calculate_PC(U_star)
CI_star = calculate_CI(U_star)
print(f"PC = {PC_star:.4f}")
print(f"CI = {CI_star:.4f}")

# 2. Зашумлені кластеризації
print("\n" + "=" * 60)
print("ЗАШУМЛЕНІ КЛАСТЕРИЗАЦІЇ")
print("=" * 60)

results = []
results.append(("Еталонна U*", U_star, CI_star, PC_star, g_star))

noise_levels = [0.05, 0.1, 0.2]
for noise in noise_levels:
    U_noisy = add_noise_to_matrix(U_star, noise)
    PC_noisy = calculate_PC(U_noisy)
    CI_noisy = calculate_CI(U_noisy)
    results.append((f"Шум {noise}", U_noisy, CI_noisy, PC_noisy, g_star))

    if noise == 0.1:  # показуємо одну зашумлену матрицю
        print_matrix(U_noisy, f"ЗАШУМЛЕНА МАТРИЦЯ (шум = {noise})")
        print(f"PC = {PC_noisy:.4f}")
        print(f"CI = {CI_noisy:.4f}")

# 3. Різна кількість кластерів
print("\n" + "=" * 60)
print("РІЗНА КІЛЬКІСТЬ КЛАСТЕРІВ")
print("=" * 60)

for g in [2, 4, 5]:
    if g != g_star:
        U_diff = generate_membership_matrix(X, g)
        PC_diff = calculate_PC(U_diff)
        CI_diff = calculate_CI(U_diff)
        results.append((f"g = {g}", U_diff, CI_diff, PC_diff, g))

        if g == 4:  # показуємо одну матрицю з іншим g
            print_matrix(U_diff, f"МАТРИЦЯ З g = {g} (замість g* = {g_star})")
            print(f"PC = {PC_diff:.4f}")
            print(f"CI = {CI_diff:.4f}")

# Таблиця результатів
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТИ КЛАСТЕРИЗАЦІЇ")
print("=" * 60)
print(f"{'Тип':<20} {'CI':<8} {'PC':<8} {'g':<3}")
print("-" * 40)

for name, U, CI, PC, g in results:
    print(f"{name:<20} {CI:<8.4f} {PC:<8.4f} {g:<3}")

# Знаходимо найкращий CI
best_CI = max(results, key=lambda x: x[2])
print("-" * 40)
print(f"Найкращий CI: {best_CI[2]:.4f} ({best_CI[0]})")

# Показати що на U* індекс CI найбільший
print(f"\nПеревірка: еталонна U* дає найвищий CI = {CI_star:.4f}")
all_CI = [r[2] for r in results]
is_best = CI_star == max(all_CI)
print(f"Це підтверджує теорію: {is_best}")

# Візуалізація
print("\n" + "=" * 60)
print("ВІЗУАЛІЗАЦІЯ")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, (name, U, CI, PC, g) in enumerate(results[:6]):
    if i < len(axes):
        # Жорстке розбиття для візуалізації
        hard_labels = np.argmax(U, axis=0)

        axes[i].scatter(
            X[:, 0], X[:, 1], c=hard_labels, cmap="viridis", alpha=0.7
        )
        axes[i].set_title(f"{name}\nCI = {CI:.3f}")
        axes[i].set_xlabel("X1")
        axes[i].set_ylabel("X2")
        axes[i].grid(True, alpha=0.3)

for i in range(len(results), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Графік порівняння CI
plt.figure(figsize=(10, 6))
names = [r[0] for r in results]
ci_values = [r[2] for r in results]

bars = plt.bar(range(len(names)), ci_values, alpha=0.7)
# Виділяємо еталонну червоним
bars[0].set_color("red")
bars[0].set_alpha(0.9)

plt.title("Порівняння індексу чіткості CI")
plt.ylabel("CI")
plt.xticks(range(len(names)), names, rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nВисновок: найкращий розбиття U* має CI = {CI_star:.4f}")
