import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
seq_file = "results_seq.csv"  # Вторая таблица (последовательная версия)
par_file = "results_omp_thread.csv"  # Первая таблица (параллельная версия)

seq_df = pd.read_csv(seq_file)
par_df = pd.read_csv(par_file)

# Объединение данных по N
merged = par_df.merge(seq_df[['N', 'time_ms']], on='N', suffixes=('_par', '_seq'))

# Вычисление ускорения
merged['speedup'] = merged['time_ms_seq'] / merged['time_ms_par']

# Уникальные значения N
N_values = sorted(merged['N'].astype(int).unique())
plt.style.use('default')
# Создаем отдельный график для каждого N
for N in N_values:
    plt.figure(figsize=(10, 6))
    
    # Фильтруем данные для текущего N
    N_data = merged[merged['N'] == N]
    
    # Получаем уникальные значения потоков и сортируем их
    thread_nums = sorted(N_data['thread_num'].unique())
    
    # Вычисляем ускорение для каждого количества потоков
    speedups = []
    for threads in thread_nums:
        thread_data = N_data[N_data['thread_num'] == threads]
        # Берем среднее ускорение, если есть несколько измерений
        avg_speedup = thread_data['speedup'].mean()
        speedups.append(avg_speedup)
    
    # Построение графика
    plt.plot(thread_nums, speedups, marker='o', linewidth=2, markersize=8)
    
    # Идеальное линейное ускорение (пунктирная линия)
    plt.plot(thread_nums, thread_nums, 'k--', alpha=0.5, label='Идеальное ускорение')
    
    # Настройки графика
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Параллельное ускорение', fontsize=12)
    plt.title(f'Параллельное ускорение для N = {N:,}', fontsize=14, fontweight='bold')
    
    # Настройка осей
    plt.grid(True, ls="--", linewidth=0.5, alpha=0.7)
    plt.xticks(thread_nums)
    
    # Автоматическое определение максимального значения для оси Y
    max_speedup = max(max(speedups), max(thread_nums))
    plt.ylim(0, max_speedup * 1.1)
    
    # Легенда
    plt.legend(loc='upper left')
    
    # Текст с максимальным ускорением
    max_speedup_value = max(speedups)
    max_speedup_thread = thread_nums[speedups.index(max_speedup_value)]
    plt.text(0.02, 0.98, f'Макс. ускорение: {max_speedup_value:.2f}x\nпри {max_speedup_thread} потоках',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    # Сохранение графика
    plt.savefig(f"graphs_threads/speedup_N_{N}.png", dpi=300)
    plt.close()
    print(f"График для N={N} сохранен как speedup_N_{N}.png")

# Дополнительно: один график со всеми N (с логарифмической осью N)
plt.figure(figsize=(12, 8))

for idx, N in enumerate(N_values):
    N_data = merged[merged['N'] == N]
    thread_nums = sorted(N_data['thread_num'].unique())
    speedups = []
    
    for threads in thread_nums:
        thread_data = N_data[N_data['thread_num'] == threads]
        avg_speedup = thread_data['speedup'].mean()
        speedups.append(avg_speedup)
    
    plt.plot(thread_nums, speedups, marker='o', linewidth=2, markersize=6, label=f'N = {N:,}')

# Идеальное ускорение
max_threads = max(merged['thread_num'].unique())
plt.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Идеальное ускорение')

# Настройки графика
plt.xlabel('Количество потоков', fontsize=12)
plt.ylabel('Параллельное ускорение', fontsize=12)
plt.title('Параллельное ускорение для разных N', fontsize=14, fontweight='bold')
plt.grid(True, ls="--", linewidth=0.5, alpha=0.7)
plt.xticks(range(2, max_threads + 1, 2))

# Легенда (в 2 колонки)
plt.legend(ncol=2, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(f"graphs_threads/speedup_all_N.png", dpi=300)
plt.close()

# print("\nСводная таблица максимальных ускорений:")
# summary_data = []

# for N in N_values:
#     N_data = merged[merged['N'] == N]
#     thread_nums = sorted(N_data['thread_num'].unique())
    
#     max_speedup = 0
#     best_threads = 0
#     seq_time = seq_df[seq_df['N'] == N]['time_ms'].values[0]
    
#     for threads in thread_nums:
#         thread_data = N_data[N_data['thread_num'] == threads]
#         avg_speedup = thread_data['speedup'].mean()
        
#         if avg_speedup > max_speedup:
#             max_speedup = avg_speedup
#             best_threads = threads
#             best_par_time = thread_data['time_ms_par'].mean()
    
#     summary_data.append({
#         'N': N,
#         'T_seq (ms)': f"{seq_time:.4f}",
#         'T_par (ms)': f"{best_par_time:.4f}",
#         'Лучшие потоки': best_threads,
#         'Макс. ускорение': f"{max_speedup:.2f}x"
#     })

# summary_df = pd.DataFrame(summary_data)
# print(summary_df.to_string(index=False))