import pandas as pd
import matplotlib.pyplot as plt

seq_file = "results_seq.csv"
par_file = "results_omp.csv"

seq_df = pd.read_csv(seq_file)
par_df = pd.read_csv(par_file)

merged = par_df.merge(seq_df[['N', 'time_ms']], on='N', suffixes=('_par', '_seq'))

merged['speedup'] = merged['time_ms_seq'] / merged['time_ms_par']

N_values = sorted(merged['N'].astype(int).unique())

Y_max = 7
Y_ticks = list(range(Y_max + 1))

schedules = ['static', 'dynamic', 'guided', 'default']

def try_int(x):
    try:
        return int(x)
    except ValueError:
        return x

merged['chunk'] = merged['chunk'].apply(try_int)

for schedule in schedules:
    plt.figure(figsize=(10, 5))
    schedule_df = merged[merged['schedule'] == schedule]

    for chunk in sorted(schedule_df['chunk'].unique()):
        df = schedule_df[schedule_df['chunk'] == chunk]

        df = df.set_index('N').reindex(N_values).reset_index()

        plt.plot(df['N'], df['speedup'], marker='o', label=f"chunk={chunk}")

    plt.xlabel('Количество элементов N')
    plt.ylabel('Ускорение (T_seq / T_par)')
    plt.title(f'Параллельное ускорение для расписания {schedule}')
    plt.xscale('log')
    plt.xticks(N_values, [str(n) for n in N_values], rotation=45)
    plt.yticks(Y_ticks)
    plt.ylim(0, Y_max)
    plt.grid(True, ls="--", linewidth=0.5, which='both')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"graphs/{schedule}.png", dpi=300)
    plt.close()