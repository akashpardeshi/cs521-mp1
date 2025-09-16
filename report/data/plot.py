import matplotlib.pyplot as plt
import numpy as np

def grouped_bar_chart(labels, y1, y2, platform):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, y1, width, label='Size=100')
    ax.bar(x + width/2, y2, width, label='Size=1000')

    ax.set_title(f'Ablation: Speedup of Optimizations ({platform})')
    ax.set_xlabel('Optimization')
    ax.set_ylabel('Speedup')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # beautify
    ax.set_yscale('log')
    ax.grid(which='major', axis='y', linewidth=0.3)
    ax.grid(which='minor', axis='y', linewidth=0.1)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=False, left=True)

    plt.tight_layout()
    plt.show()

def ablation_cpu():
    labels = ['Naive', 'O1', 'O2', 'O3', 'O4']

    times_100 = [4.56893, 2.30759, 2.13045, 0.54101, 0.0952605]
    times_1000 = [3441.93, 1903.96, 1936.16, 339.696, 21.9339]

    speedup_100 = [times_100[0] / t for t in times_100]
    speedup_1000 = [times_1000[0] / t for t in times_1000]

    grouped_bar_chart(labels, speedup_100, speedup_1000, platform='CPU')

def scaling_plot(runtimes, platform):
    x = [100, 1000, 5000, 10000]
    cublas = [0.363248, 1.09712, 41.8807, 473.379]

    fig, ax = plt.subplots()
    ax.set_title(f'Scalability of {platform} Implementation')
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Runtime (ms)')
    ax.plot(x, runtimes, 'o-', label='O4' if platform == 'CPU' else 'O3')
    if platform == 'GPU':
        ax.plot(x, cublas, 'o-', label='cuBLAS sgemm')

    # beautify
    ax.set_yscale('log')
    ax.grid(which='major', axis='y', linewidth=0.3)
    ax.grid(which='minor', axis='y', linewidth=0.1)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=False, left=True)

    plt.legend()
    plt.show()

def ablation_gpu():
    labels = ['Naive', 'O1', 'O2', 'O3']

    times_100 = [64.8592, 0.026224, 0.014208, 0.014896]
    times_1000 = [151350, 7.68019, 4.65174, 2.90034]

    speedup_100 = [times_100[0] / t for t in times_100]
    speedup_1000 = [times_1000[0] / t for t in times_1000]

    grouped_bar_chart(labels, speedup_100, speedup_1000, platform='GPU')

# ablation_cpu()
# scaling_plot([0.0952605, 21.9339, 4814.07, 66533.6], platform='CPU')
# ablation_gpu()
# scaling_plot([0.014896, 2.90034, 438.898, 3837.84], platform='GPU')
