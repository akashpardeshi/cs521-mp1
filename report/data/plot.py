import matplotlib.pyplot as plt
import numpy as np

def grouped_bar_chart(labels, y1, y2):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, y1, width, label='Size=100')
    ax.bar(x + width/2, y2, width, label='Size=1000')

    ax.set_ylabel('Values')
    ax.set_title('Ablation: Speedup of Optimizations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

def ablation_cpu():
    labels = ['Naive', 'O1', 'O2', 'O3', 'O4']

    times_100 = [4.56893, 2.30759, 2.13045, 0.54101, 0.0952605]
    times_1000 = [3441.93, 1903.96, 1936.16, 339.696, 21.9339]

    speedup_100 = [times_100[0] / t for t in times_100]
    speedup_1000 = [times_1000[0] / t for t in times_1000]

    grouped_bar_chart(labels, speedup_100, speedup_1000)

def scaling_cpu():
    x = [100, 1000, 5000, 10000]
    times = [0.0952605, 21.9339, 4814.07, 66533.6]

    fig, ax = plt.subplots()
    ax.plot(x, times, 'o-')
    plt.show()

# ablation_cpu()
scaling_cpu()
