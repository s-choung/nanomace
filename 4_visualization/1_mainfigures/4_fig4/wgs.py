import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from util import *


data = {
    "Temperature (°C)": [120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480],
    "Au/CeO2-M": [19.263, 39.028, 74.372, 91.625, 98.157, 99.162, 99.665, 98.827, 98.827, 98.660, 98.492, 98.325, 97.822],
    "Au/CeO2-R": [5.193, 10.888, 23.283, 42.546, 79.397, 94.137, 98.660, 97.822, 97.487, 96.482, 96.147, 95.980, 95.812],
    "Au/CeO2-C": [1.005, 1.843, 3.015, 4.690, 6.533, 8.543, 10.888, 13.233, 18.090, 22.446, 32.496, 45.561, 60.637]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting function
def plot_co_conversion(df):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    labels = df.columns[1:]
    name_map = {"Au/CeO2-M": 'Au/CeO$\mathregular{_2}$ Mace', "Au/CeO2-R": 'Au/CeO$\mathregular{_2}$ Rod', "Au/CeO2-C": 'Au/CeO$\mathregular{_2}$ Cube'}

    for i, label in enumerate(labels):
        x = df["Temperature (°C)"]
        y = df[label]
        line = plt.plot(x, y, color=colors[i], linewidth=1.5, zorder=2 - 0.1 * i)[0]
        scatter = plt.scatter(x, y, color=colors[i], s=100, marker='o', linewidth=1.5, zorder=2 - 0.1 * i)
        # Create a legend handle that combines line and scatter
        ax.legend([plt.Line2D([0], [0], color=colors[i], marker='o', linestyle='-', 
                             markersize=10, linewidth=1.5) for i, _ in enumerate(labels)],
                 [name_map[label] for label in labels],
                 prop=font_properties_tick, loc='center', bbox_to_anchor=(0.6, 0.5), frameon=False)

    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.text(350, 5, 'Water Gas Shift', fontproperties=font_properties_label)

    plt.ylim(0,100)
    plt.xlim(120,480)

    ax.set_position([0.2, 0.2, 0.666, 0.666])  # 4/5 = 0.8

    plt.savefig(f'./output/fig4_wgs.png', dpi=200)
    #plt.show()

# Call the function to plot the CO conversion data
plot_co_conversion(df)
