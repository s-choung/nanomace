import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = '../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig.2d', skiprows=1)

def plot_stability_conversion(df):
    plt.figure(figsize=(5, 4))
    labels = df.columns[1:] 
    print(labels)
    for i, label in enumerate(labels):
        x = df["Temperature (°C)"].iloc[::2] 
        y = df[label].iloc[::2] 
        plt.plot(x, y, label=label, color=colors[i], linewidth=1.5, zorder=2-0.1*i)
        plt.scatter(x, y, color=colors[i],s=50, marker='o', linewidth=1.5, zorder=2-0.1*i)#colors[i]
    plt.xlabel("Time on stream (hours)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.tight_layout()
    plt.ylim(0,100)
    plt.xlim(0,80)
    plt.savefig(f'./1_pngs/fig2_c.png', dpi=200)
    plt.show()

plot_stability_conversion(df)