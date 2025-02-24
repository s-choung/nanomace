import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = './240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig. S5', skiprows=1)
print(df)
def plot_co_conversion(df):
    plt.figure(figsize=(5, 4))
    labels = df.columns[1:] 
    for i, label in enumerate(labels):
        x = df["Temperature (°C)"]
        y = df[label]
        plt.plot(x, y, label=label, color=colors[i+2], linewidth=1.5, zorder=2-0.1*i)
        plt.scatter(x, y, color=colors[i+2], s=100, marker='o', linewidth=1.5, zorder=2-0.1*i)
    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.tight_layout()
    
    plt.savefig(f'./1_pngs/Fig. S5.png', dpi=200)
    plt.show()

plot_co_conversion(df)
