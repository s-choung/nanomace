import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = './240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig. S14c,f,i', skiprows=1)
print(df)
def plot_distribution(df):
    plt.figure(figsize=(5, 4))
    labels =df.columns[[2]]
    for i, label in enumerate(labels):
        x = df["particle size"]
        y = df[label]
        plt.bar(x, y, label=label, color=colors[i+1], width=0.4, zorder=2-0.1*i)
    plt.ylabel("Frequency Counts", fontproperties=font_properties_label)
    plt.xlabel("Particle Size (nm)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.ylim(0,70)
    plt.tight_layout()
    
    plt.savefig(f'./1_pngs/Fig. S14f.png', dpi=200)
    plt.show()

plot_distribution(df)
