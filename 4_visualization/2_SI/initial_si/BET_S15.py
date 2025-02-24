import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = './240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig. S15', skiprows=1)
print(df)
def plot_BET(df):
    plt.figure(figsize=(5, 4))
    labels =df.columns[[1,2,3]]
    for i, label in enumerate(labels):
        x = i
        y = df[label]
        plt.bar(x, y, label=label, color=colors[i], width=0.5, zorder=2-0.1*i)
    plt.ylabel("$S_A (m^2/g)$", fontproperties=font_properties_label)
    plt.xticks([])
    plt.yticks(fontproperties=font_properties_tick)
    plt.xlim(-0.5,2.5)
    plt.ylim(0,60)
    plt.tight_layout()
    
    plt.savefig(f'./1_pngs/Fig. S15.png', dpi=200)
    plt.show()

plot_BET(df)
