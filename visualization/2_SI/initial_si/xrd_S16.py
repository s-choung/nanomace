import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = './240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig. S16', skiprows=1)
print(df)
def plot_xrd(df):
    plt.figure(figsize=(5, 4))
    labels =df.columns[[1,3,5]]
    references =df.columns[[7]]
    for i, label in enumerate(labels):
        x = df["Degree (2-theta)"]
        y = df[label]
        plt.plot(x, y, label=label, color='black', linewidth=1.5, zorder=2-0.1*i)
    for i, reference in enumerate(references):
        for j, degree in enumerate(df.columns[[6]]):
            x = df[degree]
            y = df[reference]
            plt.bar(x, y, label=reference, color='grey', width=0.2, zorder=2-0.1*i)

    plt.xlabel("Degree (2-theta)", fontproperties=font_properties_label)
    plt.ylabel("Intensity (a.u)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks([])
    plt.xlim(20, 60)
    plt.tight_layout()
    
    plt.savefig(f'./1_pngs/Fig. S16.png', dpi=200)
    plt.show()

plot_xrd(df)
