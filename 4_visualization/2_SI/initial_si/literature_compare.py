import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
from adjustText import adjust_text

file_path='../resources/nanomace_compare_t20.xlsx'
df = pd.read_excel(file_path, sheet_name=0)#, skiprows=1)
# Convert non-numeric entries to NaN for 'T50' and 'T20'
df['T50'] = pd.to_numeric(df['T50'], errors='coerce')
df['T20'] = pd.to_numeric(df['T20'], errors='coerce')

# Drop rows where either 'T50' or 'T20' is NaN
df = df.dropna(subset=['T50', 'T20'])

# Define colors and markers
highlight_cases = ['Au/CeO2-M', 'Au/CeO2-R', 'Au/CeO2-C']

def plot_T50_vs_T20_with_labels(df, highlight_cases, colors):
    plt.figure(figsize=(4, 4))
    texts = []

    # Plot all cases in gray first, then highlight specific cases
    for index, row in df.iterrows():
        color = 'gray'  # Default color
        marker = 'o'    # Default marker
        ref=row['ref']
        s=100
        catalyst=subscript(row['Oxide supported metal catalyst'])+f'_ref[{ref}]'
        for i, case in enumerate(highlight_cases):
            if case in row['Oxide supported metal catalyst']:
                color = colors[i]
                marker = '*'  # Highlighted marker
                s=300
                catalyst=subscript(row['Oxide supported metal catalyst'])
                break  # Stop checking once a match is found

        plt.scatter(row['T20'], row['T50'], color=color, marker=marker, s=s, alpha=0.7)
        texts.append(plt.text(row['T20'], row['T50'], catalyst,size=6,color=color))#,fontproperties=font_properties_tick))

    plt.xlabel("T20 (°C)", fontproperties=font_properties_label)
    plt.ylabel("T50 (°C)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.ylim(100,500)
    plt.ylim(100,500)

    # Use adjustText to automatically adjust text labels
    adjust_text(texts,  force_text=1.0, force_points=1, arrowprops=dict(arrowstyle='-', color='gray',alpha=0.3))
    plt.savefig(f'./1_pngs/literature.png', dpi=200)

    plt.show()

plot_T50_vs_T20_with_labels(df, highlight_cases, colors)
