import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../utility')
from util import *
from adjustText import adjust_text
file_path='../../../resources/nanomace_compare_t20_250207.xlsx'
df = pd.read_excel(file_path, sheet_name=0)#, skiprows=1)
# Convert non-numeric entries to NaN for 'T50' and 'T20'
df['T50'] = pd.to_numeric(df['T50'], errors='coerce')
df['T20'] = pd.to_numeric(df['T20'], errors='coerce')

# Drop rows where either 'T50' or 'T20' is NaN
df = df.dropna(subset=['T50', 'T20'])

# Define colors and markers
highlight_cases = ['Au/CeO2-M', 'Au/CeO2-R', 'Au/CeO2-C']

def plot_T50_vs_T20_with_labels(df, highlight_cases, colors):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)    
    texts = []

    # Plot all cases in gray first, then highlight specific cases
    for index, row in df.iterrows():
        color = 'gray'  # Default color
        marker = 'o'    # Default marker
        ref = row['ref']
        s = 50
        catalyst = f'ref.{ref}'
        is_highlighted = False  # Flag to track if case is highlighted

        for i, case in enumerate(highlight_cases):
            if case in row['Oxide supported metal catalyst']:
                color = colors[i]
                marker = '*'  # Highlighted marker
                s = 300
                is_highlighted = True
                break

        plt.scatter(row['T20'], row['T50'], color=color, marker=marker, s=s, alpha=0.7)
        
        # Only add text if it's not a highlighted case
        if not is_highlighted:
            texts.append(plt.text(row['T20'], row['T50'], catalyst, size=10, color=color, alpha=0.5, fontproperties=font_properties_tick))

    plt.xlabel("T20 (°C)", fontproperties=font_properties_label)
    plt.ylabel("T50 (°C)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.ylim(100,500)
    plt.ylim(100,500)

    x = df['T20']
    y = df['T50']
    # Use adjustText to automatically adjust text labels
    adjust_text(texts, x=x, y=y, 
                force_text=5.0,
                force_points=20.0,
                arrowprops=dict(
                    arrowstyle='-',
                    color='gray',
                    alpha=0.5,
                    lw=0.5,
                    shrinkA=5,
                    shrinkB=5,
                    # Remove connectionstyle to avoid transform issues
                ),
                expand_points=(15, 15),
                expand_text=(15, 15),
                add_objects=[plt.scatter([], [])],
                only_move={'points':'xy', 'texts':'xy'},
                avoid_self=True,
                avoid_points=True,
                )
    ax.set_position([0.2, 0.2, 0.666, 0.666])


    plt.savefig(f'./output/literature.png', dpi=200)

    plt.show()

plot_T50_vs_T20_with_labels(df, highlight_cases, colors)
