import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../utility')
from util import *
# Load data from xlsx file
file_path = '../../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig.2d', skiprows=1)

def plot_stability_conversion_with_break(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 2.2), gridspec_kw={'height_ratios': [3, 2]})
    
    labels = df.columns[1:]
    
    for i, label in enumerate(labels):
        x = df["Temperature (°C)"].iloc[::2]
        y = df[label].iloc[::2]
        
        ax1.plot(x, y, color=colors[i], linewidth=1.5, zorder=2-0.1*i)
        ax1.scatter(x, y, color=colors[i], s=50, marker='o', linewidth=1.5, zorder=2-0.1*i)
        ax2.plot(x, y, color=colors[i], linewidth=1.5, zorder=2-0.1*i)
        ax2.scatter(x, y, color=colors[i], s=50, marker='o', linewidth=1.5, zorder=2-0.1*i)

    # Set limits
    ax1.set_ylim(75, 100)  # Upper subplot: 70-100%
    ax2.set_ylim(0, 15)    # Lower subplot: 0-20%
    ax1.set_xlim(0, 80)
    ax2.set_xlim(0, 80)
    
    # Remove bottom frame of ax1 and top frame of ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Remove x ticks from ax1
    ax1.set_xticks([])
    
    # Add wavy break lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
    wave_height = 0.015
    
    # Create wavy lines at the bottom of ax1
    y = np.linspace(-wave_height, wave_height, 20)
    x = np.zeros_like(y) + 0.005 * np.sin(50 * y)
    ax1.plot(x, y, **kwargs)
    ax1.plot(x + 1, y, **kwargs)
    
    # Create wavy lines at the top of ax2
    kwargs.update(transform=ax2.transAxes)
    y = np.linspace(1-wave_height, 1+wave_height, 20)
    x = np.zeros_like(y) + 0.005 * np.sin(50 * y)
    ax2.plot(x, y, **kwargs)
    ax2.plot(x + 1, y, **kwargs)
    
    # Remove the individual y-labels
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    
    # Add a single y-label for both subplots
    fig.supylabel("CO Conversion (%)", fontproperties=font_properties_label, x=0.04,y=0.6)
    ax2.set_xlabel("Time on stream (hours)", fontproperties=font_properties_label)

    # Fix: Set proper tick parameters
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_properties_tick)
    
    plt.tight_layout()
    plt.savefig('./output/b.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    plot_stability_conversion_with_break(df)