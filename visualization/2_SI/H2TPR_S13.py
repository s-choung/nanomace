import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = '../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig. S13', skiprows=1)
print(df)

def find_peaks(x, y, min_distance=100):
    """Find peaks in H2TPR data."""
    peaks_x = []
    peaks_y = []
    i = 0
    while i < len(x):
        # Define window
        window_mask = (x >= x[i] - min_distance/2) & (x <= x[i] + min_distance/2)
        window_y = y[window_mask]
        
        if len(window_y) == 0:
            i += 1
            continue
            
        max_value = np.max(window_y)
        if max_value == y[i]:
            peaks_x.append(x[i])
            peaks_y.append(y[i])
            i += len(window_y)
        else:
            i += 1
            
    return peaks_x, peaks_y

def plot_H2TPR(df):
    plt.figure(figsize=(5, 4))
    temperature = df.columns[0]
    labels = df.columns[[1,3,5]]
    min_list=[]
    max_list=[]
    # Plot each curve and find its baseline
    for i, label in enumerate(labels):
        x = df[temperature]
        y = df[label]-1000*i
        
        # Plot main curve
        plt.plot(x, y, label=label, color=colors[i], linewidth=2.5)
        
        # Find and plot peaks
        peaks_x, peaks_y = find_peaks(x, y)
        for peak_x, peak_y in zip(peaks_x, peaks_y):
            if peak_x > 100:
                plt.text(peak_x, peak_y+200, f'{peak_x:.0f}', fontproperties=font_properties_annotate, color=colors[i], ha='center', va='bottom')
                plt.plot([peak_x, peak_x], [peak_y+200, peak_y], color=colors[i], linestyle='-', alpha=1,linewidth=1)

        # Add baseline
        baseline = np.min(y)
        print(baseline)
        max_list.append(np.max(y))
        plt.fill_between(x, baseline, y, where=(y > baseline), color=colors[i], alpha=0.3)
        min_list.append(baseline)
    print(min_list)
    print(min_list-np.min(min_list))
    print(max_list)
    print(max_list-np.max(max_list))
    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("Intensity (a.u)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks([])
    plt.xlim(50,900)
    plt.ylim(np.min(min_list),np.max(max_list)+1000)
    plt.tight_layout()
    plt.savefig(f'./output/H2TPR_S13.png', dpi=200)
    plt.show()


plot_H2TPR(df)
