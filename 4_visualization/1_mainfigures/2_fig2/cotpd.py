import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = '../../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='co tpd', skiprows=1)

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

def plot_COTPD(df):
    plt.figure(figsize=(5, 3.88))
    temperature = df.columns[0]
    labels = df.columns[[1,3,5]]
    min_list = []
    max_list = []
    temp_10_list = []
    areas = []  # List to store calculated areas
    surf_area=[60.04,52.04,38.76]
    # Plot each curve and find its baseline
    span = (np.max(df[labels[0]])-np.min(df[labels[0]]))*1.4
    for i, label in enumerate(labels):
        x = df[temperature]
        y = df[label]-span*i
        # Plot main curve
        plt.plot(x, y, label=label, color=colors[i], linewidth=2.5)
        baseline = np.min(y)
        max_val = np.max(y)
        diff = max_val-baseline
        threshold = baseline + 0.05 * diff

        # Calculate area only up to 500°C
        mask = (x[y > baseline] < 800)
        x_limited = x[y > baseline][mask]
        y_limited = y[y > baseline][mask]
        area = np.trapz(y_limited - baseline, x_limited)
        areas.append(area)
        area_normalized = area/surf_area[i]
        # Add area annotation
        print(f'Surface area: {surf_area[i]:.0f}')
        print(f'Area: {area}')
        print(f'Area: {area_normalized}')

        # Rest of the code for temperature annotation
        for j in range(len(x)-1):
            if y[j] <= threshold <= y[j+1]:
                temp_10 = x[j] + (x[j+1]-x[j])*(threshold-y[j])/(y[j+1]-y[j])
                temp_10_list.append(temp_10)
                plt.plot([temp_10, temp_10], [baseline, threshold+diff*0.2], 
                        color=colors[i], linestyle='-', alpha=0.5, linewidth=1)
                plt.text(temp_10, threshold+diff*0.2, f'{temp_10:.0f}°C', 
                        ha='right', va='bottom', color=colors[i], 
                        fontproperties=font_properties_annotate)
                break
                
        max_list.append(np.max(y))
        plt.fill_between(x, baseline, y, where=(y > baseline), color=colors[i], alpha=0.3)
        min_list.append(baseline)

    # Add text annotations after collecting all data points

    # Rest of the plotting code remains the same
    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("Intensity (a.u)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks([])
    plt.xlim(50,500)
    plt.ylim(np.min(min_list), np.max(max_list)*1.15)
    plt.tight_layout()
    plt.savefig(f'./output/coTPD_S13.png', dpi=200)
    plt.show()


plot_COTPD(df)