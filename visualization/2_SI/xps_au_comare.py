import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *


def find_local_maxima(wavenumbers, absorbance_values, min_distance=20):
    """Find local maxima with a minimum distance between peaks."""
    maxima_indices = []
    maxima_values = []
    
    # Convert to numpy arrays if they aren't already
    wavenumbers = np.array(wavenumbers)
    absorbance_values = np.array(absorbance_values)
    i = 0
    while i < len(wavenumbers):
        # Define the window based on wavenumber span
        window_mask = (wavenumbers >= wavenumbers[i] - min_distance/2) & \
                     (wavenumbers <= wavenumbers[i] + min_distance/2)
        
        window_values = absorbance_values[window_mask]
        if len(window_values) == 0:
            i += 1
            continue
            
        max_value = np.max(window_values)
        if max_value == absorbance_values[i] and max_value > 0.001:  # Add threshold to filter noise
            maxima_indices.append(i)
            maxima_values.append(max_value)
            # Skip the rest of the window
            i += len(window_values)
        else:
            i += 1
            
    return wavenumbers[maxima_indices], maxima_values
def plot_xps(dfs, colors):
    plt.figure(figsize=(6, 5))
    
    for df, color in zip(dfs, colors):
        raw = df.columns[1]
        x = df["Intensity (a.u)"]
        min_value = np.min(df[raw])
        plt.scatter(x, df[raw]-min_value,
                   marker='o',
                   s=50,
                   facecolor='white',
                   edgecolor=color,
                   alpha=0.3,
                   linewidth=1.5)
        
        peak_x, peak_y = find_local_maxima(x, df[raw], min_distance=2)
        for px, py in zip(peak_x, peak_y):
            if 81 < px < 91:
                plt.text(px, py-min_value, f'{px:.1f}', fontsize=10, ha='center',color=color, va='bottom')


    
    plt.xlabel("Binding energy (eV)", fontproperties=font_properties_label)
    plt.ylabel("Intensity (a.u.)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks([])
    plt.xlim(91, 81)
    plt.tight_layout()



file_path = '../../resources/240527_source_data/240527_source_data.xlsx'
df_a = pd.read_excel(file_path, sheet_name='Fig. S20d', skiprows=1)
df_b = pd.read_excel(file_path, sheet_name='Fig. S20e', skiprows=1)
df_c = pd.read_excel(file_path, sheet_name='Fig. S20f', skiprows=1)

plot_xps([df_a, df_b, df_c], colors)

plt.tight_layout()
plt.savefig('./output/xps_test.png',dpi=300)
plt.show()
