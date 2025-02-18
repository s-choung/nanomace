import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
from scipy.signal import find_peaks

# Load data from xlsx file (provide correct path)
file_path = '../../resources/au_mace_xps_taein.xlsx'

def find_local_maxima(x, y, min_distance=2):
    # Convert x and y to numpy arrays if they aren't already
    x = np.array(x)
    y = np.array(y)
    
    # Find peaks
    peaks, _ = find_peaks(y, distance=min_distance)
    
    # Return x and y coordinates of peaks
    return x[peaks], y[peaks]
def plot_xps(df, ax,color):
    raw = df.columns[1]
    treated = df.columns[2]
    baseline_column = df.columns[3]
    baseline = df[baseline_column]
    Ce_four = df.columns[[4, 5, 7, 8, 10, 12]]
    Ce_three = df.columns[[6,9,11,13]]#, 11, 13]]
    x = df["Intensity (a.u)"]
    
    # Use scatter plot for raw data
    ax.scatter(x, df[raw],
               marker='o',
               s=50,
               facecolor='white',
               edgecolor='gray',
               alpha=0.3,
               linewidth=1.5)
    
    # Comment out treated and baseline plots if not needed
    # ax.plot(x, df[treated], label="treated", color='black', linewidth=1.5)
    # ax.plot(x, baseline, label="baseline", color='black', linewidth=1.5)
    
    for i, label in enumerate(Ce_four):
        y = df[label]
        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)
        ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.3)

    for i, label in enumerate(Ce_three):
        y = df[label]
        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)
        ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.7)

    ax.set_xlabel("Binding energy (eV)", fontproperties=font_properties_label)
    ax.set_ylabel("Intensity (a.u.)", fontproperties=font_properties_label)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(), fontproperties=font_properties_tick)
    ax.set_yticks([])
    ax.set_xlim(922, 875)
    #ax.legend(loc='lower left',prop=font_properties_label)
    plt.tight_layout()
    
    # Calculate areas
    ce4_area = 0
    ce3_area = 0
    
    # Calculate Ce4+ area
    for label in Ce_four:
        y = df[label]
        area = np.trapz(y[y > baseline] - baseline[y > baseline], x[y > baseline])
        ce4_area += area
    
    # Calculate Ce3+ area
    for label in Ce_three:
        y = df[label]
        area = np.trapz(y[y > baseline] - baseline[y > baseline], x[y > baseline])
        ce3_area += area
    
    # Print results
    total_area = ce3_area + ce4_area
    ce3_percentage = (ce3_area / total_area) * 100
    ce4_percentage = (ce4_area / total_area) * 100
    
    print(f"\nResults for dataset:")
    print(f"Ce3+ area: {ce3_area:.2f} ({ce3_percentage:.1f}%)")
    print(f"Ce4+ area: {ce4_area:.2f} ({ce4_percentage:.1f}%)")
    print(f"Total area: {total_area:.2f}")


df_a = pd.read_excel(file_path, sheet_name='task1', skiprows=1)
df_b = pd.read_excel(file_path, sheet_name='task2', skiprows=1)
df_c = pd.read_excel(file_path, sheet_name='task3', skiprows=1)

# Add debugging prints
print("\nColumn headers in df_a:")
print(df_a.columns.tolist())

# Print specifically Ce3+ and Ce4+ related columns
print("\nCe4+ columns:")
print(df_a.columns[[4, 5, 7, 8, 10, 12]].tolist())
print("\nCe3+ columns:")
print(df_a.columns[[6, 9, 11, 13]].tolist())

# Add debugging prints before plotting
print("\nDataset C Column headers:")
print(df_c.columns.tolist())
print("\nCe4+ columns in df_c:")
print(df_c.columns[[4, 5, 7, 8, 10, 12]].tolist())
print("\nCe3+ columns in df_c:")
print(df_c.columns[[6, 9, 11, 13]].tolist())

# Create 2x3 subplots for plot_xps and plot_xps_def
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot for plot_xps
plot_xps(df_a, axes[0], colors[0])
plot_xps(df_b, axes[1], colors[1])
plot_xps(df_c, axes[2], colors[2])


plt.tight_layout()
plt.savefig('./output/au_mace_ce_xps.png',dpi=300)
plt.show()
