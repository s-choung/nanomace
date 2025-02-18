import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
from scipy.signal import find_peaks
# Load data from xlsx file (provide correct path)
file_path = '../../resources/240527_source_data/240527_source_data.xlsx'

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
    Ce_three = df.columns[[6, 9, 11, 13]]
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
        if i == 0:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.3, 
                          label=r'$\mathregular{Ce^{4+}}$')    
        else:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.3)
        
        # Add peak labels
        for px, py in zip(peak_x, peak_y):
            ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
            ax.scatter(px, py, marker='v', s=10, color=color, alpha=0.3)
    
    for i, label in enumerate(Ce_three):
        y = df[label]
        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)
        if i == 0:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.7, 
                          label=r'$\mathregular{Ce^{3+}}$')
        else:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.7)
        
        # Add peak labels
        for px, py in zip(peak_x, peak_y):
            ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
            ax.scatter(px, py, marker='v', s=10, color=color, alpha=0.7)
    
    ax.set_xlabel("Binding energy (eV)", fontproperties=font_properties_label)
    ax.set_ylabel("Intensity (a.u.)", fontproperties=font_properties_label)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(), fontproperties=font_properties_tick)
    ax.set_yticks([])
    ax.set_xlim(922, 875)
    #ax.legend(loc='lower left',prop=font_properties_label)
    plt.tight_layout()

def plot_xps_def(df, ax,color):
    raw = df.columns[1]
    treated = df.columns[2]
    baseline_column = df.columns[3]
    baseline = df[baseline_column]
    Olatt = df.columns[[4]]
    Oads = df.columns[[5]]
    Owet = df.columns[[6]]
    x = df["Intensity (a.u)"]
    ax.scatter(x, df[raw],
                marker='o',
                s=50,
                facecolor='white',
                edgecolor='gray',
                alpha=0.3,
                linewidth=1.5)
    #ax.plot(x, df[treated], label="treated", color='black', linewidth=1.5)
    #ax.plot(x, baseline, label="baseline", color='black', linewidth=1.5)   
    for i, label in enumerate(Olatt):
        y = df[label]
        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)
        ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.7, 
                       label=r'$\mathregular{O_{ads}}$')
        # Add peak labels
        for px, py in zip(peak_x, peak_y):
            ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
            ax.scatter(px, py, marker='v', s=10, color=color, alpha=0.7)
    for i, label in enumerate(Oads):
        y = df[label]
        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)
        ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.3, 
                       label=r'$\mathregular{O_{lat}}$')
        # Add peak labels
        for px, py in zip(peak_x, peak_y):
            ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
            ax.scatter(px, py, marker='v', s=10, color=color, alpha=0.3)
    for i, label in enumerate(Owet):
        y = df[label]
        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)
        ax.fill_between(x, baseline, y, where=(y > baseline), color='gray', alpha=0.3, 
                       label=r'H$\mathregular{_{2}}$O')
        # Add peak labels
        for px, py in zip(peak_x, peak_y):
            ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
            ax.scatter(px, py, marker='v', s=10, color='gray', alpha=0.3)

    ax.set_xlabel("Binding energy (eV)", fontproperties=font_properties_label)
    ax.set_ylabel("Intensity (a.u.)", fontproperties=font_properties_label)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(), fontproperties=font_properties_tick)
    ax.set_yticks([])
    ax.set_xlim(537, 525)
    #ax.legend(loc='upper left',prop=font_properties_label)
    #plt.ylim(15000, 160000)
    plt.tight_layout()

df_a = pd.read_excel(file_path, sheet_name='Fig. S4a', skiprows=1)
df_b = pd.read_excel(file_path, sheet_name='Fig. S4b', skiprows=1)
df_c = pd.read_excel(file_path, sheet_name='Fig. S4c', skiprows=1)

df_d = pd.read_excel(file_path, sheet_name='Fig. S4d', skiprows=1)
df_e = pd.read_excel(file_path, sheet_name='Fig. S4e', skiprows=1)
df_f = pd.read_excel(file_path, sheet_name='Fig. S4f', skiprows=1)

# Create 2x3 subplots for plot_xps and plot_xps_def
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Plot for plot_xps
plot_xps(df_a, axes[0, 0], colors[0])
plot_xps(df_b, axes[0, 1], colors[1])
plot_xps(df_c, axes[0, 2], colors[2])

# Plot for plot_xps_def
plot_xps_def(df_d, axes[1, 0], colors[0])
plot_xps_def(df_e, axes[1, 1], colors[1])
plot_xps_def(df_f, axes[1, 2], colors[2])

plt.tight_layout()
plt.savefig('./output/test.png',dpi=300)
plt.show()
