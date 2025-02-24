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
def plot_xps(df, ax,color):
    raw = df.columns[1]
    treated = df.columns[2]
    baseline_column = df.columns[3]
    baseline = df[baseline_column]
    au_delta = df.columns[[4, 5]]
    au_0 = df.columns[[6, 7]]
    au_oxid = df.columns[[8, 9]]
    x = df["Intensity (a.u)"]
    
    # Find peaks for raw data
    
    # Plot raw data with peaks
    ax.scatter(x, df[raw],
               marker='o',
               s=50,
               facecolor='white',
               edgecolor='gray',
               alpha=0.3,
               linewidth=1.5)
    
    for i, label in enumerate(au_0):
        y = df[label]
        #ax.plot(x, y, label=label, color=color, linewidth=1.5, zorder=2-0.1*i)
        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)

        '''        for px, py in zip(peak_x, peak_y):
                    if px<90:
                        ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
                        ax.scatter(px, py, marker='v', s=10, color='gray',alpha=0.7)'''
        if i == 0:
            ax.fill_between(x, baseline, y, where=(y > baseline), color='gray', alpha=0.7, 
                          label=r'$\mathregular{Au^{0}}$')
        else:
            ax.fill_between(x, baseline, y, where=(y > baseline), color='gray', alpha=0.7)
    for i, label in enumerate(au_delta):
        y = df[label]
        #ax.plot(x, y, label=label, color='gray', linewidth=1.5, zorder=2-0.1*i)
        '''        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)

                for px, py in zip(peak_x, peak_y):
                    if px<90:
                        ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
                        ax.scatter(px, py, marker='v', s=10, color=color,alpha=0.3)'''
        if i == 0:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.3, 
                          label=r'$\mathregular{Au^{+}}$')    
        else:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.3)
    for i, label in enumerate(au_oxid):
        y = df[label]
        #ax.plot(x, y, label=label, color=color, linewidth=1.5, zorder=2-0.1*i)
        '''        peak_x, peak_y = find_local_maxima(x, df[label], min_distance=2)

                for px, py in zip(peak_x, peak_y):
                    if px<90:
                        ax.text(px, py, f'{px:.1f}', fontsize=10, ha='center', va='bottom')
                        ax.scatter(px, py, marker='v', s=10, color=color)'''
        if i == 0:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.7, 
                          label=r'$\mathregular{Au^{3+}}$')
        else:
            ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.7)


    #ax.legend(loc='upper left',prop=font_properties_label)
    ax.set_xlabel("Binding energy (eV)", fontproperties=font_properties_label)
    ax.set_ylabel("Intensity (a.u.)", fontproperties=font_properties_label)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(), fontproperties=font_properties_tick)
    ax.set_yticks([])
    ax.set_xlim(91, 81)
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
        #ax.plot(x, y, label=label, color=color, linewidth=1.5, zorder=2-0.1*i)
        ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.7, 
                       label=r'$\mathregular{O_{ads}}$')
    for i, label in enumerate(Oads):
        y = df[label]
        #ax.plot(x, y, label=label, color=color, linewidth=1.5, zorder=2-0.1*i)
        ax.fill_between(x, baseline, y, where=(y > baseline), color=color, alpha=0.3, 
                       label=r'$\mathregular{O_{lat}}$')
    for i, label in enumerate(Owet):
        y = df[label]
        #ax.plot(x, y, label=label, color='gray', linewidth=1.5, zorder=2-0.1*i)
        ax.fill_between(x, baseline, y, where=(y > baseline), color='gray', alpha=0.3, 
                       label=r'H$\mathregular{_{2}}$O')

    ax.set_xlabel("Binding energy (eV)", fontproperties=font_properties_label)
    ax.set_ylabel("Intensity (a.u.)", fontproperties=font_properties_label)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(), fontproperties=font_properties_tick)
    ax.set_yticks([])
    ax.set_xlim(537, 525)
    #ax.legend(loc='upper left',prop=font_properties_label)
    #plt.ylim(15000, 160000)
    plt.tight_layout()


file_path = '../../resources/240527_source_data/240527_source_data.xlsx'
df_a = pd.read_excel(file_path, sheet_name='Fig. S20d', skiprows=1)
df_b = pd.read_excel(file_path, sheet_name='Fig. S20e', skiprows=1)
df_c = pd.read_excel(file_path, sheet_name='Fig. S20f', skiprows=1)

df_d = pd.read_excel(file_path, sheet_name='Fig. S20a', skiprows=1)
df_e = pd.read_excel(file_path, sheet_name='Fig. S20b', skiprows=1)
df_f = pd.read_excel(file_path, sheet_name='Fig. S20c', skiprows=1)

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
plt.savefig('./output/xps_au_mace.png',dpi=300)
plt.show()
