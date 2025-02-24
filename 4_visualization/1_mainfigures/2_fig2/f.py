import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utility')
from util import *
import numpy as np
import os
from pandas import read_csv
# Load data from xlsx file


def f(csv_directory,
      csv_files,
      figsize,
      labels):
    def get_absorbance_near_peak(data_set, wavenumber_target, tolerance=5):
        filtered_data = data_set[(data_set['cm-1'] >= wavenumber_target - tolerance) & 
                                  (data_set['cm-1'] <= wavenumber_target + tolerance)]
        absorbance = filtered_data.max(axis=0, numeric_only=True)[1:].values  # Max absorbance within tolerance
        return absorbance
    target_wavenumber = 2360
    tolerance = 5

    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axis

    for i, (filename, label) in enumerate(zip(csv_files, labels)):
        path = os.path.join(csv_directory, filename)
        data_set = read_csv(path, header=1).dropna(axis=0)
        time = np.array([float(i) for i in data_set.columns[1:]])  # Time points
        absorbance_near_2360 = get_absorbance_near_peak(data_set, target_wavenumber, tolerance)

        ax.plot(time, absorbance_near_2360, color=colors[i], label=label)
        ax.scatter(time, absorbance_near_2360, s=50, color=colors[i])
    
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.01)
    for x_val in [15, 45]:
        ax.axvline(x=x_val, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize plot
    ax.set_xlabel("Time (min)", fontproperties=font_properties_label)
    ax.set_ylabel("CO$_2$ Absorbance (a.u.)", fontproperties=font_properties_label)
    ax.tick_params(axis='x', labelsize=font_properties_tick.get_size())
    ax.tick_params(axis='y', labelsize=font_properties_tick.get_size())
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # Add this line
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_properties_tick)
        
    plt.subplots_adjust(
    wspace=0.15,
    hspace=0.1,
    bottom=0.25,
    left=0.15,
    top=0.9
        )   

    plt.savefig(f'./output/f.png', dpi=200)
    plt.show()

    return fig, ax  # Return the figure and axis

if __name__ == "__main__":

    CSV_DIR = "/Users/sean/Library/CloudStorage/OneDrive-postech.ac.kr/연구/1_projects_mace/resources/240527_source_data/DRIFTS_CeO2_CO-O2_step_reaction"
    CSV_FILES = ['drift_CM.csv', 'drift_CR.csv', 'drift_CC.csv']

    fig_f, ax_f = f(csv_directory=CSV_DIR,
        csv_files=CSV_FILES,
        figsize=(6,2),
        labels=['CM', 'CR', 'CC'])
    