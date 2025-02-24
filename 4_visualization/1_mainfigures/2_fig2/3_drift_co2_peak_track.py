import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from tqdm import tqdm
from util import *  # Import style-related variables

# Define the path to your CSV files
csv_directory = "/Users/sean/Library/CloudStorage/OneDrive-postech.ac.kr/연구/1_projects_mace/resources/240527_source_data/DRIFTS_CeO2_CO-O2_step_reaction"
csv_files = ['drift_CM.csv', 'drift_CR.csv', 'drift_CC.csv']
labels = ['CM', 'CR', 'CC']

def get_absorbance_near_peak(data_set, wavenumber_target, tolerance=5):
    """Get absorbance values near a specific wavenumber within a tolerance."""
    filtered_data = data_set[(data_set['cm-1'] >= wavenumber_target - tolerance) & 
                             (data_set['cm-1'] <= wavenumber_target + tolerance)]
    absorbance = filtered_data.max(axis=0, numeric_only=True)[1:].values  # Max absorbance within tolerance
    return absorbance

def plot_absorbance_time_series():
    # Target wavenumber and tolerance
    target_wavenumber = 2360
    tolerance = 5

    plt.figure(figsize=(7, 3))

    # Loop through files and extract absorbance near 2360 cm^-1
    for i, (filename, label) in enumerate(zip(csv_files, labels)):
        path = os.path.join(csv_directory, filename)
        data_set = read_csv(path, header=1).dropna(axis=0)
        time = np.array([float(i) for i in data_set.columns[1:]])  # Time points
        absorbance_near_2360 = get_absorbance_near_peak(data_set, target_wavenumber, tolerance)

        plt.plot(time, absorbance_near_2360, color=colors[i], label=label)
        plt.scatter(time, absorbance_near_2360, s=50, color=colors[i])
    plt.xlim(0, 60)
    plt.ylim(0, 0.01)
    for x_val in [15, 45]:
        plt.axvline(x=x_val, color='gray', linestyle='--', linewidth=1,alpha=0.5)
    # Customize plot
    plt.xlabel("Time (min)", font_properties=font_properties_label)
    plt.ylabel("Absorbance at 2360 cm$^{-1}$ (a.u.)", font_properties=font_properties_label)
    plt.xticks(font_properties=font_properties_tick)
    plt.yticks(font_properties=font_properties_tick)
    plt.tight_layout()

    # Save and show plot
    plt.savefig('./2_drifts/absorbance_2360_time_series.png', dpi=200)
    plt.show()

# Call the function
plot_absorbance_time_series()
