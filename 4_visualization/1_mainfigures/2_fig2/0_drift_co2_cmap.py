import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utility')
from util import *
import numpy as np
import os
from pandas import read_csv
# Load data from xlsx file


def get_absorbance(data, wavenumber, temperature):
    matched_rows = data[np.abs(data['cm-1'] - wavenumber) < 0.5]
    if matched_rows.empty:
        return 0.0
    temp_col = str(int(temperature))
    if temp_col not in matched_rows.columns:
        return 0.0
        
    return float(matched_rows[temp_col].iloc[0])


def e(
    csv_directory,
    csv_files,
    output_path='./2_drifts',
    max_time=60,
    figsize=(16, 4),
    wavenumber_range=(2280, 2380),
    vertical_lines=(15, 45),
    colormap='inferno',
    absorbance_levels=(0.0, 0.008, 8),
    interval=20
):
    def get_absorbance(data_set, wavenumber, temperature,interval):
        absorbance = data_set[lambda data_set: np.abs(data_set['cm-1'] - wavenumber) < 1e-3][str(int(temperature))]
        return float(absorbance.iloc[0])

    fig, axes = plt.subplots(len(csv_files), 1, figsize=figsize, sharex=True)
    if len(csv_files) == 1:
        axes = [axes]  # Make axes iterable if only one plot
    fig.subplots_adjust(hspace=0.05)

    for filename, ax in zip(csv_files, axes):
        path = os.path.join(csv_directory, filename)
        
        # Read and process data
        data_set = read_csv(path, header=1).dropna(axis=0)
        time_columns = [col for col in data_set.columns[1:] if float(col) <= max_time]
        time = np.array([float(i) for i in time_columns])
        wavenumber = np.array(data_set['cm-1'])
        print(np.max(wavenumber), np.min(wavenumber))
        # Reduce data size for testing
        time = time[::interval]  ############ Use every second time point
        wavenumber = wavenumber[::interval]  ############ Use every second wavenumber

        # Calculate absorbance
        Absorbance = np.zeros((len(time), len(wavenumber)))
        for i in range(len(wavenumber)):  # Removed tqdm for speed
            for j in range(len(time)):
                Absorbance[j][i] = get_absorbance(data_set, wavenumber[i], time[j],interval)

        time_grid, wavenumber_grid = np.meshgrid(time, wavenumber, indexing='ij')
        levels = np.linspace(*absorbance_levels)
        contour = ax.contourf(time_grid, wavenumber_grid, Absorbance, 
                            cmap=colormap, alpha=1.0, levels=levels, 
                            antialiased=False, extend='both')

        # Configure axes
        ax.set_ylim(*wavenumber_range)
        ax.set_xlim(0, max_time)
        ax.tick_params(axis='x', labelsize=font_properties_tick.get_size())
        ax.tick_params(axis='y', labelsize=font_properties_tick.get_size())

        # Add vertical lines
        for x_val in vertical_lines:
            ax.axvline(x=x_val, color='white', linestyle='--', linewidth=1)

        # Style settings
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.25)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_properties_tick)


    # Adjust margins and spacing
    plt.subplots_adjust(
    wspace=0.15,
    hspace=0.1,
    bottom=0.1,
    left=0.15,
    top=0.9
        )   

    # Adjust y-label position
    fig.supylabel('Wavenumber ($\mathregular{cm^{-1}}$)', fontproperties=font_properties_label, x=0.02, y=0.5)
    fig.supxlabel('Time $(min)$', fontproperties=font_properties_label, x=0.5, y=0.0)

    # Add colorbar (modified position and style)
    cbar_ax = fig.add_axes([0.75, 0.93, 0.15, 0.02])  # [left, bottom, width, height]
    # Get the colormap object
    cmap = plt.get_cmap(colormap)

    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks([])  # Correct way to remove ticks
    # Position label to the left of colorbar
    cbar.ax.set_xlabel('Absorbance (a.u.)',  fontproperties=font_properties_annotate, labelpad=5)
    cbar.ax.xaxis.set_label_position('top')
    plt.savefig(f'./2_drifts/test.png', dpi=200)
    plt.show()

    return fig, axes  # Return the figure and axes


if __name__ == "__main__":

    CSV_DIR = "/Users/sean/Library/CloudStorage/OneDrive-postech.ac.kr/연구/1_projects_mace/resources/240527_source_data/DRIFTS_CeO2_CO-O2_step_reaction"
    CSV_FILES = ['drift_CM.csv', 'drift_CR.csv', 'drift_CC.csv']

    fig_e, ax_e = e(
        csv_directory=CSV_DIR,
        csv_files=CSV_FILES,
        figsize=(6, 4),
        interval=1
    )
    