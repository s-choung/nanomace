import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from tqdm import tqdm
from util import *  # Import style-related variables

def plot_drifts_contour(
    csv_directory,
    csv_files,
    output_path='./2_drifts',
    max_time=60,
    figsize=(8, 4),
    wavenumber_range=(2280, 2380),
    vertical_lines=(15, 45),
    colormap='inferno',
    absorbance_levels=(0.0, 0.008, 8)  # start, end, num_levels
):

    def get_absorbance(data_set, wavenumber, temperature):
        absorbance = data_set[lambda data_set: np.abs(data_set['cm-1'] - wavenumber) < 1e-3][str(int(temperature))]
        return float(absorbance)

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

        # Calculate absorbance
        Absorbance = np.zeros((len(time), len(wavenumber)))
        for i in tqdm(range(len(wavenumber)), desc=f"Processing {filename}"):
            for j in range(len(time)):
                Absorbance[j][i] = get_absorbance(data_set, wavenumber[i], time[j])

        # Create contour plot
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

    # Add colorbar
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='vertical', format='%.3f')
    cbar.set_label('Absorbance (a.u.)', labelpad=10, font_properties=font_properties_label)
    cbar.ax.tick_params(width=1.25)
    cbar.outline.set_linewidth(1.25)
    for label in cbar.ax.get_yticklabels():
        label.set_font_properties(font_properties_tick)
    fig.supylabel('Wavenumber ($\mathregular{cm^{-1}}$)', fontproperties=font_properties_label, x=0.0, y=0.5)
    fig.supxlabel('Time $(min)$', fontproperties=font_properties_label, x=0.5, y=0.0)

    # Final adjustments and save
    fig.subplots_adjust(left=0.1, right=0.81, top=0.95, bottom=0.1)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f'combined_absorbance_plot_{colormap}.png'), 
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return fig, axes

# Example usage
if __name__ == "__main__":
    # Configuration
    CSV_DIR = "/Users/sean/Library/CloudStorage/OneDrive-postech.ac.kr/연구/1_projects_mace/resources/240527_source_data/DRIFTS_CeO2_CO-O2_step_reaction"
    CSV_FILES = ['drift_CM.csv', 'drift_CR.csv', 'drift_CC.csv']
    
    # Generate plot
    fig, axes = plot_drifts_contour(
        csv_directory=CSV_DIR,
        csv_files=CSV_FILES,
        figsize=(8, 4)
    )
