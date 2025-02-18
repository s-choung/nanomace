import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, concat
from tqdm import tqdm
from util import *  # Import style-related variables
from PIL import Image

def get_base_directories():

    base_path = "/Users/sean/Library/CloudStorage/OneDrive-postech.ac.kr/연구/1_projects_mace/resources/240527_source_data/DRIFTS_CeO2_CO-O2_step_reaction"
    base_directories = {
        'CM': os.path.join(base_path, 'CM'),
        'CR': os.path.join(base_path, 'CR'),
        'CC': os.path.join(base_path, 'CC')
    }
    csv_files = ['1.csv', '2.csv', '3.csv']
    
    return base_directories, csv_files

def get_absorbance(data, wavenumber, temperature):
    """
    Get absorbance value for a specific wavenumber and temperature.
    
    Args:
        data (DataFrame): Input data
        wavenumber (float): Target wavenumber
        temperature (float): Temperature value
        
    Returns:
        float: Absorbance value
    """
    # Increase tolerance for wavenumber matching
    matched_rows = data[np.abs(data['cm-1'] - wavenumber) < 0.5]
    if matched_rows.empty:
        return 0.0
    
    temp_col = str(int(temperature))
    if temp_col not in matched_rows.columns:
        return 0.0
        
    return float(matched_rows[temp_col].iloc[0])

def create_combined_plot(base_directories, output_path, output_filename, csv_files, time_points, interval):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(8, 10), 
                                    gridspec_kw={'width_ratios': [380, 700]})
    axes_pairs = [(ax1[0], ax1[1]), (ax2[0], ax2[1]), (ax3[0], ax3[1])]

    cmap = plt.cm.inferno
    time_points = time_points
    norm = plt.Normalize(min(time_points), max(time_points))
    time_colors = cmap(np.linspace(0, 0.8, len(time_points)))
    max_absorbance = 0
    
    # First pass: collect data and find maximum absorbance
    for dataset_name, base_directory in base_directories.items():
        all_data = []
        for filename in csv_files:
            path = os.path.join(base_directory, filename)
            data = read_csv(path, header=1).dropna(axis=0)
            all_data.append(data)
            
        for time in time_points:
            for data in all_data:
                wavenumber = np.array(data['cm-1'])
                for mask in [(wavenumber >= 2000) & (wavenumber <= 2380),
                           (wavenumber >= 1100) & (wavenumber <= 1800)]:
                    selected_wavenumbers = wavenumber[mask]
                    absorbance_values = [get_absorbance(data, wave, time) 
                                       for wave in selected_wavenumbers]
                    if absorbance_values:
                        max_absorbance = max(max_absorbance, max(absorbance_values))
    
    # Add 10% padding to the maximum value
    y_max = max_absorbance * 1.1
    
    # Update colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.2, 0.85, 0.2, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Time (min)', fontproperties=font_properties_label)
    cbar.ax.tick_params(labelsize=font_properties_tick.get_size())
    
    # Second pass: actual plotting
    for (dataset_name, base_directory), (ax1, ax2) in zip(base_directories.items(), axes_pairs):
        all_data = []

        for filename in csv_files:
            path = os.path.join(base_directory, filename)
            data = read_csv(path, header=1).dropna(axis=0)
            all_data.append(data)
        
        for time, color in zip(time_points, time_colors):
            for data in all_data:
                wavenumber = np.array(data['cm-1'])
                wavenumber = wavenumber[::interval]
                # Plot first region (2380-2000)
                mask1 = (wavenumber >= 2000) & (wavenumber <= 2380)
                selected_wavenumbers = wavenumber[mask1]
                absorbance_values = [get_absorbance(data, wave, time) 
                                   for wave in selected_wavenumbers]
                ax1.plot(selected_wavenumbers, absorbance_values, 
                        linewidth=1.5, color=color, alpha=0.8)
                
                # Plot second region (1800-1100)
                mask2 = (wavenumber >= 1100) & (wavenumber <= 1800)
                selected_wavenumbers = wavenumber[mask2]
                absorbance_values = [get_absorbance(data, wave, time) 
                                   for wave in selected_wavenumbers]
                ax2.plot(selected_wavenumbers, absorbance_values, 
                        linewidth=1.5, color=color, alpha=0.8)

    # Style all subplots with dynamic y-axis limits
    for ax1, ax2 in axes_pairs:
        ax1.set_xlim(2380, 2000)
        ax2.set_xlim(1800, 1100)
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)
        
        # Style both axes
        for ax in [ax1, ax2]:
            ax.set_xlabel('Wavenumber $(cm^{-1})$', font_properties=font_properties_label)
            ax.tick_params(axis='x', labelsize=font_properties_tick.get_size(), rotation=30)
            ax.tick_params(axis='y', labelsize=font_properties_tick.get_size())
            
            for spine in ax.spines.values():
                spine.set_linewidth(1.25)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_properties_tick)
        
        # Remove right spine from ax1 and left spine from ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # Set the same x-axis tick frequency
        ax1.xaxis.set_major_locator(plt.MultipleLocator(100))  # Adjust 100 to your desired spacing
        ax2.xaxis.set_major_locator(plt.MultipleLocator(100))
        
        # Remove individual x-labels
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        
        # Match y-axis limits
        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)
        ax1.tick_params(right=False)
        ax2.tick_params(left=False, labelleft=False)
        
        # Add break marks (small wavy lines) at top and bottom
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        
        # Create small wavy lines at the end of ax1
        wave_height = 0.015  # Height of each wave section
        for y_pos in [0, 1]:  # Bottom and top of the axis
            y = np.linspace(y_pos-wave_height, y_pos + wave_height, 20)
            x = np.ones_like(y) + 0.005 * np.sin(50 * y)
            ax1.plot(x, y, **kwargs)
        
        # Create small wavy lines at the start of ax2
        kwargs.update(transform=ax2.transAxes)
        for y_pos in [0, 1]:
            y = np.linspace(y_pos- wave_height, y_pos + wave_height, 20)
            x = np.zeros_like(y) + 0.005 * np.sin(50 * y)
            ax2.plot(x, y, **kwargs)
        
        # Add vertical lines for specific wavenumbers
        for ax in [ax1, ax2]:
            if ax == ax1:
                # Add lines for CO region
                ax.axvline(x=2360, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # CO2(g)
                ax.axvline(x=2143, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # CO(g)
                ax.axvline(x=2187, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # CO*
            else:
                pass
    
    fig.text(0.5, 0.02, 'Wavenumber $(cm^{-1})$', ha='center', 
             fontproperties=font_properties_label)
    fig.text(0.02, 0.5, 'Absorbance $(a.u.)$', va='center', 
             rotation='vertical', fontproperties=font_properties_label)
    
    plt.subplots_adjust(wspace=0.1, bottom=0.1, left=0.15, top=0.9)
    plt.savefig(os.path.join(output_path, output_filename), dpi=200)
    plt.close()

if __name__ == "__main__":
    # Get the base directories and csv files
    base_directories, csv_files = get_base_directories()
    
    # Create the plot
    create_combined_plot(
        base_directories=base_directories,
        output_path='./2_drifts',
        output_filename='combined_samples_plot_all_test.png',
        csv_files=csv_files,
        time_points=np.arange(0,60,1),
        interval=1
    )