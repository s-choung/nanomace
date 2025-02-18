import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, concat
from tqdm import tqdm
from util import *  # Import style-related variables
from PIL import Image

def get_base_directories(dataset_type):
    base_path = "/Users/sean/Library/CloudStorage/OneDrive-postech.ac.kr/연구/1_projects_mace/resources/240527_source_data"
    
    if dataset_type == 'co_o2':
        folder_name = 'drift_au_co_o2'
    else:
        folder_name = 'drift_au_co'
    
    base_directories = {
        'AM': os.path.join(base_path, folder_name, 'AM'),
        'AR': os.path.join(base_path, folder_name, 'AR'),
        'AC': os.path.join(base_path, folder_name, 'AC')
    }
    
    csv_files = ['1.csv', '2.csv', '3.csv']
    
    return base_directories, csv_files

def get_absorbance(data, wavenumber, temperature):

    matched_rows = data[np.abs(data['cm-1'] - wavenumber) < 0.5]
    if matched_rows.empty:
        print('no matched rows')
        return 0.0
    
    temp_col = str(int(temperature))
    if temp_col not in matched_rows.columns:
        print('no time col')
        return 0.0
        
    return float(matched_rows[temp_col].iloc[0])

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

def combined(base_directories, output_path, output_filename, csv_files, xtick_onlyone=False,time_points=[],interval=1):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(12, 10), 
                                    gridspec_kw={'width_ratios': [380, 700]})

    axes_pairs = [(ax1[0], ax1[1]), (ax2[0], ax2[1]), (ax3[0], ax3[1])]
    
    # Define colormap and time points
    cmap = plt.cm.inferno
    time_points = time_points
    norm = plt.Normalize(min(time_points), max(time_points))
    time_colors = cmap(np.linspace(0, 0.8, len(time_points)))
    
    
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
                # Reduce data size for testing
                wavenumber = wavenumber[::interval]  ########### Use every second wavenumber
                for mask in [(wavenumber >= 2000) & (wavenumber <= 2380),
                           (wavenumber >= 1100) & (wavenumber <= 1800)]:
                    selected_wavenumbers = wavenumber[mask]
                    absorbance_values = [get_absorbance(data, wave, time) 
                                       for wave in selected_wavenumbers]

    
    y_max = 0.4
    
    # Update colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.75, 0.935, 0.15, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xlabel('Time (min)',  fontproperties=font_properties_label, labelpad=5)
    cbar.ax.xaxis.set_label_position('top')

    cbar.ax.tick_params(labelsize=font_properties_annotate.get_size())
    cbar.ax.tick_params(axis='x', labelsize=font_properties_annotate.get_size())
    for label in cbar.ax.get_xticklabels():
        label.set_fontproperties(font_properties_annotate)
    cbar.ax.yaxis.set_ticks([])

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
                
                # Find and plot local maxima for time=15
                if time == time_points[-1]:
                    peak_wavenumbers, peak_values = find_local_maxima(
                        selected_wavenumbers, absorbance_values)
                    for wave, value in zip(peak_wavenumbers, peak_values):
                        if wave < 2300:
                            ax1.plot([wave, wave], [value-0.01, value+0.03], color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                            ax1.text(wave, value+0.03, f'{wave:.0f}', fontproperties=font_properties_annotate, color='black', ha='center', va='bottom')
                # Plot second region (1800-1100)
                mask2 = (wavenumber >= 1100) & (wavenumber <= 1800)
                selected_wavenumbers = wavenumber[mask2]
                absorbance_values = [get_absorbance(data, wave, time) 
                                   for wave in selected_wavenumbers]
                ax2.plot(selected_wavenumbers, absorbance_values, 
                        linewidth=1.5, color=color, alpha=0.8)
                # Find and plot local maxima for time=15
                if time == time_points[-1]:
                    peak_wavenumbers, peak_values = find_local_maxima(
                        selected_wavenumbers, absorbance_values)
                    for wave, value in zip(peak_wavenumbers, peak_values):
                        ax2.plot([wave, wave], [value-0.01, value+0.05], color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                        ax2.text(wave, value+0.05, f'{wave:.0f}', fontproperties=font_properties_annotate, color='black', ha='center', va='bottom')

    # Style all subplots with dynamic y-axis limits
    for (ax1, ax2), is_last_row in zip(axes_pairs, [False, False, True]):
        ax1.set_xlim(2380, 2000)
        ax2.set_xlim(1800, 1100)
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)
        
        # Style both axes
        for ax in [ax1, ax2]:
            # Show x-ticks only for last row if xtick_onlyone is True
            if xtick_onlyone and not is_last_row:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', labelsize=font_properties_tick.get_size(), rotation=30)
            
            ax.tick_params(axis='y', labelsize=font_properties_tick.get_size())

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
        
        # Remove right y-axis ticks from left plot and left y-axis ticks from right plot
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
        
        kwargs.update(transform=ax2.transAxes)
        for y_pos in [0, 1]:
            y = np.linspace(y_pos- wave_height, y_pos + wave_height, 20)
            x = np.zeros_like(y) + 0.005 * np.sin(50 * y)
            ax2.plot(x, y, **kwargs)
        
        for ax in [ax1, ax2]:
            if ax == ax1:
                # Add lines for CO region
                ax.axvline(x=2360, color='gray', linestyle='--', linewidth=1, alpha=0.5)  # CO2(g)
                ax.axvline(x=2143, color='gray', linestyle='--', linewidth=1, alpha=0.5)  # CO(g)
            else:
                # Add any specific lines for the second region if needed
                pass
        plt.subplots_adjust(
        wspace=0.15,
        hspace=0.1,
        bottom=0.15,
        left=0.15,
        top=0.9
          )   

    fig.supxlabel( 'Wavenumber $(cm^{-1})$', fontproperties=font_properties_label, x=0.5,y=0.08)
    fig.supylabel( 'Absorbance $(a.u.)$', fontproperties=font_properties_label, x=0.08,y=0.52)
    plt.savefig(f'./output/{output_filename}', dpi=200)
    #plt.show()
    return fig, (ax1, ax2, ax3)  # Return the figure and axes


if __name__ == "__main__":
    base_directories, csv_files = get_base_directories('co')
    fig_d, (ax_d1, ax_d2, ax_d3) = combined(
        base_directories=base_directories,
        output_path='./output',
        output_filename='d.png',
        csv_files=csv_files,
        xtick_onlyone=True,time_points=np.arange(0,15,1),interval=1  # Set your preferred value here
    )
    base_directories, csv_files = get_base_directories('co_o2')
    fig_e, (ax_e1, ax_e2, ax_e3) = combined(
        base_directories=base_directories,
        output_path='./output',
        output_filename='e.png',
        csv_files=csv_files,
        xtick_onlyone=True,time_points=np.arange(0,62,2),interval=1  # Set your preferred value here
    )
