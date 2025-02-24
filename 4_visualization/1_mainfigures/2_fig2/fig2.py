import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, concat
from tqdm import tqdm
from util import *  # Import style-related variables
from PIL import Image
import pandas as pd
from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def a(df,figsize=(5, 4)):
    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axis
    labels = df.columns[1:] 
    for i, label in enumerate(labels):
        x = df["Temperature (°C)"]
        y = df[label]
        ax.plot(x, y, label=label, color=colors[i], linewidth=1.5, zorder=2-0.1*i)
        ax.scatter(x, y, color=colors[i], s=100, marker='o', linewidth=1.5, zorder=2-0.1*i)
    ax.set_xlabel("Temperature (°C)", fontproperties=font_properties_label)
    ax.set_ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    ax.tick_params(axis='x', labelsize=font_properties_tick.get_size())
    ax.tick_params(axis='y', labelsize=font_properties_tick.get_size())
    plt.tight_layout()
    return fig, ax  # Return the figure and axis
def b(df,figsize=(5, 4)):
    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and axis
    labels = df.columns[1:] 
    print(labels)
    for i, label in enumerate(labels):
        x = df["Temperature (°C)"].iloc[::2] 
        y = df[label].iloc[::2] 
        ax.plot(x, y, label=label, color=colors[i], linewidth=1.5, zorder=2-0.1*i)
        ax.scatter(x, y, color=colors[i], s=50, marker='o', linewidth=1.5, zorder=2-0.1*i)
    ax.set_xlabel("Time on stream (hours)", fontproperties=font_properties_label)
    ax.set_ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    ax.tick_params(axis='x', labelsize=font_properties_tick.get_size())
    ax.tick_params(axis='y', labelsize=font_properties_tick.get_size())
    plt.tight_layout()
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 80)

    return fig, ax  # Return the figure and axis

def c(df,figsize=(10, 3)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)  # Create a figure and axes
    co_data = df.iloc[:, :4]  # First 4 columns
    o2_data = df.iloc[:, 4:]  # Last 4 columns

    # Plot CO partial pressure
    for i in range(1, len(co_data.columns)):
        x = co_data.iloc[:, 0]
        y = co_data.iloc[:, i]
        coeffs = np.polyfit(np.log(x), np.log(y), 1)  # Linear fit in log-log space
        poly = np.poly1d(coeffs)
        axs[0].plot(x, np.exp(poly(np.log(x))), label=co_data.columns[i], color=colors[i-1])
        axs[0].scatter(co_data.iloc[:, 0], co_data.iloc[:, i], s=50, color=colors[i-1])
    axs[0].set_xlabel("P$_{CO}$(Kpa)", fontproperties=font_properties_label)
    axs[0].set_ylabel("Specific activity\n(μmolCO$_{2}$/g$_{ceria}$s$^{-1}$)", fontproperties=font_properties_label)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylim(0.25, 10)
    axs[0].set_xlim(0.5, 20)
    axs[0].tick_params(axis='x', labelsize=font_properties_tick.get_size())
    axs[0].tick_params(axis='y', labelsize=font_properties_tick.get_size())

    # Plot O2 partial pressure
    for i in range(1, len(o2_data.columns)):
        x = o2_data.iloc[:, 0]
        y = o2_data.iloc[:, i]
        coeffs = np.polyfit(np.log(x), np.log(y), 1)  # Linear fit in log-log space
        poly = np.poly1d(coeffs)
        axs[1].plot(x, np.exp(poly(np.log(x))), label=o2_data.columns[i], color=colors[i-1])
        axs[1].scatter(o2_data.iloc[:, 0], o2_data.iloc[:, i], s=50, color=colors[i-1])
    axs[1].set_xlabel("P$_{O_{2}}$(Kpa)", fontproperties=font_properties_label)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_ylim(0.25, 10)
    axs[1].set_xlim(0.5, 20)
    axs[1].tick_params(axis='x', labelsize=font_properties_tick.get_size())
    axs[1].tick_params(axis='y', labelsize=font_properties_tick.get_size())
    axs[1].tick_params(axis='y', labelsize=font_properties_tick.get_size())

    plt.tight_layout()

    return fig, axs  # Return the figure and axes


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
    matched_rows = data[np.abs(data['cm-1'] - wavenumber) < 0.5]
    if matched_rows.empty:
        return 0.0
    temp_col = str(int(temperature))
    if temp_col not in matched_rows.columns:
        return 0.0
        
    return float(matched_rows[temp_col].iloc[0])

def d(base_directories, output_path, output_filename, csv_files, figsize=(8, 10)):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=figsize, 
                                    gridspec_kw={'width_ratios': [380, 700]})

    axes_pairs = [(ax1[0], ax1[1]), (ax2[0], ax2[1]), (ax3[0], ax3[1])]
    
    # Define colormap and time points
    cmap = plt.cm.inferno
    time_points = [0, 15, 30, 45, 60]
    norm = plt.Normalize(min(time_points), max(time_points))
    time_colors = cmap(np.linspace(0, 0.8, len(time_points)))
    
    # Track maximum absorbance value across all datasets
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
                # Reduce data size for testing
                wavenumber = wavenumber[::10]  ########### Use every second wavenumber
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
                # Reduce data size for testing
                wavenumber = wavenumber[::10]  ########### Use every second wavenumber
                
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
            ax.set_xlabel('Wavenumber $(cm^{-1})$', fontproperties=font_properties_label)
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
                ax.axvline(x=2360, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # CO2(g)
                ax.axvline(x=2143, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # CO(g)
                ax.axvline(x=2187, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # CO*
            else:
                # Add any specific lines for the second region if needed
                pass
    
    fig.text(0.5, 0.02, 'Wavenumber $(cm^{-1})$', ha='center', 
             fontproperties=font_properties_label)
    fig.text(0.02, 0.5, 'Absorbance $(a.u.)$', va='center', 
             rotation='vertical', fontproperties=font_properties_label)
    
    plt.subplots_adjust(wspace=0.1, bottom=0.1, left=0.15, top=0.9)
    return fig, (ax1, ax2, ax3)  # Return the figure and axes


def e(
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

        # Reduce data size for testing
        time = time[::10]  ############ Use every second time point
        wavenumber = wavenumber[::10]  ############ Use every second wavenumber

        # Calculate absorbance
        Absorbance = np.zeros((len(time), len(wavenumber)))
        for i in range(len(wavenumber)):  # Removed tqdm for speed
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
    cbar.set_label('Absorbance (a.u.)', labelpad=10, fontproperties=font_properties_label)
    cbar.ax.tick_params(width=1.25)
    cbar.outline.set_linewidth(1.25)
    for label in cbar.ax.get_yticklabels():
        label.set_font_properties(font_properties_tick)

    # Final adjustments and save
    fig.subplots_adjust(left=0.1, right=0.81, top=0.95, bottom=0.1)
    plt.show()

    return fig, axes  # Return the figure and axes


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
    ax.set_ylabel("Absorbance at 2360 cm$^{-1}$ (a.u.)", fontproperties=font_properties_label)
    ax.tick_params(axis='x', labelsize=font_properties_tick.get_size())
    ax.tick_params(axis='y', labelsize=font_properties_tick.get_size())
    plt.tight_layout()

    return fig, ax  # Return the figure and axis


def combine_all_plots(fig_a, fig_b, fig_c, fig_d, fig_e, fig_f):
    fig = plt.figure(figsize=(10, 12))
    gs0 = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 2], width_ratios=[2, 2])

    # Top left: Figure a
    ax_a = fig.add_subplot(gs0[0, 0])
    canvas = FigureCanvas(fig_a)
    canvas.draw()
    image_a = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_a = image_a.reshape(canvas.get_width_height()[::-1] + (3,))
    ax_a.imshow(image_a)
    ax_a.axis('off')  # Hide axes for figure a

    # Top right: Figures b and c stacked vertically
    gs01 = gs0[0, 1].subgridspec(2, 1)
    ax_b = fig.add_subplot(gs01[0, 0])
    canvas = FigureCanvas(fig_b)
    canvas.draw()
    image_b = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_b = image_b.reshape(canvas.get_width_height()[::-1] + (3,))
    ax_b.imshow(image_b)
    ax_b.axis('off')  # Hide axes for figure b

    ax_c = fig.add_subplot(gs01[1, 0])
    canvas = FigureCanvas(fig_c)
    canvas.draw()
    image_c = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_c = image_c.reshape(canvas.get_width_height()[::-1] + (3,))
    ax_c.imshow(image_c)
    ax_c.axis('off')  # Hide axes for figure c

    # Bottom left: Figure d
    ax_d = fig.add_subplot(gs0[1, 0])
    canvas = FigureCanvas(fig_d)
    canvas.draw()
    image_d = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_d = image_d.reshape(canvas.get_width_height()[::-1] + (3,))
    ax_d.imshow(image_d)
    ax_d.axis('off')  # Hide axes for figure d

    # Bottom right: Figures e and f stacked vertically
    gs11 = gs0[1, 1].subgridspec(2, 1)
    ax_e = fig.add_subplot(gs11[0, 0])
    canvas = FigureCanvas(fig_e)
    canvas.draw()
    image_e = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_e = image_e.reshape(canvas.get_width_height()[::-1] + (3,))
    ax_e.imshow(image_e)
    ax_e.axis('off')  # Hide axes for figure e

    ax_f = fig.add_subplot(gs11[1, 0])
    canvas = FigureCanvas(fig_f)
    canvas.draw()
    image_f = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_f = image_f.reshape(canvas.get_width_height()[::-1] + (3,))
    ax_f.imshow(image_f)
    ax_f.axis('off')  # Hide axes for figure f

    plt.tight_layout()
    plt.savefig('./combined_plots.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    file_path = '../resources/240527_source_data/240527_source_data.xlsx'
    df_a = pd.read_excel(file_path, sheet_name=0, skiprows=1)
    df_b = pd.read_excel(file_path, sheet_name='Fig.2d', skiprows=1)
    df_c = pd.read_excel(file_path, sheet_name='Fig.2b', skiprows=1)
    fig_a, ax_a = a(df_a,figsize=(5, 4))
    fig_b, ax_b = b(df_b,figsize=(5, 2))
    fig_c, ax_c = c(df_c,figsize=(5, 2))
    base_directories, csv_files = get_base_directories()

    # Configuration
    CSV_DIR = "/Users/sean/Library/CloudStorage/OneDrive-postech.ac.kr/연구/1_projects_mace/resources/240527_source_data/DRIFTS_CeO2_CO-O2_step_reaction"
    CSV_FILES = ['drift_CM.csv', 'drift_CR.csv', 'drift_CC.csv']

    fig_d, (ax_d1, ax_d2, ax_d3) = d(
        base_directories=base_directories,
        output_path='./2_drifts',
        output_filename='combined_samples_plot_all.png',
        csv_files=csv_files,
        figsize=(5, 8)  # Don't forget to pass csv_files as well       
    )
    fig_e, ax_e = e(
        csv_directory=CSV_DIR,
        csv_files=CSV_FILES,
        figsize=(5, 4)
    )
    fig_f, ax_f = f(csv_directory=CSV_DIR,
        csv_files=CSV_FILES,
        figsize=(5, 4),
        labels=['CM', 'CR', 'CC'])
    combine_all_plots(fig_a, fig_b, fig_c, fig_d, fig_e, fig_f)
    # 이거 할려다가 plot 별로 모양맞추는거 힘들어서 잠깐 넘어감 12/31 2024