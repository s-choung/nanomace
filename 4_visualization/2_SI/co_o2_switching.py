import pandas as pd
import matplotlib.pyplot as plt
from util import *
import glob
import os
from matplotlib.lines import Line2D
import numpy as np
# Define the data directory
data_dir = '../resources/mace_exp_raw_data/CO-O2 switching/'

def read_csv_data(file_path,mz=44):
    # Try different encodings
    try:
        df = pd.read_csv(file_path, skiprows=6, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, skiprows=6, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, skiprows=6, encoding='euc-kr')
    
    # Find the temperature column (any column containing 'temp' and 'cell')
    time_col = 'Time(s)'  
    
    # Filter the DataFrame to start after 'Time(s)' > 600
    df = df[df['Time(s)'] > 1635]
    
    # Convert time to minutes by subtracting initial time and dividing by 60
    df['Time(s)'] = (df['Time(s)'] - 1635) / 60
    
    # Extract only the temperature and m/z=44 columns
    return df[[time_col, f'm/z={mz}(A)']]

def find_local_max(df, center_time, span=500,mz=44):
    mask = (df['Time(s)'] >= center_time - span) & (df['Time(s)'] <= center_time + span)
    window_df = df[mask]
    if not window_df.empty:
        max_idx = window_df[f'm/z={mz}(A)'].idxmax()
        return window_df.loc[max_idx]
    return None

def calculate_area_between_maxima(df, time1, time2, mz=44):
    """Calculate the area between two time points using trapezoidal integration"""
    mask = (df['Time(s)'] >= time1) & (df['Time(s)'] <= time2)
    window_df = df[mask]
    
    if window_df.empty:
        return 0
    
    # Calculate dt for each point
    dt = np.diff(window_df['Time(s)'].values)
    y_values = window_df[f'm/z={mz}(A)'].values
    # Use trapezoidal rule for integration
    area = np.sum(0.5 * dt * (y_values[1:] + y_values[:-1]))
    return area

def co_o2_switching(xmax,ymax):
    cmap = plt.cm.inferno
    categories = ['Normal']#, '-20']
    prefixes = ['AM', 'AR','AC']
    color_map = {'AM': 'blue', 'AC': 'green', 'AR': 'red'}
    name_map = {'AM': 'Au/CeO$_2$ Mace', 'AR': 'Au/CeO$_2$ Rod', 'AC': 'Au/CeO$_2$ Cube'}
    mz_values = [28, 32, 44]

    # Get all CSV files from the data directory
    files = glob.glob(os.path.join(data_dir, '*.CSV'))
    
    # Create two separate figures
    figs = []

    for category in categories:
        fig, axes = plt.subplots(1, len(prefixes), figsize=(12, 8), sharex=False, sharey=False)
        figs.append(fig)
        
        for i, structure in enumerate(prefixes):
            matched_files = [f for f in files if structure in os.path.basename(f) and 
                           (('-20' in os.path.basename(f)) == (category == '-20'))]
            print(matched_files)
            file = matched_files[0]
            
            for k, mz in enumerate(mz_values):
                ax = axes[i]  # Get the correct subplot
                df = read_csv_data(file, mz)
                if df is None:
                    continue

                if mz == 44:
                    color = colors[i]
                    marker = 'o'
                    alpha = 1
                elif mz == 28:
                    color = 'gray'
                    marker = 's'
                    alpha = 0.3
                elif mz == 32:
                    color = 'black'
                    marker = 'o'
                    alpha = 0.6

                # Find and print local maxima
                if mz == 44:  # Only print maxima for m/z=44
                    # Convert target times from seconds to minutes
                    target_times = [(t - 1635) / 60 for t in [2000, 5000, 8000, 11000]]
                    max_points = []
                    print(f"\nLocal maxima for {structure}, m/z={mz}:")
                    for target_time in target_times:
                        max_point = find_local_max(df, target_time, 10, mz)  # Adjusted span to minutes (600/60=10)
                        if max_point is not None:
                            max_points.append(max_point)
                            print(f"Time: {max_point['Time(s)']:.1f}min, Signal: {max_point[f'm/z={mz}(A)']:.2e}")
                            ax.scatter(max_point['Time(s)'], max_point[f'm/z={mz}(A)'], color=color, s=10, marker=marker)
                            ax.annotate(f'{max_point[f"m/z={mz}(A)"]}', 
                                      (max_point['Time(s)'], max_point[f'm/z={mz}(A)']), 
                                      textcoords="offset points", xytext=(20,10), 
                                      ha='center', color=color, fontproperties=font_properties_annotate)
                    # Calculate and print areas between consecutive maxima
                    print(f"\nIntegrated areas for {structure}, m/z={mz}:")
                    for i in range(len(max_points)):
                        time1 = max_points[i]['Time(s)']  # Now in minutes
                        if i == len(max_points)-1:
                            time2 = xmax  # xmax is already in minutes
                        else:
                            time2 = max_points[i+1]['Time(s)']
                        area = calculate_area_between_maxima(df, time1, time2, mz)
                        print(f"Area between {time1:.1f}min and {time2:.1f}min: {area:.2e}")

                term = 1
                time_col = 'Time(s)'  
                mz_col = f'm/z={mz}(A)'
                y_data = df[mz_col]

                #ax.scatter(df[time_col][::term], y_data[::term], color=color, s=10, marker=marker, alpha=alpha, label=mz)
                ax.plot(df[time_col][::term], y_data[::term], color=color, alpha=alpha, label=mz)
                ax.set_xlabel("Time (min)", fontproperties=font_properties_label)
                ax.set_ylabel(f"MS signal (a.u.)", fontproperties=font_properties_label)
                ax.set_title(f"{name_map[structure]}", fontproperties=font_properties_label)
                
                # Update tick labels font
                ax.tick_params(axis='both', which='major', labelsize=12)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontproperties(font_properties_tick)
                
                # Update legend font
                ax.legend(prop=font_properties_tick, loc='lower right')

            ax.set_yscale('log')
            if category == 'Normal':
                ax.set_xlim(0, xmax)
                ax.set_ylim(1e-14, ymax)
            else:
                ax.set_xlim(0, xmax)
                ax.set_ylim(1e-14, ymax)
            ax.vlines(x=5, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
            ax.vlines(x=56, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
            ax.vlines(x=107, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
            ax.vlines(x=159, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
            ax.annotate('CO', (25, 0.1*ymax), textcoords="offset points", xytext=(0,10), 
                        ha='center', color='gray', fontproperties=font_properties_annotate)
            ax.annotate('O$_2$', (75, 0.1*ymax), textcoords="offset points", xytext=(0,10), 
                        ha='center', color='gray', fontproperties=font_properties_annotate)
            ax.annotate('CO', (120, 0.1*ymax), textcoords="offset points", xytext=(0,10), 
                        ha='center', color='gray', fontproperties=font_properties_annotate)
            ax.annotate('O$_2$', (170, 0.1*ymax), textcoords="offset points", xytext=(0,10), 
                        ha='center', color='gray', fontproperties=font_properties_annotate)
        plt.tight_layout()
        plt.savefig(f'./1_pngs/co-o2_switching_{category}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_co2_integral(xmax, ymax):
    """Create separate plots showing cumulative CO2 production"""
    data_dir = '../resources/mace_exp_raw_data/CO-O2 switching/'
    categories = ['Normal']#, '-20']
    prefixes = ['AM', 'AR', 'AC']
    name_map = {'AM': 'Au/CeO$_2$ Mace', 'AR': 'Au/CeO$_2$ Rod', 'AC': 'Au/CeO$_2$ Cube'}

    # Get all CSV files
    files = glob.glob(os.path.join(data_dir, '*.CSV'))
    
    # Create figure for each category
    for category in categories:
        fig, ax = plt.subplots(figsize=(6, 5))
        
        for i, structure in enumerate(prefixes):
            # Find matching file
            matched_files = [f for f in files if structure in os.path.basename(f) and 
                           (('-20' in os.path.basename(f)) == (category == '-20'))]
            if not matched_files:
                continue
                
            # Read data
            df = read_csv_data(matched_files[0], mz=44)
            
            # Calculate cumulative integral
            time_values = df['Time(s)'].values
            signal_values = df['m/z=44(A)'].values
            dt = np.diff(time_values)
            incremental_areas = 0.5 * dt * (signal_values[1:] + signal_values[:-1])
            cumulative_integral = np.cumsum(incremental_areas)
            
            # Plot with shifted time
            ax.plot(time_values[1:], cumulative_integral, color=colors[i], 
                   label=name_map[structure], linewidth=2)
        
        # Customize plot
        ax.set_xlabel('Time (min)', fontproperties=font_properties_label)
        ax.set_ylabel('Cumulative CO$_2$ Production (a.u.·s)', fontproperties=font_properties_label)
        
        # Update tick labels font
        ax.tick_params(axis='both', which='major', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_properties_tick)
        ax.vlines(x=5, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
        ax.vlines(x=56, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
        ax.vlines(x=107, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
        ax.vlines(x=159, ymin=1e-14, ymax=ymax, color='gray', linestyle='--', linewidth=1,alpha=0.5)
        arrow_props = dict(arrowstyle='->', color='gray', lw=1.5)
        ax.annotate('CO', (5, 0.05*ymax), (25, 0.2*ymax), 
                    ha='center', color='gray', fontproperties=font_properties_annotate,
                    arrowprops=arrow_props)
        ax.annotate('O$_2$', (56, 0.05*ymax), (75, 0.2*ymax), 
                    ha='center', color='gray', fontproperties=font_properties_annotate,
                    arrowprops=arrow_props)
        ax.annotate('CO', (107, 0.05*ymax), (120, 0.2*ymax), 
                    ha='center', color='gray', fontproperties=font_properties_annotate,
                    arrowprops=arrow_props)
        ax.annotate('O$_2$', (159, 0.05*ymax), (170, 0.2*ymax), 
                    ha='center', color='gray', fontproperties=font_properties_annotate,
                    arrowprops=arrow_props)
        ax.legend(prop=font_properties_tick, loc='upper left')
        ax.set_xlim(0, xmax)  # Convert limit to minutes
        ax.set_ylim(0, ymax)
        plt.tight_layout()
        plt.savefig(f'./1_pngs/co2_integral_{category}.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    xmax = (13000 - 1635) / 60  # 13000 or 8100
    ymax = 2e-10
    co_o2_switching(xmax, 1e-9)
    plot_co2_integral(xmax, ymax)
