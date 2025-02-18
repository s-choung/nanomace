import pandas as pd
import matplotlib.pyplot as plt
from util import *
import glob
import os
from matplotlib.lines import Line2D
import numpy as np
# Define the data directory
data_dir = '../../../resources/mace_exp_raw_data/CO-O2 switching/'

def read_csv_data(file_path, mz=44):
    try:
        df = pd.read_csv(file_path, skiprows=6, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, skiprows=6, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, skiprows=6, encoding='euc-kr')
    
    time_col = 'Time(s)'
    
    # Filter data after 1625s
    df = df[df['Time(s)'] > 1625]
    
    # Calculate slope (derivative)
    time_values = df['Time(s)'].values
    signal_values = df[f'm/z={mz}(A)'].values
    slopes = np.diff(signal_values) / np.diff(time_values)
    
    # Find where slope starts increasing significantly
    # Using rolling mean of slopes to smooth out noise
    window_size = 60
    rolling_slopes = pd.Series(slopes).rolling(window=window_size).mean()
    threshold = np.percentile(rolling_slopes.dropna(), 95) * 0.1  # 10% of 95th percentile
    
    # Find first point where slope exceeds threshold
    onset_idx = rolling_slopes[rolling_slopes > threshold].index[0] + 1  # +1 because of diff
    start_time = df.iloc[onset_idx]['Time(s)']
    
    print(f"Onset detected at {start_time} seconds for file: {os.path.basename(file_path)}")
    
    # Shift time to start from 0 at the onset
    df = df[df['Time(s)'] >= start_time]
    df['Time(s)'] = (df['Time(s)'] - start_time) / 60  # Convert to minutes
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


def plot_co2_integral(xmax, ymax):
    data_dir = '../../../resources/mace_exp_raw_data/CO-O2 switching/'
    categories = ['Normal']#, '-20']
    prefixes = ['AM', 'AR', 'AC']
    name_map = {'AM': 'Au/CeO$\mathregular{_2}$ Mace', 'AR': 'Au/CeO$\mathregular{_2}$ Rod', 'AC': 'Au/CeO$\mathregular{_2}$ Cube'}

    files = glob.glob(os.path.join(data_dir, '*.CSV'))
    for category in categories:
        fig, ax = plt.subplots(figsize=(6, 6))
        
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
            
            # Plot the main data with high zorder
            ax.plot(time_values[1:], cumulative_integral, color=colors[i], 
                   label=name_map[structure], linewidth=2, zorder=5)
            
            # Add vertical lines with lower zorder
            if structure == 'AM':
                # Calculate slopes of cumulative integral
                slopes = np.diff(cumulative_integral) / np.diff(time_values[1:])
                
                # Smooth slopes using rolling mean with smaller window
                window_size = 15  # reduced from 30
                rolling_slopes = pd.Series(slopes).rolling(window=window_size).mean()
                
                # Find points of rapid change with lower threshold
                threshold = np.std(rolling_slopes.dropna()) * 1.2  # reduced multiplier from 2
                slope_changes = []
                
                # Define regions of interest
                regions = [(25, 35), (50, 60), (75, 85)]  # time ranges in minutes
                
                for region_start, region_end in regions:
                    region_mask = (time_values[1:-1] >= region_start) & (time_values[1:-1] <= region_end)
                    region_slopes = rolling_slopes[region_mask]
                    
                    if not region_slopes.empty:
                        # Find the point of maximum slope change in each region
                        max_change_idx = np.argmax(np.abs(np.diff(region_slopes)))
                        time_idx = region_slopes.index[max_change_idx]
                        slope_changes.append(time_values[time_idx+1])
                
                # Add vertical lines
                for x_val in slope_changes:
                    ax.axvline(x=x_val,color='gray', linestyle='--', alpha=0.5, zorder=1)
        
        # Customize plot
        ax.set_xlabel('Time (min)', fontproperties=font_properties_label)
        ax.set_ylabel('Cumulative $\mathregular{CO_2}$ Evolution (×10$\mathregular{^{-11}}$ a.u.)', 
                     fontproperties=font_properties_label)
        
        # Format y-axis values to show whole numbers (multiply by 1e11 for similar digit count as co_conversion)
        ax.yaxis.set_major_formatter(lambda x, pos: f'{int(x*1e11)}')
        
        # Update tick labels font
        ax.tick_params(axis='both', which='major', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_properties_tick)

        ax.legend(prop=font_properties_tick, loc='lower right',facecolor='white',edgecolor='none')

        ax.set_xlim(0, 100)
        ax.set_ylim(0, ymax)
        ax.set_position([0.2, 0.2, 0.666, 0.666])

        plt.savefig(f'./output/co2_integral_{category}.png', dpi=200)
        plt.show()

if __name__ == "__main__":
    xmax = (13000 - 1635) / 60  # 13000 or 8100
    ymax = 1.2*1e-10
    plot_co2_integral(xmax, ymax)
