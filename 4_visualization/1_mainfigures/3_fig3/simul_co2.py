import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
import json,pickle
from matplotlib.markers import MarkerStyle

def process_co2_data(co2_time_dict, area_list, time_list=None, use_sem=False):
    """
    Process CO2 conversion data and return relevant statistics
    
    Args:
        co2_time_dict: Dictionary containing CO2 conversion data
        area_list: List of areas [cube_area, rod_area, interface_area, total_area]
        time_list: Optional list of specific time points to extract
        use_sem: If True, returns Standard Error of Mean instead of Standard Deviation
        
    Returns:
        tuple containing processed data with either SEM or SD for error values
    """
    cube_list, rod_list, interface_list, tot_list = [], [], [], []
    time_values = []
    
    # Extract data
    for i in co2_time_dict.keys():
        case_data = co2_time_dict[i]
        case_times = []
        case_cube = []
        case_rod = []
        case_interface = []
        case_tot = []
        
        for time, values in case_data.items():
            case_times.append(int(time))
            case_cube.append(values['cube']/area_list[0])
            case_rod.append(values['rod']/area_list[1])
            case_interface.append(values['interface']/area_list[2])
            case_tot.append(values['tot']/area_list[3])
            
        cube_list.append(case_cube)
        rod_list.append(case_rod)
        interface_list.append(case_interface)
        tot_list.append(case_tot)
        time_values = case_times  # All cases have same times

    # Calculate standard deviations
    n = len(cube_list)  # number of samples
    stds = {
        'cube': np.std(cube_list, axis=0),
        'rod': np.std(rod_list, axis=0),
        'interface': np.std(interface_list, axis=0),
        'tot': np.std(tot_list, axis=0)
    }
    
    # Convert to SEM if requested
    if use_sem:
        for key in stds:
            stds[key] = stds[key] / np.sqrt(n)

    # Calculate means
    means = {
        'cube': np.mean(cube_list, axis=0),
        'rod': np.mean(rod_list, axis=0),
        'interface': np.mean(interface_list, axis=0),
        'tot': np.mean(tot_list, axis=0)
    }

    if time_list is None:
        return time_values, means, stds
        
    # Get values at specified time points
    time_indices = [time_values.index(t) for t in time_list]
    values = {
        'cube': means['cube'][time_indices],
        'rod': means['rod'][time_indices],
        'interface': means['interface'][time_indices]
    }
    errors = {
        'cube': stds['cube'][time_indices],
        'rod': stds['rod'][time_indices],
        'interface': stds['interface'][time_indices]
    }
    
    return time_values, means, stds, values, errors
def std_mean_cal(co2_time_dict, area, use_sem=False):
    tot_list = []
    for i in co2_time_dict.keys():
        case_data = co2_time_dict[i]
        time_values = []
        tot_values = []
        for time, values in case_data.items():
            time_values.append(int(time))
            tot_values.append(values['tot']/area)
        tot_list.append(tot_values)
    
    tot_mean_list = np.mean(tot_list, axis=0)
    tot_std_list = np.std(tot_list, axis=0)
    
    # Convert to SEM if requested
    if use_sem:
        n = len(tot_list)  # number of samples
        tot_std_list = tot_std_list / np.sqrt(n)
    
    return time_values, tot_std_list, tot_mean_list

def time_evolution_plt(co2_time_dict, area_list=[1,1,1,1], title=r'CO conversion (${CO}$/${nm^2}$)', name='temp', scatter_interval=5):
    # Get processed data
    time_values, means, stds = process_co2_data(co2_time_dict, area_list,use_sem=True)
    
    plt.figure(figsize=(6, 4))
    
    # Plot each site type
    for site, color, label in zip(['rod', 'cube', 'interface'], 
                                [colors[1], colors[2], colors[0]], 
                                ['rod site', 'cube site', 'interface site']):
        plt.plot(time_values[::scatter_interval], means[site][::scatter_interval], 
                label=label, color=color, linewidth=1.5)
        plt.scatter(time_values[::scatter_interval], means[site][::scatter_interval], 
                   color=color, s=70, linewidth=1.5)
        plt.fill_between(time_values[::scatter_interval], 
                        (means[site] - stds[site])[::scatter_interval], 
                        (means[site] + stds[site])[::scatter_interval], 
                        alpha=0.2, color=color)

    plt.xlim(0,1000)
    plt.xlabel('Time(ps)', fontproperties=font_properties_label)
    plt.ylabel(title, fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.tight_layout()
    plt.ylim(0, 16)
    plt.savefig(f'./1_pngs/time_vs_co_conv_{name}.png', dpi=500)
    plt.show()

def accumulative_time_evolution_plt(co2_time_dict, area_list=[1,1,1,1], title=r'Accumulative CO conversion', scatter_interval=5):
    # Get processed data
    time_values, means, stds = process_co2_data(co2_time_dict, area_list,use_sem=True)
    
    # Calculate accumulative values
    cube_accum = means['cube']
    cube_rod_accum = means['cube'] + means['rod']
    cube_rod_interface_accum = means['cube'] + means['rod'] + means['interface']
    
    cube_lst=means['cube'][-1]
    rod_lst=means['rod'][-1]
    interface_lst=means['interface'][-1]
    tot_lst=cube_rod_interface_accum[-1]
    print('cube:rod:interface', cube_lst, rod_lst, interface_lst)
    print('cube:rod:interface', cube_lst/tot_lst, rod_lst/tot_lst, interface_lst/tot_lst)

    plt.figure(figsize=(6, 4))
    alphas=[0.1,0.4,0.7]
    # Ensure the last point is included by modifying the time indices
    plot_indices = list(range(0, len(time_values), scatter_interval))
    if (len(time_values) - 1) not in plot_indices:
        plot_indices.append(len(time_values) - 1)
    
    # Use plot_indices instead of ::scatter_interval
    for bottom, top, alpha in [(np.zeros_like(cube_accum), cube_accum, alphas[0]),
                              (cube_accum, cube_rod_accum, alphas[1]),
                              (cube_rod_accum, cube_rod_interface_accum, alphas[2])]:
        plt.fill_between(np.array(time_values)[plot_indices], 
                        bottom[plot_indices], 
                        top[plot_indices], 
                        color=colors[0], alpha=alpha,edgecolor=None)

    # Plot error bars
    marker_list = ['o', 's', '^']
    labels = labels=['Interface site', 'Rod site', 'Cube site']
    accum_values = [cube_rod_interface_accum,cube_rod_accum,cube_accum, cube_rod_accum ]
    
    for i, (values, label, marker) in enumerate(zip(accum_values, labels, marker_list)):
        combined_std = np.sqrt(sum(stds[site]**2 for site in list([ 'interface', 'rod','cube'])[:i+1]))
        plt.errorbar(np.array(time_values)[plot_indices], values[plot_indices],
                    yerr=combined_std[plot_indices],
                    color=colors[0], capsize=2.5,alpha=alphas[2-i]+0.3)
        plt.scatter(np.array(time_values)[plot_indices], values[plot_indices],
                   color=colors[0], s=100, label=label, marker=marker, zorder=10,alpha=alphas[2-i]+0.3)
    plt.xlabel('Time(ps)', fontproperties=font_properties_label)
    plt.ylabel(title, fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.xlim(0, 1000)
    plt.ylim(0, 16)
    plt.legend( prop=font_properties_tick)
    plt.tight_layout()
    plt.savefig('./1_pngs/accumulative_time_vs_co_conv.png', dpi=500)
    
    plt.show()

def accumulative_barplot(co2_time_dict, time_list=[200, 500, 1000], area_list=[1,1,1,1], title='CO Conversion'):
    # Get processed data
    _, _, _, values, errors = process_co2_data(co2_time_dict, area_list, time_list,use_sem=True)
    
    plt.figure(figsize=(8, 4))
    bar_width = 0.8
    x = np.arange(len(time_list))

    # Calculate accumulative values
    rod_accum = values['cube'] + values['rod']
    interface_accum = rod_accum + values['interface']

    # Create stacked bars
    plt.bar(x, values['cube'], bar_width,
            label='Cube Site', color=colors[2], yerr=errors['cube'], capsize=5)
    plt.bar(x, values['rod'], bar_width,
            bottom=values['cube'], label='Rod Site', color=colors[1],
            yerr=np.sqrt(errors['cube']**2 + errors['rod']**2), capsize=5)
    plt.bar(x, values['interface'], bar_width,
            bottom=rod_accum, label='Interface Site', color=colors[0],
            yerr=np.sqrt(errors['cube']**2 + errors['rod']**2 + errors['interface']**2), capsize=5)

    plt.xlabel('Time(ps)', fontproperties=font_properties_label)
    plt.ylabel(title, fontproperties=font_properties_label)
    plt.xticks(x, time_list, fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.legend(prop=font_properties_tick)
    plt.tight_layout()
    plt.savefig('./1_pngs/accumulative_barplot.png', dpi=500)
    plt.show()
def time_evolution_plt_tot(co2_time_dict_list, area_list=[1,1,1], title='hi', scatter_interval=5):
    names = ['mace', 'rod', 'cube']
    plt.figure(figsize=(8, 4))
    markers=['o','o','o']#, 's', '^']

    #markers = ['s', MarkerStyle('s', fillstyle='bottom') ,MarkerStyle('P', fillstyle='top') ]  # Define a list of marker shapes
    for i, co2_time_dict in enumerate(co2_time_dict_list):
        time_values, tot_std_list, tot_mean_list = std_mean_cal(co2_time_dict, area_list[i])#,use_sem=True)
        # Ensure the last point is included by modifying the time indices
        plot_indices = list(range(0, len(time_values), scatter_interval))
        if len(time_values) - 1 not in plot_indices:
            plot_indices.append(len(time_values) - 1)
        plt.plot(np.array(time_values)[plot_indices], np.array(tot_mean_list)[plot_indices], label=names[i], color=colors[i], linewidth=1.5)
        plt.scatter(np.array(time_values)[plot_indices], np.array(tot_mean_list)[plot_indices], 
                    s=70, color=colors[i], linewidth=1.5, marker=markers[i])
        plt.fill_between(np.array(time_values)[plot_indices], [tot_mean_list[j]-tot_std_list[j] for j in plot_indices], 
                         [tot_mean_list[j]+tot_std_list[j] for j in plot_indices], 
                         alpha=0.2, color=colors[i])


    plt.xlabel("Time(ps)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.xlim(0,1000)
    plt.ylim(0, 16)

    #plt.savefig(f'./1_pngs/fig3_a.png', dpi=200)
    plt.show()
def time_evolution_plt_physical_mix(co2_time_dict_list, area_list=[1,1,1], title='hi', scatter_interval=5,ratios=[1,1]):
    names = ['mace', 'rod', 'cube']
    plt.figure(figsize=(6, 4))
    markers=['o','o','o']
    labels=['Mace', 'Rod', 'Cube']
    # Store rod and cube data for summing later
    rod_data = None
    cube_data = None
    
    for i, co2_time_dict in enumerate(co2_time_dict_list):
        time_values, tot_std_list, tot_mean_list = std_mean_cal(co2_time_dict, area_list[i])
        plot_indices = list(range(0, len(time_values), scatter_interval))
        if len(time_values) - 1 not in plot_indices:
            plot_indices.append(len(time_values) - 1)
            
        # Store rod and cube data
        if names[i] == 'rod':
            rod_data = (np.array(tot_mean_list), np.array(tot_std_list))
        elif names[i] == 'cube':
            cube_data = (np.array(tot_mean_list), np.array(tot_std_list))
            
        plt.plot(np.array(time_values)[plot_indices], np.array(tot_mean_list)[plot_indices], 
                label=labels[i], color=colors[i], linewidth=1.5,zorder=2)
        plt.scatter(np.array(time_values)[plot_indices], np.array(tot_mean_list)[plot_indices], 
                    s=70, color=colors[i], linewidth=1.5, marker=markers[i],zorder=1)
        plt.fill_between(np.array(time_values)[plot_indices], 
                        [tot_mean_list[j]-tot_std_list[j] for j in plot_indices], 
                        [tot_mean_list[j]+tot_std_list[j] for j in plot_indices], 
                        alpha=0.3, color=colors[i],zorder=1)

    # Add rod+cube sum line if we have both datasets
    if rod_data is not None and cube_data is not None:
        sum_mean = rod_data[0]*ratios[0] + cube_data[0]*ratios[1]
        sum_std = np.sqrt(rod_data[1]**2*ratios[0]**2 + cube_data[1]**2*ratios[1]**2)  # Error propagation
        
        plt.plot(np.array(time_values)[plot_indices], sum_mean[plot_indices], 
                label='Physically mixed', color='gray', linewidth=1.5,linestyle='--',zorder=2)


    plt.xlabel("Time(ps)", fontproperties=font_properties_label)
    plt.ylabel(title, fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.xlim(0,1000)
    plt.ylim(0, 16)
    plt.legend(prop=font_properties_tick)  # Added legend to show all lines
    plt.tight_layout()
    plt.savefig('./1_pngs/time_evolution_plt_physical_mix.png', dpi=500)
    plt.show()
def scatter_time_plot(co2_time_dict, time_list=[200, 500, 1000], area_list=[1,1,1,1], title='CO Conversion'):
    _, _, _, values, errors = process_co2_data(co2_time_dict, area_list, time_list,use_sem=True)
    
    plt.figure(figsize=(3, 4))
    x = np.arange(3)  # 3 positions for cube, rod, interface[0,0,0]
    markers = ['^', 's','o']   # Different markers for each time stamp
    sizes=[100,100,100]
    #markers = ['s', MarkerStyle('s', fillstyle='bottom') ,MarkerStyle('P', fillstyle='top') ]  # Define a list of marker shapes
    #sizes=[100,150,200]
    # Plot for each time point
    
    for i, time in enumerate(time_list):
        # Adjust x values to add space between scatter points
        adjusted_x = [x[j] + (i-1) * 1/5 for j in range(3)]  # Add a small offset for each time point
        
        for j, site in enumerate(['cube', 'rod', 'interface']):
            plt.scatter(adjusted_x[j], values[site][i], 
                       color=colors[0], alpha=1/len(time_list)+i*1/len(time_list), 
                       marker=markers[j], 
                       s=sizes[j], 
                       label=f'{time_list[i]} ps' if j == 0 else "",
                       zorder=2)
            plt.errorbar(adjusted_x[j], values[site][i], 
                        yerr=errors[site][i], 
                        color=colors[0], alpha=1/len(time_list)+i*1/len(time_list),
                        capsize=5,
                        zorder=1)
            

    # Customize the plot
    plt.ylabel(title, fontproperties=font_properties_label)
    plt.xticks(np.array(x), ['Cube\nsite', 'Rod\nsite', 'Interface\nsite'], fontproperties=font_properties_tick, rotation=30)
    plt.yticks(fontproperties=font_properties_tick)
    plt.tight_layout()
    plt.xlim(-0.5,2.5)
    plt.ylim(0,0.5)
    plt.legend(prop=font_properties_tick)
    plt.savefig('./1_pngs/time_scatter_plot.png', dpi=200)
    plt.show()

def final_values_barplot(co2_time_dict_list, area_list=[1,1,1], ratios=[1,1],title='CO$_2$ count'):
    # Get final values for MACE, Rod, and Cube
    final_values = []
    final_errors = []
    
    for i, co2_time_dict in enumerate(co2_time_dict_list):
        _, tot_std_list, tot_mean_list = std_mean_cal(co2_time_dict, area_list[i])
        final_values.append(tot_mean_list[-1])
        final_errors.append(tot_std_list[-1])
    
    # Calculate physically mixed value
    phys_mix_value = final_values[1] * ratios[0] + final_values[2] * ratios[1]
    phys_mix_error = np.sqrt((final_errors[1] * ratios[0])**2 + (final_errors[2] * ratios[1])**2)
    
    # Reorder values: Cube, Rod, Physically mixed, MACE
    names = ['Cube', 'Rod', 'Physically\nmixed', 'Mace']
    ordered_values = [final_values[2], final_values[1], phys_mix_value, final_values[0]]
    ordered_errors = [final_errors[2], final_errors[1], phys_mix_error, final_errors[0]]
    colors_ordered = [colors[2], colors[1], 'gray', colors[0]]
    
    # Create bar plot
    plt.figure(figsize=(4, 4))
    x = np.arange(len(names))
    
    bars = plt.bar(x, ordered_values, capsize=5, color=colors_ordered, alpha=0.5)
    
    # Fix: Loop through each point to plot individual error bars and add text labels
    for i in range(len(x)):
        plt.errorbar(x[i], ordered_values[i], yerr=ordered_errors[i], 
                    capsize=5, color=colors_ordered[i], zorder=1)
        plt.scatter(x[i], ordered_values[i], color=colors_ordered[i], s=0, zorder=2)
        
        # Add text label above each bar (including error)
        label_height = ordered_values[i] + ordered_errors[i] + 0.2  # Adjust the 0.5 offset as needed
        plt.text(x[i], label_height, 
                f'{ordered_values[i]:.1f} ± {ordered_errors[i]:.1f}', 
                ha='center', va='bottom',fontproperties=font_properties_tick)  # Adjust fontsize as needed
    
    plt.ylabel(title, fontproperties=font_properties_label)
    plt.xticks(x, names, fontproperties=font_properties_tick,rotation=20)
    plt.yticks(fontproperties=font_properties_tick)
    plt.ylim(0, 16)
    
    plt.tight_layout()
    plt.savefig('./1_pngs/final_values_barplot.png', dpi=500)
    plt.show()
    
    # Print the values
    for name, value, error in zip(names, ordered_values, ordered_errors):
        print(f"{name}: {value:.2f} ± {error:.2f}")

with open('../resources/mace_co2_time_dict.pkl', 'rb') as pkl_file:
    co2_time_dict = pickle.load(pkl_file)
co2_dict_mace = {
    str(system_id): {
        str(time): {'tot': values['tot']}
        for time, values in times.items()
    }
    for system_id, times in co2_time_dict.items()
}
with open('../resources/co2_time_data.json', 'r') as json_file:
    co2_time_dict_list = json.load(json_file)

co2_time_dict_list_new=[co2_dict_mace,co2_time_dict_list[1],co2_time_dict_list[2]]

cube_x_y=27
int_z=11
cube_area=24/10*cube_x_y/10 * 4 + cube_x_y/10*cube_x_y/10 
rod_area=55.15/10*55.71/10  -   cube_x_y/10*cube_x_y/10
rod_area_original=55.15/10*55.71/10
int_area=cube_x_y/10 * 4 * int_z/10
totarea=cube_area+rod_area+int_area
area_list=[cube_area,rod_area,int_area,totarea]

rod_ratio=rod_area/rod_area_original
print(rod_ratio)
cube_ratio=5/6
#accumulative_barplot(co2_time_dict, [200, 500, 1000], [1,1,1,1], 'CO Conversion')
#time_evolution_plt(co2_time_dict,area_list,r'CO conversion/nm$^{2}$','site')
#time_evolution_plt_tot(co2_time_dict_list_new,[1,1,1,1],'CO conversion', scatter_interval=2)
#time_evolution_plt_physical_mix(co2_time_dict_list_new,[1,1,1,1],'CO$_2$ count', scatter_interval=2,ratios=[rod_ratio,cube_ratio])
accumulative_time_evolution_plt(co2_time_dict, [1,1,1,1], 'CO$_2$ count', scatter_interval=2)
#scatter_time_plot(co2_time_dict, [200,500,1000], area_list, r'CO$_2$ count/nm$^{2}$')   
#final_values_barplot(co2_time_dict_list_new, [1,1,1], ratios=[rod_ratio, cube_ratio],title='CO$_2$ count')
