import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
from ase import Atoms
from ase.io import Trajectory, write, read

import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
def periodic_converter_traj(traj):

    x = traj[0].cell[0][0]
    y = traj[0].cell[1][1]
    z = traj[0].cell[2][2]
    print(x, y, z)
    new_traj=[]
    for atoms in traj:
        for atom in atoms:
            #pbc_fixed_atom= atom.copy()
            # Apply PBC for the x-axis
            if atom.position[1] > y :
                atom.position = atom.position - [0, int(atom.position[1]/y)*y, 0]
            elif atom.position[1] < 0 :       
                atom.position = atom.position - [0, (int(atom.position[1]/y)-1)*y, 0]
            else:
                continue
        #print('miny : ',np.min(atoms.positions[:,1]))
        new_traj.append(atoms)
    print(f'Y_axis traj is done') 
    new_traj_xy=[]
    for atoms in new_traj:
        for atom in atoms:
            #pbc_fixed_atom= atom.copy()
            # Apply PBC for the x-axis
            if atom.position[0] > x :
                atom.position = atom.position - [int(atom.position[0]/x)*x, 0, 0]
            elif atom.position[0] < 0 :       
                atom.position = atom.position - [(int(atom.position[0]/x)-1)*x, 0, 0]
            else:
                continue
        #print('minx : ',np.min(atoms.positions[:,0]))
        new_traj_xy.append(atoms)
    print(f'Xaxis_traj is done') 
    return new_traj_xy

def periodic_converter_atoms(atoms):
        x = atoms.cell[0][0]
        y = atoms.cell[1][1]
        z = atoms.cell[2][2]
        for atom in atoms:
            #pbc_fixed_atom= atom.copy()
            # Apply PBC for the x-axis
            if atom.position[1] > y :
                atom.position = atom.position - [0, int(atom.position[1]/y)*y, 0]
            elif atom.position[1] < 0 :       
                atom.position = atom.position - [0, (int(atom.position[1]/y)-1)*y, 0]
            else:
                continue
        for atom in atoms:
            #pbc_fixed_atom= atom.copy()
            # Apply PBC for the x-axis
            if atom.position[0] > x :
                atom.position = atom.position - [int(atom.position[0]/x)*x, 0, 0]
            elif atom.position[0] < 0 :       
                atom.position = atom.position - [(int(atom.position[0]/x)-1)*x, 0, 0]
            else:
                continue
        for atom in atoms:
            #pbc_fixed_atom= atom.copy()
            # Apply PBC for the x-axis
            if atom.position[2] > z :
                atom.position = atom.position - [0, 0, int(atom.position[2]/z)*z]
            elif atom.position[2] < 0 :       
                atom.position = atom.position - [0, 0, (int(atom.position[2]/z)-1)*z]
            else:
                continue
        return atoms

def co2_detector(atoms):
    cut_off=1.5
    C_indice = np.array([atom.index for atom in atoms if atom.symbol == 'C'])
    O_indice = np.array([atom.index for atom in atoms if atom.symbol == 'O' and atom.position[2]>5])
    c_dictionary = {}

    for c_index in tqdm(C_indice, desc="Carbon coord calculation"):
        c_position = atoms[c_index].position
        c_o_bonds = []
        O_in_interest=[o_index for o_index in O_indice if np.abs(c_position[2]-atoms[o_index].position[2])<cut_off] # z axis로 가까운 애들만 조사. 
        for o_index in O_in_interest:
            o_position = atoms[o_index].position
            dist_origin=np.linalg.norm(c_position - o_position)
            x_pbc =o_position + atoms.cell[0]
            y_pbc = o_position + atoms.cell[1]
            distances = [dist_origin,
                         np.linalg.norm(c_position - o_position+x_pbc),
                         np.linalg.norm(c_position - o_position-x_pbc),
                         np.linalg.norm(c_position - o_position+y_pbc),
                         np.linalg.norm(c_position - o_position-y_pbc)
                        ]
            min_distance = min(distances)
            if min_distance < cut_off:
                c_o_bonds.append(o_index)
        c_dictionary[c_index] = c_o_bonds
    
    return c_dictionary

def position_identifier(co2_origin_index,initial_atoms):
    cube,rod,interface=0,0,0
    for i,idx in enumerate(co2_origin_index):
        position=initial_atoms[idx].position
        if position[2]>14 and 10<position[1]<48 and 10<position[0]<48 :
            where= 'cube'
            cube=cube+1
        elif 6< position[2] <14 and 10<position[1]<48 and 10<position[0]<48 :
            where='interface'
            interface=interface+1
        else:
            where='rod'
            rod=rod+1
        print(f"{i+1} {position} :::::: {where} ")
    return cube,rod,interface
    
def co2_analyzer(ini_atoms,fin_atoms):
    co2_dict=co2_detector(fin_atoms)
    co2_origin_index=[]
    co2_count=0
    for c_index,values in co2_dict.items():
        if 3>len(values)>1:
            co2_origin_index.append(values[0])
            co2_count=co2_count+1
        elif len(values)==3: # carbonate는 둘중에 많이 움직인애 기준으로 initial O source를 얘기함.
            diff_1=np.linalg.norm(ini_atoms[values[0]].position-fin_atoms[values[0]].position)
            diff_2=np.linalg.norm(ini_atoms[values[1]].position-fin_atoms[values[1]].position)
            #print(diff_1,diff_2)
            if diff_1>diff_2:
                co2_origin_index.append(values[0])
            else: 
                co2_origin_index.append(values[1])
            co2_count=co2_count+1
    cube,rod,interface = position_identifier(co2_origin_index,ini_atoms)
    return cube,rod,interface , co2_count
def traj_to_atoms(traj_filename,num):
    traj = Trajectory(traj_filename)
    fin_atoms=periodic_converter_atoms_xyz(traj[num])
    return fin_atoms

trajectory=Trajectory('../resources/mace_0.traj')
print(len(trajectory))
ini_atoms=trajectory[0]
video_save=[]
for time in list(range(0, 1001, 25)): 
    value_list=[]
    fin_atoms=periodic_converter_atoms(trajectory[time])
    video_save.append(fin_atoms)
    #cube,rod,interface,co2_count = co2_analyzer(ini_atoms,fin_atoms)
    #print(f' {time}_total::::::Cube:Rod:Interface=',co2_count,cube,rod,interface)
    #print(f' 검산',co2_count,cube+rod+interface)

write('./mace_vid.xyz',video_save)
write('last_mace.xyz',fin_atoms)

video_save=[]
trajectory=Trajectory('../resources/co_act_cube_500c_0_800_400ps_more.traj')
for time in list(range(0, len(trajectory), 25)): 
    fin_atoms=periodic_converter_atoms(trajectory[time])
    video_save.append(fin_atoms)
write('./cube_vid.xyz',video_save)
write('last_cube.xyz',fin_atoms)
video_save=[]
trajectory=Trajectory('../resources/co_act_rod_500c_4_800ps_more.traj')
for time in list(range(0, len(trajectory), 25)): 
    fin_atoms=periodic_converter_atoms(trajectory[time])
    video_save.append(fin_atoms)
write('./rod_vid.xyz',video_save)
write('last_rod.xyz',fin_atoms)

rod=read('rod.pdb')
write('rod.xyz',rod)
#write('./mace_final.xyz',fin_atoms)
#write('./mace_ini.xyz',periodic_converter_atoms(trajectory[0]))
#write('./mace_10.xyz',periodic_converter_atoms(trajectory[10]))
#write('./mace_30.xyz',periodic_converter_atoms(trajectory[30]))
#write('./mace_200.xyz',periodic_converter_atoms(trajectory[200]))

