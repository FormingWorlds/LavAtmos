# Standard python packages
import os
import csv 
import numpy as np 
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

# Local packages and paths
sys.path.append(os.getcwd())
wkdir = '.'
os.chdir(wkdir)
#sys.path.insert(1,'wkdir')
sys.path.append(wkdir)
from paths import paths_importer
paths = paths_importer()
melt_comp_path = paths.lava_comps

import lavatmos
import lavatmos2
import lavatmos3

wkdir=wkdir+'ThermoEngine/LavAtmos'
sys.path.append(wkdir)
# Import input
T_surf = float(sys.argv[1])
P_volatile = float(sys.argv[2])
melt_comp_name = sys.argv[3]
output_dir = sys.argv[4]
lavatmos_version = sys.argv[5]
run_name = sys.argv[6]
melt_fraction = sys.argv[7]
elementfile = sys.argv[8]

# Import melt composition
melt_comp_fname = melt_comp_path+melt_comp_name+'.csv'
print(f'Magma composition read from: {melt_comp_fname}')
melt_comp_df = pd.read_csv(melt_comp_fname,names=['spec','abund'])
melt_comp = {}
for i in melt_comp_df.index:
    melt_comp[melt_comp_df['spec'].loc[i]] = melt_comp_df['abund'].loc[i]

# Import volatile composition
if lavatmos_version == 'lavatmos2' or lavatmos_version == 'lavatmos3':
    volatile_comp_fname = paths.input_dir+'volatile_comp.csv'
    print(f'Volatile composition read from: {volatile_comp_fname}')
    volatile_comp = {}
    with open(volatile_comp_fname, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            volatile_comp[row['']] = float(row['mole_fraction'])

# Initiate and run instance of LavAtmos
if lavatmos_version == 'lavatmos1':
    system = lavatmos.melt_vapor_system()
    lavatmos_output = system.vaporise(T_surf, melt_comp, P_melt=P_volatile)

elif lavatmos_version == 'lavatmos2':
    system = lavatmos2.melt_vapor_system(paths)
    lavatmos_output = system.vaporise(T_surf, P_volatile, melt_comp, volatile_comp)

elif lavatmos_version == 'lavatmos3':
    system = lavatmos3.melt_vapor_system(paths)
    lavatmos_output = system.vaporise(T_surf, P_volatile, melt_comp, volatile_comp ,elementfile, melt_fraction)

# Save results
output_name = f'{run_name}.csv'
lavatmos_output.to_csv(output_dir+output_name)