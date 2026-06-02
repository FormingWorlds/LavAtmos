import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import subprocess
import shutil
import pickle
import glob
import argparse

sys.path.append(os.getcwd())
wkdir = '/data3/leoni/PROTEUS/LavAtmos/'
os.chdir(wkdir)

import lavatmos3
# species db class comes from HELIOS code Kitzmann+2017

# Custom modules
class paths_importer:

    def __init__(self):

        '''

        Change the paths as needed. If you don't change the dir structure,
        it should be enough to only change the wkdir.

        '''

        # General directory structure
        self.wkdir = '/data3/leoni/PROTEUS/LavAtmos/'
        self.output_dir = self.wkdir+'output/'
        self.input_dir = self.wkdir+'input/'

        # Inputs
        self.lava_comps = self.input_dir+'lava_compositions/'

        # FastChem 3
        self.fastchem3_dir = os.environ.get("FC_DIR")
        #self.fastchem3_dir = self.wkdir+'FastChem/fastchem3/'
        self.fastchem3_input = self.wkdir+'input/fastchem3/'
        self.fastchem3_config_template = self.fastchem3_input+'config_template.input'
        self.element_abundances3 = self.fastchem3_input+'element_abundances/'


class set_magmaproperties:

    def __init__(self, config: Config, hf_row: dict, volatile_comp):

        '''

        reading in properties from the output file

        '''

        # General directory structure
        paths = paths_importer()
        # Import input
        if hf_row['T_magma'] > 1500:
            self.T_surf = hf_row['T_magma']
        else:
            self.T_surf = 1500
        self.P_volatile = hf_row['P_surf']
        self.melt_comp_name = 'BSE_palm'
        self.output_dir = paths.output_dir
        self.lavatmos_version = 'lavatmos3'
        self.run_name = 'proteus_run'
        self.melt_fraction = 1.0
        self.elementfile = 'element_output_test.dat'
        self.volatile_comp = volatile_comp
        # Saving volatile comp to csv for so that LavAtmos can read it later
        #need to find better way to read in volatile composition that from a parameter dictionary maybe ?



mp = 1.6726231e-27  # kg
kB = 1.38064e-23  # JK-1
particles_per_mol = 6.02214076e23


abundances={'C' : 2.6905e-04,
                        'H' : 9.9965e-01,
                        'N' : 6.7585e-05,
                        'S' : 1.3178e-05,
                        'P' : 2.5695e-20}

def get_input(grid,modelname):
    print('getting grid:', grid)
    compositions=pd.read_csv('/data3/leoni/condensates/'+grid,sep=',')
    compvals=compositions.loc[compositions['comp'] == modelname].to_dict('records')
    compdict=compvals[0]
    for i in abundances:
	    if i in compdict:
		    abundances[i]=float(compdict[i])
    return abundances



parameters = {

    # General parameters
    'run_name' : 'run_proteus_test',

    # Melt parameters
    'lava_comp' : 'BSE_palm',
    'silicate_abundances' : 'lavatmos3', # 'lavatmos1', 'lavatmos2', 'manual'

    # Volatile parameters
    'P_volatile' : 10, # bar
    'oxygen_abundance' : 'degassed', # 'degassed', 'manual'
    'volatile_comp' :  abundances, # I used renormalised solar composition here
    'grid': 'grid_lavatmos_comp.csv'}



#LavAtmos_params={'lava_comp':'BSE_palm','P_volatile':10,'grid':'evolution_output.csv','model':'model'}

class model:
    def __init__(self, abundances, temperature, pvol):
        self.abundances = abundances  
        self.temperature = temperature
        self.pvol = pvol


if __name__ == "__main__":

    modelnames=['testrun']
    grid='grid_lavatmos_comp.csv'
    output_dir = '/data3/leoni/PROTEUS/LavAtmos/output/'

    for i, modelname in enumerate(modelnames):

        abundances=get_input(grid,modelname)
        T=2000
        parameters.update({'volatile_comp':abundances})

        #lavatmos_instance = container_lavatmos(parameters)
        #lavatmos_instance.run_lavatmos(T) # Tboa to update with HELIOS runs

        system = lavatmos3.melt_vapor_system()
        lavatmos_bse = system.vaporise(T,parameters['P_volatile'], parameters['lava_comp'],abundances)
        name = 'Lavatmos_output_{}.csv'.format(modelname)
        print(f'Saving results to: {output_dir+name}')
        lavatmos_bse.to_csv(output_dir+name)