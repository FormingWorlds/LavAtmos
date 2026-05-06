import pandas as pd
from scipy.interpolate import interp1d
import os

def barin_data_importer():
    '''
    Imports Barin (1997) data for specified species.
    
    Returns
    -------
    logK : dict
        LogK data for species.
    
    '''

    species = ['K2O(l)']
    logK = {}

    # path to this folder
    dir = os.path.dirname(__file__)

    for spec in species:
        data = pd.read_csv(f'{dir}/barin/data/barin_data_{spec}.csv')
        logK[spec] = interp1d(data['T'],data[spec])

    return logK