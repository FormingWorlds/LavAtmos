#this module of lavamos 3 has added elementfile as input option such that one can pick which elementfile to pick as output abundances
#this is useful for future runs and needs to be implemented in the main code

from __future__ import annotations
import logging

log = logging.getLogger('fwl.' + __name__)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar
import subprocess
import os

# Thermoengine modules
import thermoengine
from thermoengine import equilibrate 
from thermoengine import model,core
import warnings


# Local modules
from data.databases.janaf_data_importer_gef import janaf_data_importer
from data.databases.barin_data_importer_gef import barin_data_importer
#from library.cacher import CachedResults

class LavAtmosConvergenceError(RuntimeError):
    """Raised when a root-find / FastChem run does not give a trustworthy result."""

class melt_vapor_system:
    
    '''
    Class in which the identities and properties of the melt vapor system 
    are saved. Also loads the important functions from thermoengine. Refrain 
    from initialising more than necessary due to time it takes to load 
    thermo data.
    '''

    def __init__(self, paths=None):
        
        
        # Constants
        self._R = 8.314
        self.Tr = 298.15

        # Warm-start cache for the inner O-abundance solve (see
        # mass_balance_equation_fastchem / _solve_scalar_root)
        self._last_logO = None

        # Points used as a smart start for fO2
        t_dep_points = {}
        t_dep_points = {}
        # log10 fO2 [bar] solved for a volatile-free BSE melt (BSE_palm) at each T
        # (~IW+2.3 to +3.4, O'Neill & Eggins 2002). The previous guesses
        # [1e-16,1e-11,1e-5,1e-2,1e0] at [2000,...,4000] K were 6-8 dex lower, so the
        # default bracket [guess-4, guess+6] missed the root at <= 3000 K.
        t_dep_points['T'] = [1750,2000,2250,2500,3000,3500,4000]
        t_dep_points['fO2'] = [-5.7,-4.4,-3.5,-2.6,-1.2,-0.1,0.8]
        #t_dep_points['fO2'] = np.log10([1e-18,1e-13,1e-7,1e-4,1e-2])
        self.fO2_interp_func = interp1d(t_dep_points['T'], t_dep_points['fO2'], fill_value='extrapolate')
 
        
        # Importing stoichiometries for reactions
        fname_cdef_values = 'cdef_values_vapor_reactions_2.csv'
        dname_cdef_values = os.path.join(paths.lavatmos_dir, 'data')
        self.cdef = pd.read_csv(os.path.join(dname_cdef_values, fname_cdef_values)).set_index('vapor')

        fname_ml_values = 'mass_law_values.csv'
        dname_ml_values = os.path.join(paths.lavatmos_dir, 'data')
        self.ml_values = pd.read_csv(os.path.join(dname_ml_values, fname_ml_values)).set_index('species')
        
        # Importing thermo data
        self.thermo_data = janaf_data_importer() # janaf data
        self.thermo_data.update(barin_data_importer())

        # FastChem 
        self.fastchem_dir = paths.fastchem3_dir
        self.fastchem_output=paths.fastchem3_output
        self.fastchem_column_names = ['#p(bar)','T(K)','n_<tot>(cm-3)','n_g(cm-3)','m(u)']
        #self.fastchem_column_names = ['Pbar','Tk','n_<tot>','n_g','mu']
        self.elementfile = paths.element_abundance_output
        self.abundances_template = paths.element_abundance_template
        self.species_data_file = paths.species_data_file
        self.species_data_file_cond = paths.species_data_file_cond
        self.config_template = paths.fastchem3_config_template
        self.lavatmos_dir=paths.lavatmos_dir

        #######################################################################

        # Initialise states
        self.melt = MeltState(paths)
        
        # Allowed input oxides
        self.oxides = ['SiO2','MgO','Al2O3','FeO','CaO','Na2O','K2O','TiO2','Fe2O3']

        # Vaporised elements
        self.vaporised_elements = ['Al', 'Ca', 'Fe', 'K', 'Mg', 'Na',\
                                   'O', 'Si', 'Ti']
                            
        # Includes the noble gases (He, Ne, Ar, Kr, Xe) so the full element set
        # supported by PROTEUS/AGNI can be carried through the FastChem
        # speciation. The noble gases are chemically inert: they take no part in
        # the vapour reactions and are passed through as monatomic gas.
        self.all_elements = ['C','H','He','N','O','P','S','Si','Ti',\
                             'V','Cl','K','Na','Mg','F','Ca','Fe','Al',\
                             'Ne','Ar','Kr','Xe']

        # Noble gases carry a zero mass-law contribution: being inert, they do
        # not enter the fO2 mass-balance residual, only dilute the atmosphere.
        self.mass_law_contribution = {'C' : 0,
                                      'H' : 0,
                                      'He': 0,
                                      'N' : 0,
                                      'O' : 0.5,
                                      'P' : 0,
                                      'S' : 0,
                                      'Si':-1,
                                      'Ti':-1,
                                      'V' : 0,
                                      'Cl': 0,
                                      'K' :-0.25,
                                      'Na':-0.25,
                                      'Mg':-0.5,
                                      'F' : 0,
                                      'Ca':-0.5,
                                      'Fe':-0.75,
                                      'Al':-0.75,
                                      'Ne': 0,
                                      'Ar': 0,
                                      'Kr': 0,
                                      'Xe': 0}
                                        


        # Solar elemental abundance
        self.solar_abundance = {
                        'e-' : 0.00,
                        'Al' : 6.45,
                        'Ar' : 0.00,
                        'C' : 8.43,
                        'Ca' : 6.34,
                        'Cl' : 5.50,
                        'Co' : 4.99,
                        'Cr' : 5.64,
                        'Cu' : 4.19,
                        'F' : 4.56,
                        'Fe' : 7.50,
                        'Ge' : 3.65,
                        'H' : 12.0,        
                        'He' : 10.93,
                        'K' : 5.03,
                        'Mg' : 7.60,
                        'Mn' : 5.43,
                        'N' : 7.83,
                        'Na' : 6.24,
                        'Ne' : 7.93,
                        'Ni' : 6.22,
                        'O' : 8.69,
                        'P' : 5.41,
                        'S' : 7.12,
                        'Si' : 7.51,
                        'Ti' : 4.95,
                        'V' : 3.93,
                        'Zn' : 4.56
                        }

    def melt_O_budget_bar(self, vapor_pp):

        '''O_stoich and O_vap in bar (as atoms), from the cdef vapour species.
        O_stoich uses each reaction's own d_i (2 d_i = n_O,i - O_stoich,i),
        so Fe2O3- and Fe2SiO4-based rows are both handled correctly.'''

        A = {el: 0.0 for el in self.vaporised_elements}
        p = np.zeros(len(self.cdef.index))
        for i, spec in enumerate(self.cdef.index):
            p[i] = float(vapor_pp[i].iloc[0])
            for el, n in self.parse_formula(spec).items():
                if el in A:
                    A[el] += n * p[i]
        O_vap = A['O']
        O_stoich = O_vap - 2.0 * np.sum(self.cdef['di'].to_numpy(dtype=float) * p)
        return O_stoich, O_vap
    
    def parse_formula(self,formula):
        import re
        tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
     
        composition = {}
        for element, count in tokens:
            composition[element] = (
                composition.get(element, 0) + int(count or 1)
            )
     
        return composition
        
    def pure_melt_logfO2(self, t, xlo=-40.0, xhi=12.0):
        '''log10 fO2 of the melt with no envelope: -sum_i d_i p_i(fO2) = fO2.'''
        d = self.cdef['di'].to_numpy(dtype=float)
        def h(x):
            vp = self.vapor_partial_pressure_calc(10.0**x, [t])
            p = vp.iloc[0].to_numpy(dtype=float)
            return -np.sum(d * p) - 10.0**x
        return brentq(h, xlo, xhi, xtol=1e-6)  
    
     
    def _solve_fO2_bracketed(self, residual, x0, step=1.0, grow=1.5,
                             xmin=-45.0, xmax=12.0, max_steps=40,
                             xtol=1e-6, allow_unconverged=False,
                             increasing=None, label='fO2'):
        '''
        Steps outward from x0 on both sides (step growing by `grow` each
        round) until the sign of f changes between two consecutive finite
        points on one side, then refines that bracket with brentq.
 
        f returns np.nan where it cannot be evaluated (e.g. FastChem fails);
        those points are stepped over. Each side stops on its own when it
        leaves [xmin, xmax]. f is cached, so no point is evaluated twice.
 
        Returns (root, info). Raises LavAtmosConvergenceError if no sign
        change is found.
        '''
        cache = {}
        errors = {}
        def fc(x):
            if x not in cache:
                try:
                    cache[x] = float(residual(x))
                except (ValueError, LavAtmosConvergenceError) as e:
                    cache[x] = np.nan
                    errors[x] = f'{type(e).__name__}: {e}'
            return cache[x]
 
        def refine(a, b):
            """Root in [a, b] (f(a), f(b) finite, opposite signs). brentq first;
            if it runs into NaN points inside the bracket, NaN-tolerant bisection."""
            def strict(x):
                v = fc(x)
                if not np.isfinite(v):
                    raise LavAtmosConvergenceError('NaN inside bracket')
                return v
            try:
                return brentq(strict, a, b, xtol=xtol, maxiter=100)
            except LavAtmosConvergenceError:
                pass
            fa = fc(a)
            while b - a > xtol:
                m = None
                for frac in (0.5, 0.25, 0.75, 0.125, 0.875, 0.0625, 0.9375):
                    x = a + frac * (b - a)
                    if np.isfinite(fc(x)):
                        m = x
                        break
                if m is None:
                    raise LavAtmosConvergenceError(
                        f'{label}: residual NaN throughout ({a:.3f}, {b:.3f}) inside the bracket')
                fm = fc(m)
                if fm == 0:
                    return m
                if fa * fm < 0:
                    b = m
                else:
                    a, fa = m, fm
            return 0.5 * (a + b)
 
        def check_new_point(x):
            """Sign change between x and its nearest finite neighbour on the x0 side
            (or on either side). Works across x0, also when f(x0) is NaN."""
            fx = cache[x]
            fin = sorted(k for k, v in cache.items() if np.isfinite(v))
            i = fin.index(x)
            for j in (i - 1, i + 1):
                if 0 <= j < len(fin) and cache[fin[j]] * fx < 0:
                    return tuple(sorted((x, fin[j])))
            return None
 
        f0 = fc(x0)
        if f0 == 0:
            return x0, dict(converged=True, residual=0.0, bracket=None, n_eval=len(cache))
 
        pos  = {-1: x0, +1: x0}
        done = {-1: False, +1: False}
 
        # If f is known to be monotonic, the sign at x0 tells on which side the
        # root lies: only search that side.
        if increasing is not None and np.isfinite(f0):
            root_is_lower = (f0 > 0) if increasing else (f0 < 0)
            done[+1 if root_is_lower else -1] = True
        dx = step
 
        while not (done[-1] and done[+1]):
            for s in (-1, +1):
                if done[s]:
                    continue
                prev = pos[s]
                pos[s] += s * dx
                if not (xmin <= pos[s] <= xmax):
                    # overshot: evaluate the bound itself once, so a root between
                    # the last point and the bound is not skipped
                    pos[s] = xmin if s < 0 else xmax
                    done[s] = True
                    if prev == pos[s]:
                        continue
                fx = fc(pos[s])
                if not np.isfinite(fx):
                    continue
                br = check_new_point(pos[s])
                if br is not None:
                    a, b = br
                    root = refine(a, b)
                    return root, dict(converged=True, residual=fc(root),
                                      bracket=(a, b), n_eval=len(cache))
            dx *= grow
 
        finite = {x: v for x, v in cache.items() if np.isfinite(v)}
        if allow_unconverged and finite:
            xb = min(finite, key=lambda x: abs(finite[x]))
            warnings.warn(f'{label}: no sign change in [{xmin}, {xmax}] -- returning point '
                          f'with smallest |residual| ({xb:.3f}: {finite[xb]:.3g}).')
            return xb, dict(converged=False, residual=finite[xb], bracket=None, n_eval=len(cache))
        raise LavAtmosConvergenceError(
            f'{label}: no sign change in [{xmin}, {xmax}] from x0={x0:.2f}; '
            f'{len(cache)} evaluations, {len(finite)} finite. '
            f'Closest: {min(finite.items(), key=lambda kv: abs(kv[1])) if finite else None}.\n'
            f'  Evaluated points (x: residual): '
            + ', '.join(f'{x:.2f}: {v:.3g}' for x, v in sorted(cache.items()))
            + (f'\n  First failure: {next(iter(errors.items()))}' if errors else ''))       

    def read_phi_mix(self):
        '''Atoms per gas molecule (n_<tot>/n_g) of the last FastChem run.'''
        raw = pd.read_csv(self.fastchem_output_path(), sep=r'\s+')
        return float(raw['n_<tot>'].iloc[0] / raw['n_g'].iloc[0])
 
    def solve_P_boa(self, O_abun, T, vapor_partial_pressures, volatile_comp, fO2,
                    rtol=1e-5, maxiter=30):
        '''
        Self-consistent surface pressure from the atom inventory:
 
            P_boa = A_tot / phi_env(P_boa)
 
        A_tot = phi_env*P_volatile + sum_i n_i p_i + (O_stoich - O_vap)   [atoms, in bar]
        phi_env = atoms per molecule of the mixed gas, from FastChem.
 
        This replaces both closures:
          LavAtmos2: P_boa = sum p_vap + fO2 + P_vol     (released O assumed not to add molecules)
          LavAtmos3: P_boa = sum p_vap - sum d_i p_i + P_vol  (released O assumed to stay as free O2)
        and reduces to each of them in its own limit. Returns (P_boa, partial_pressures).
        '''
        # Start: previous solution if there is one, else the LavAtmos2 mole count
        if self._P_boa_prev is not None:
            P = self._P_boa_prev
        else:
            P = vapor_partial_pressures.sum(axis=1).iloc[0] + fO2 + self.P_volatile
        if not (np.isfinite(P) and P > 0):
            raise LavAtmosConvergenceError(f'Bad starting P_boa: {P}')
 
        for _ in range(maxiter):
            pp = self.calculate_partial_pressures_fastchem_loop(O_abun, T, [P],
                                                                vapor_partial_pressures, volatile_comp)
            self.phi_mix = self.read_phi_mix() #atoms per gas molecule from gastchem nags/ntot
            if not (self.A_tot > 0):
                raise LavAtmosConvergenceError(f'Non-positive atom inventory A_tot = {self.A_tot}')
            P_new = self.A_tot / self.phi_mix
            if abs(P_new - P) <= rtol * P:
                self._P_boa_prev = P
                return P, pp
            P = P_new
 
        raise LavAtmosConvergenceError(f'P_boa did not converge in {maxiter} iterations (last {P:.4e} bar)')

    
    def vaporise(self, T, P_volatile, melt_comp, volatile_comp , melt_fraction=1.0 , P_melt = 0.01,\
                          fO2_initial_guess = 1e-10, fO2_tries_from_last=False, verbose=True, allow_unconverged = False,
                          phi_env = None, logfO2_min = -45.0, logfO2_max = None,
                          one_sided = False):

        self._P_boa_prev = None
        self.P_volatile = P_volatile


        # Ensures that T is iterable even if just one value is given
        if type(T) != np.ndarray:
            T = np.array([T])
        
        self.melt_comp = self.melt.set_melt_comp(melt_comp,verbose) 
        self.melt.calculate_melt_chemical_potentials(T,P_melt,self.thermo_data)
        self.melt.calculate_melt_activities(T,P_melt)            
        self.logKr = self.logKr_calc(T,self.melt.mu0_liquid)

        #find initial O2 partial pressure before outgassing -> intheory we actually know this from calliope, so sould also use that output instead
        #not sure if this approach works - maybe I need to instead find the overall O-inventory
        if fO2_tries_from_last:
            logfO2_min = np.log10(fO2_initial_guess / 1000)
            logfO2_max = np.log10(fO2_initial_guess * 1000)
        else:
            logfO2_min = -45
            logfO2_max = 6

        if phi_env is None:
            # any non-zero value works here: phi_env cancels in the normalisation of an
            # envelope-only run (the old code used 0.0 at the first T -> all-zero abundances)
            self.phi_env = 1.0
            self.calculate_partial_pressures_fastchem_loop(0.0, T, [self.P_volatile],
                                                                           vapor_partial_pressures=None,
                                                                           volatile_comp=volatile_comp)
            self.phi_env = self.read_phi_mix() #atoms per molecule - scales molecular pressure to atomic pressures
        else:
            self.phi_env = float(phi_env)
            self._P_boa_prev = None
         
        def residual(logfO2, t=T):
            '''function can only be evaluated if thepressure is not too high or to low for fastchem'''
            try:
                return self.mass_balance_equation_fastchem(None, 10.0**logfO2, T, volatile_comp)
            except (ValueError, LavAtmosConvergenceError):
                return np.nan
                                                        
         
        
        # start from the previous temperature's root if there is one, else the no-envelope fO2
        x0 = np.log10(fO2_initial_guess) if fO2_tries_from_last else self.pure_melt_logfO2(T[0])
        x0 = min(max(x0, logfO2_min), logfO2_max)       # start inside the allowed range


        logfO2_best, info = self._solve_fO2_bracketed(residual, x0,
                xmin=logfO2_min, xmax=logfO2_max,
                allow_unconverged=allow_unconverged,
                increasing=True if one_sided else None,
                label=f'fO2 at T={T[0]:.0f} K')
        
        self._x0_prev = logfO2_best
        log.debug(f'best fO2 {10**logfO2_best:.3e} | residual {info["residual"]:.2e} '
                              f'| {info.get("n_eval", "?")} FastChem evaluations')  

        #subtract previouly available oxygen from oxygen that will be outgassed.
        fO2_outgassed = 10**logfO2_best  #maybe better from mass balance

        vapor_partial_pressures = self.vapor_partial_pressure_calc(fO2_outgassed, T)
        P_boa, pp = self.solve_P_boa(None, T, vapor_partial_pressures, volatile_comp, fO2_outgassed)
        P_outgassed = vapor_partial_pressures.sum(axis=1).iloc[0]+fO2_outgassed
        log.debug(f'P_boa {P_boa:.4e} bar | phi_mix {self.phi_mix:.4f} | phi_env {self.phi_env:.4f}')
        

        # The vapour above was computed at fO2_best, so FastChem's O2 must match it. It
        # does not when the inner O-abundance solve fails, e.g. when a volatile oxygen
        # budget alone holds pO2 above fO2_best; the vapour is then out of equilibrium
        # with the gas it is mixed into.
        pO2_gas = pp['O2'].iloc[0]
        if abs(np.log10(pO2_gas / fO2_outgassed)) > 0.01:
            log.warning(f'LavAtmos: FastChem pO2 ({pO2_gas:.3e} bar) differs from the solved '
                        f'fO2 ({fO2_outgassed:.3e} bar); melt-vapour equilibrium is not satisfied')

        return pp




    def mass_balance_equation_fastchem(self,fO2,T, volatile_comp,xatol=1e-6):

        '''
        Equation that needs to be solved for fO2.
        
        Parameters
        ----------
        input : float
            fO2 in bar.
        
        Returns
        -------
        eq1 : float
            The result of the mass balance equation.    
        '''
        
        vapor_partial_pressures = self.vapor_partial_pressure_calc(fO2, T)
         
        # P_boa from the atom inventory instead of a mole count guessed before speciation
        _, fastchem_partial_pressures = self.solve_P_boa(O_abun, T, vapor_partial_pressures, volatile_comp, fO2)
        fO2_output = fastchem_partial_pressures['O2'].iloc[0]
        residual = np.log10(fO2) - np.log10(fO2_output)
            
        return residual



    def calculate_partial_pressures_fastchem_loop(self,O_abun,T,P_boa,vapor_partial_pressures, volatile_comp, meltfrac):


        '''melt fraction can only be set <1 at the end of the loop when the balance has been determined. Then fO2 is uppdated one last time showing 
        in the partial pressures of the fastchem output'''
    
        self.calculate_fastchem_abundance(O_abun, vapor_partial_pressures, P_boa, volatile_comp, meltfrac)
        self.edit_fastchem_configs(T,P_boa)
        self.run_fastchem()

        fastchem_partial_pressures = self.read_fastchem_partial_pressures()

        return fastchem_partial_pressures


    def calculate_fastchem_abundance(self, O_abun, vapor_partial_pressures, P_boa,\
                                     volatile_comp, meltfrac=1.0, final_output=False):

        if self.is_iterable(P_boa):
            P_boa = P_boa[0]

        if self.is_iterable(O_abun):
            O_abun = O_abun[0]

        frac = {}

        if vapor_partial_pressures is not None:
        
            for el in self.vaporised_elements:
                frac[el] = 0

                mole_fractions = vapor_partial_pressures/P_boa

            for i,spec in enumerate(self.cdef.index):
                for el in self.vaporised_elements:

                    if (el+'3') in spec:
                        frac[el] += 3*mole_fractions[i].iloc[0]

                    elif (el+'2') in spec:
                        frac[el] += 2*mole_fractions[i].iloc[0]

                    elif (el) in spec:
                        frac[el] += mole_fractions[i].iloc[0]

         # Melt-derived O: either given, or (O_abun=None) set by stoichiometry
        if vapor_partial_pressures is not None and O_abun is None:
            
            O_stoich, O_vap = self.melt_O_budget_bar(vapor_partial_pressures)
            O_abun = (O_stoich - O_vap) / P_boa
            self.O_abun = O_abun
 
        frac['O'] = frac.get('O', 0.0) + (O_abun or 0.0)

        #budget for O comes from interior and atmosphere not only atmosphere so need to consider all sources for fastchem
        for vol in volatile_comp.keys(): #these are CHNOPS
            if P_boa==0: #[P_outgassed + self.P_volatile] This can happen before outgassing and if O in input element abundances but aboundace of O=0.0
                print('surface pressure of volatiles is zero no volatile envelope present')
                frac[vol] = 0.0
            else:
                contribution = self.P_volatile * self.phi_env * volatile_comp[vol] / P_boa
 
                # Preserve any contribution already accumulated from vapor species.
                frac[vol] = frac.get(vol, 0.0) + contribution
 
        '''
        Normalize elemental abundances
        '''
        fracN = {}
        # print(frac.values())
        sum_fracs = sum(frac.values())
        if not np.isfinite(sum_fracs):
            raise LavAtmosConvergenceError(f'Non-finite elemental abundances passed to FastChem: {frac}')
        # Total atom inventory in bar-equivalents. Every frac term is (atoms in bar)/P_boa,
        # so this product does not depend on the P_boa that was passed in.
        self.A_tot = sum_fracs * P_boa
        # print(sum_fracs)
        for el in frac:
            if sum_fracs > 0.0:
                fracN[el] = frac[el]/sum_fracs
            else:
                fracN[el]=0.0
 
        verbose = False
        if verbose:
            print('Normalized fractional abundances:')
            for el in fracN:
                print(el)
                print(fracN[el])
 
        # Open abundance file template
        # TODO: Consider not hardcoding this (moving to paths file)
        template_name = 'element_abundances_template2.dat'
        #template_name = 'element_abundances_noe-.dat'
        output_name = self.elementfile
        solar_abund_name = 'element_abundances_solar.dat'
 
        template = open(self.abundances_location+template_name, 'r')
        elab_file = template.read()
        template.close()
 
        #all refractory element abundances are zero if no outgassing yet
        if vapor_partial_pressures is None:
            for el in self.vaporised_elements:
                if el != 'O':
                    fracN[el] = 0.0
 
    
        # Set abundances to outgassed abundances
        for el in self.all_elements:
            if el in self.vaporised_elements and fracN[el] != 0 or el in volatile_comp and fracN[el] != 0:
                abund = 12 + np.log10(fracN[el]*1e20)
 
                if self.is_iterable(abund):
                    abund = abund[0]
                
                elab_file = elab_file.replace(f'<<{el}>>', str(abund))
            else:
                abund = 0
                elab_file = elab_file.replace(f'<<{el}>>', str(abund))
        
        # Save abundance file
        self.abundance_fname = self.abundances_location + output_name
        
        g = open(self.abundance_fname, 'w')
        g.write(elab_file)
        g.close()
 
        return P_boa

    def vapor_partial_pressure_calc(self, fO2, T):

        partial_pressures_vapor = pd.DataFrame()
        a = self.melt.a.loc[T]
        logKr = self.logKr.loc[T]

        # O does not require oxide, hence separate
        partial_pressures_vapor[0] = 10**logKr[0] * fO2**self.cdef['di'].iloc[0]

        for i in range(1,len(self.cdef.index)):
            
            # Importing names and values from cdef file
            endmember = self.cdef.iloc[i]['endmember']
            if pd.notna(self.cdef.iloc[i]['liq_oxide1']) :
                liq_oxide1 = self.cdef.iloc[i]['liq_oxide1'].replace('(l)','')
            else:
                liq_oxide1 = 'None'
            if pd.notna(self.cdef.iloc[i]['liq_oxide2']) :
                liq_oxide2 = self.cdef.iloc[i]['liq_oxide2'].replace('(l)','')
            else:
                liq_oxide2 = 'None'
            ci = self.cdef.iloc[i]['ci']
            di = self.cdef.iloc[i]['di']
            ei = self.cdef.iloc[i]['ei'] 
            fi = self.cdef.iloc[i]['fi']
            
            # Retrieving activities            
            if endmember in a:
                a_endmember = a[endmember]
            else: 
                a_endmember = 0
            
            if liq_oxide1 == 'None':
                a_liq_oxide1 = 1
            else:
                a_liq_oxide1 = a[liq_oxide1]

            if liq_oxide2 == 'None':
                a_liq_oxide2 = 1
            else:
                a_liq_oxide2 = a[liq_oxide2]
            # Calculating partial pressure
            partial_pressures_vapor[i] = 10**logKr[i]\
                                  * a_endmember**ci\
                                  * fO2**di\
                                  * a_liq_oxide1**ei\
                                  * a_liq_oxide2**fi


        return partial_pressures_vapor

    def logKr_calc(self, T, mu0_liquid):
        '''
        Calculates the logK values for all of the vapor reactions at a 
        given temperature.
        
        Assumes:

        logKr = logKf_products - v*logKf_reactants
        
        And since logKf of O2 is zero for all relevant T:

        logKr_vapor = logKf_vapor - ci*logKf_melt_oxide

        Parameters
        ----------
        T : array, float, or int
            Equilibrium temperature in kelvin.

        mu0_liquid : DataFrame
            Pandas dataframe containing Gibbs standard free energy values
            for all endmembers in the melt.
            
        Returns
        -------
        logK : dict
            LogK values for all vapor reactions.

        TODO: Rewrite in terms of matrix multiplications.
              If needed, add option to use MELTS chempot values.
        '''
        
        logKr = {}
        
        # Defines temperatures above which thermo values need to be 
        # extrapolated from JANAF values 
        T_interp = T[T<=6000]
        T_extrap = T[T>6000]

        # Interpolating for values within JANAF range
        gibbs_O2 = - T_interp * self.thermo_data['O2(g)'].gef(T_interp)\
                   + self.thermo_data['O2(g)'].DeltaH(self.Tr)

        # Extrapolating from last few values for T outside JANAF range 
        # (not advised)
        if T_extrap.size > 0:
            T_temp = self.thermo_data['O2(g)'].T 
            gibbs_temp = - T_temp * self.thermo_data['O2(g)'].gef(T_temp)\
                         + self.thermo_data['O2(g)'].DeltaH(self.Tr)
            gibbs_O2 = np.append(gibbs_O2,np.poly1d(np.polyfit(T_temp[-4:],\
                                                 gibbs_temp[-4:],1))(T_extrap))
        
        for i in range(len(self.cdef.index)):
            
            #Calculate gibbs energy of a vapor for a given T
            vapor = self.cdef.index[i]
            # Importing names and values from cdef file
            endmember = self.cdef.iloc[i]['endmember']
            if pd.notna(self.cdef.iloc[i]['liq_oxide1']) :
                liq_oxide1 = self.cdef.iloc[i]['liq_oxide1'].replace('(l)','')
            else:
                liq_oxide1 = 'None'
            if pd.notna(self.cdef.iloc[i]['liq_oxide2']) :
                liq_oxide2 = self.cdef.iloc[i]['liq_oxide2'].replace('(l)','')
            else:
                liq_oxide2 = 'None'
            ci = self.cdef.iloc[i]['ci']
            di = self.cdef.iloc[i]['di']
            ei = self.cdef.iloc[i]['ei'] 
            fi = self.cdef.iloc[i]['fi']

            # Interpolating for values within JANAF range
            gibbs_vapor = - T_interp * self.thermo_data[vapor].gef(T_interp)\
                          + self.thermo_data[vapor].DeltaH(self.Tr)
            
            # Extrapolating from last few values for T outside JANAF range 
            # (not advised)
            if T_extrap.size > 0:
                T_temp = self.thermo_data[vapor].T 
                gibbs_temp = - T_temp * self.thermo_data[vapor].gef(T_temp)\
                             + self.thermo_data[vapor].DeltaH(self.Tr)
                gibbs_vapor = np.append(gibbs_vapor,\
                                        np.poly1d(np.polyfit(T_temp[-4:],
                                                  gibbs_temp[-4:],1))\
                                        (T_extrap))

            # Calculate K value of a vapor for a given T
            if vapor != 'O(g)':
                # Retrieving gibbs energies
                gibbs_endmember = np.array(mu0_liquid[endmember])

                if liq_oxide1 == 'None':
                    gibbs_liq_oxide1 = 0
                else:
                    gibbs_liq_oxide1 = np.array(mu0_liquid[liq_oxide1])

                if liq_oxide2 == 'None':
                    gibbs_liq_oxide2 = 0
                else:
                    gibbs_liq_oxide2 = np.array(mu0_liquid[liq_oxide2])
                
                logKr[i] = (-np.log10(np.e))/(self._R*T)\
                            * (gibbs_vapor\
                               - ci*gibbs_endmember\
                               - di*gibbs_O2\
                               - ei*gibbs_liq_oxide1\
                               - fi*gibbs_liq_oxide2)
            else: 
                logKr[i] = (-np.log10(np.e))/(self._R*T) * (gibbs_vapor\
                                                            - di * gibbs_O2)
        
        # Convert to dataframe
        logKr = pd.DataFrame(logKr,index = T)
        
        return logKr

    def edit_fastchem_configs(self,T,P):
        '''
        Editing TP point
        '''


        template_path_tp = os.path.join(self.lavatmos_dir, 'input/fastchem3/tp_point_template.dat')

        # Open parameter template fileself.fastchem_dir + 'input/logK/logK.dat'
        template_tp = open(template_path_tp)
        tp_config = template_tp.read()
        template_tp.close()


        tp_config = tp_config.replace(f'<<P>>', f'{P[0]:.6e}')
        tp_config = tp_config.replace(f'<<T>>', f'{T[0]:.6e}')
    
        tp_point_path = self.fastchem_output +'tp_point.dat' 
        tp_config_file = open(tp_point_path, 'w')
        tp_config_file.write(tp_config)
        tp_config_file.close()


        '''
        Editing fastchem input config file
        '''
       
        config_path = self.fastchem_output+'config.input'
        param_path = self.fastchem_output+'parameters.dat'

        # Config file
        template_path_config = self.config_template

        # Open parameter template fileself.fastchem_dir + 'input/logK/logK.dat'
        template = open(template_path_config)
        configurations = template.read()
        template.close()

        fastchem_config = {
            'param_path' : param_path,
            'tp_grid_path' : tp_point_path, 
            'output_abundance_fname' : self.fastchem_output +'boa_chem.dat',
            'output_condensate_fname' : self.fastchem_output +'condensates.dat',
            'output_monitor_fname' : self.fastchem_output +'monitor.dat',
            'element_abundance_file' : self.abundance_fname.replace('FastChem/',''),
            'species_data_file' : self.species_data_file ,
            'species_data_file_cond' : self.species_data_file_cond
            #'species_data_file' : '/data3/leoni/LavAtmos/FastChem/fastchem3/input/logK/logK.dat',
            #'species_data_file_cond' : '/data3/leoni/LavAtmos/FastChem/fastchem3/input/logK/logK_condensates.dat',
            }
        
        for config in fastchem_config:
            configurations = configurations.replace(f'<<{config}>>', fastchem_config[config])


        config_file = open(config_path, 'w')
        config_file.write(configurations)
        config_file.close()

        '''
        Editing param file
        '''
        # Parameter file

        '''
        template_path_param = self.fastchem_data+'parameters_template.dat'

        # Open parameter template file
        template = open(template_path_param)
        parameters = template.read()
        template.close()

        fastchem_param = {\
            'element_abundance_file' : self.abundance_fname,
            'species_data_file' : self.fastchem_data+'logK/mixed_ion.dat'
        }

        for param in fastchem_param:
            parameters = parameters.replace(f'<<{param}>>',fastchem_param[param])

        # Save abundance file
        param_file = open(param_path, 'w')
        param_file.write(parameters)
        param_file.close()
        '''


    def run_fastchem(self):
        # Check call instead of call can catch the error if FastChem cannot run properly, e.g. if it is not compiled.

        try: 
            subprocess.check_call([f'./fastchem {self.fastchem_output}config.input'],\
                                shell=True,\
                                cwd=f'{self.fastchem_dir}',stdout=subprocess.DEVNULL)
        except: 
            log.error(f'Exited during FastChem call.')
            log.error(f'If this was not intended, try re-compiling FastChem at: {self.fastchem_dir}')

    def read_fastchem_partial_pressures(self):

        fname = os.path.join(self.fastchem_output, 'boa_chem.dat')
        fastchem_partial_pressures = pd.read_csv(fname, sep=r'\s+')
        return fastchem_partial_pressures.drop(columns=self.fastchem_column_names)\
               *fastchem_partial_pressures[self.fastchem_column_names[0]].iloc[0]

    def is_iterable(self,variable):
        try:
            iter(variable)
            return True
        except TypeError:
            return False


class MeltState:
    '''
    Class for holding the vapor state for the thermodynamic parameters 
    determined by melts. 
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    '''

    def __init__(self,paths=None,melts_version='1.0.2'): 

        # Constants
        self._R = 8.314 # Gas constant
        self.Tr = 298.15

        # Initialising parameters to be calculated
        self.a = None
        
        # Initialising thermoengine classes
        #self.melts = equilibrate.MELTSmodel(melts_version)
        self.elm_sys = ['H','O','Na','Mg','Al','Si','P','K','Ca','Ti','Cr','Mn','Fe','Co','Ni']
        thermoengine.model.Database.DATABASE_DATA_DIR=os.path.dirname(thermoengine.__file__)+"/../../phase_library/databases/"
        self.modelDB = model.Database(database_name="MELTS_v1_0")
        self.liq_phs = self.modelDB.get_phase('Liq')
        self.phs_sys= self.modelDB.get_phases(['Liq'])

        self.equil = thermoengine.equilibrate.Equilibrate(self.elm_sys,self.phs_sys)
        #melts = equil.execute(T, P, bulk_comp=blk_cmp, debug=debug, stats=stats)

        self.endmember_names = self.liq_phs.endmember_names
        self.nonendmember_oxide_names = ['MgO','FeO','CaO','Na2O',\
                                         'K2O','Cr2O3']

        # Importing data for liquid reactions
        self.lavatmos_dir = paths.lavatmos_dir
        fname_cdef_liq_values = 'cdef_values_liquid_reactions.csv'
        if paths.lavatmos_dir is None:
            raise ValueError('LavAtmos directory not set.')
        else:
            dname_cdef_liq_values =  os.path.join(self.lavatmos_dir, 'data')
        self.cdef_liq = pd.read_csv(os.path.join(dname_cdef_liq_values, fname_cdef_liq_values))\
                                .set_index('liq_oxide')
        
        # Oxides that are included in calculations
        self.used_oxides = ['SiO2','MgO','Al2O3','FeO','CaO','Na2O','K2O',\
                            'TiO2','Fe2O3']

        
    def set_melt_comp(self,input_comp,verbose):
        '''
        Checks if input composition is valid before setting melt 
        composition in thermoengine class (MELTS). 
        
        Parameters
        ----------
        input_comp : dict
            Input composition.

        verbose : bool
            True/False for printing text or not. 

        Returns
        -------
        melt_comp : dict
            Melt composition as accepted by MELTS.
                
        '''

        # Set dictionary labels to oxides allowed by melts
        melt_comp = {} # dict([(ox,0) for ox in self.liq_phs.OXIDES])
        for species in input_comp:

            if species not in self.liq_phs.OXIDES: # Check validity of input
                log.error(f'{species} not allowed. Composition not set.')
                return  
            else: 
                melt_comp[species] = input_comp[species]
        
            if species not in self.used_oxides:
                log.warning(f'input value for {species} passed. Species not'
                      + f' (yet) included in vaporisation calculations.')
            
        # Save included oxides
        self.included_oxides = {}
        for ox in melt_comp:
            if melt_comp[ox] == 0:
                self.included_oxides[ox] = False
            else:
                self.included_oxides[ox] = True
        self.comp=melt_comp
        return melt_comp
    
    
    def calculate_melt_chemical_potentials(self, T, P_melt, thermo_data):
        '''
        Calculates the chemical potentials of the oxide species in the melt. 
        For the oxides that are not included in MELTS as endmemebers, data 
        from the thermo databases is used. 
        
        Parameters
        ----------
        T : float
            Equilibrium temperature in kelvin
            
        P_melt : float
            Pressure at which the melt equilibrium chemistry calculations are
            done.
            
        thermo_data : dict
            Dictionary containing Gibbs energy function values for each species 
            relevant to the calculations.
                
        '''
        # Calculates endmember chemical potentials using MELTS function
        mu0_endmember = np.array([self.liq_phs.gibbs_energy(T,P_melt,mol=imol)\
                                  for imol in np.eye(15)])
        self.mu0_liquid = pd.DataFrame(mu0_endmember,index=self.liq_phs.endmember_names,columns=T).T

        for oxide in self.nonendmember_oxide_names:
            # Try to interpolate, if outside range of database uses linear fit 
            # of last four data points to extrapolate to higher temperatures. 
            phase=thermo_data[oxide+'(l)']
            max_temp = thermo_data[oxide+'(l)'].T.iloc[-1]
            T_interp = T[T<=max_temp]
            T_extrap = T[T>max_temp]

            gibbs = - T_interp * thermo_data[oxide+'(l)'].gef(T_interp) \
                    + thermo_data[oxide+'(l)'].DeltaH(self.Tr)
            if T_extrap.size > 0: 
                T_temp = thermo_data[oxide+'(l)'].T 
                gibbs_temp = - T_temp * thermo_data[oxide+'(l)'].gef(T_temp)\
                             + thermo_data[oxide+'(l)'].DeltaH(self.Tr)
                gibbs = np.append(gibbs,np.poly1d(np.polyfit(T_temp[-4:],
                                        gibbs_temp[-4:],1))(T_extrap))

            self.mu0_liquid[oxide] = gibbs
            
    
    def calculate_melt_activities(self, T, P_melt):
        '''
        Retrieves the activity of the endmembers using a Thermoengine (MELTS)
        function.
        
        Parameters
        ----------
        T : float
            Equilibrium temperature in kelvin
            
        P_melt : float
            Pressure at which the melt equilibrium chemistry calculations are
            done.                
        '''
        a = {}
        mantle_comp=self.comp
        self.mol_oxides = core.chem.format_mol_oxide_comp(mantle_comp,convert_grams_to_moles=True)
        self.moles_end,self.oxide_res= self.liq_phs.calc_endmember_comp(mol_oxide_comp=self.mol_oxides, method='intrinsic', output_residual=True)
        if not self.liq_phs.test_endmember_comp(self.moles_end):
            log.warning("Calculated composition is infeasible!")
        self.mol_elm = self.liq_phs.convert_endmember_comp(self.moles_end,output='moles_elements')
        blk_cmp = []
        for elm in self.elm_sys: 
            index = core.chem.PERIODIC_ORDER.tolist().index(elm)
            blk_cmp.append(self.mol_elm[index]) 
        self.blk_cmp = np.array(blk_cmp)
        R = 8.31446261815324
        # Calculates endmember activities using MELTS function
        for t in T:
            temp=float(t)
            # Calculate excess chemical potential

            state=self.equil.execute(temp,P_melt,bulk_comp=self.blk_cmp)

            #getting activities:
            #a_i = exp((μ_i - μ_i°) / RT)

            liq = state.phase_d['Liquid']
            phase = state.phase_d['Liquid']['obj']
            mol=np.array(state.phase_d['Liquid']['moles']) #mol shoudl be teh composition of the liquid in terms of mol fraction of teh endmembers
            end_array = np.identity(phase.endmember_num)
            chem_pots = self.liq_phs.chem_potential(temp, P_melt, mol=mol) #loops over imol in mole to get mu of each species corresponding to each endmember 
            chem_pots_endms = np.zeros(phase.endmember_num)
            for i in range(phase.endmember_num):
                chem_pots_endms[i] = self.liq_phs.gibbs_energy(temp, P_melt, mol=end_array[i,:])

            a[temp] = np.exp((chem_pots-chem_pots_endms)/ R / temp)
           
        self.a = pd.DataFrame(a,index=self.liq_phs.endmember_names).T