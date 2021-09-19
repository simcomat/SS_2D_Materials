#%%
import os
import numpy as np
import warnings
import re
import pickle

from functools import cached_property


from pymatgen.io.vasp.outputs import *
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.electronic_structure.bandstructure import *
from pymatgen.electronic_structure.core import *
from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ase.dft.kpoints import *
from ase.calculators.vasp import Vasp2
from ase.dft.bandgap import bandgap

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 14})
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import LineCollection
#%%
class SSEntry:
    '''
    A class for computing Spin Splittings for 2D materials from a band structure calculation w/ VASP

    
    This code is built under such premisses:
    
    1. Only used for 2D materials placed along the xy plane (z axis is the stacking direction)
    
    2. A non-colinear bandstructure calculation was performed using ASE k-point generation scheme
    (input of a calculation permormed with another band-path convention might lead to undisered miss-interpretations)
    
    It relies on differnt modules of Pymatgen and ASE to do the parsing of the results, construction of the bandstructure data 
    and identification of the high-symmetry k-points.
    
    '''

    # Spins objects for pymatgen
    spin_up = Spin(1)
    spin_down = Spin(-1)

    trim_kpts = {'G':True, 'M':True, 'K':False, 'X':True, 'Y':True, 'S':True, 'C':True,  'H':False, 'H1':False}


    regex_header = r"# of k-points:\s+(\d+)\s+# of bands:\s+(\d+)\s+# of ions:\s+(\d+)"
    regex_kpoint = r" k-point\s+(\d+) :\s+(?: |-)(\d\.\d+)(?: |-)(\d\.\d+)(?: |-)(\d\.\d+)\s+"
    regex_band = r"band\s+(\d+) # energy\s+(-?\d+\.\d+) # occ.  (\d\.\d+)"
    regex_values = r"^(\s+\d+|tot)\s+(.*)"

    def __init__(self, calc_dir='.'):

        '''
        init function - get the basic properties necessary for computing the spin splitting types
        '''

        warnings.warn('''Warning:
        
The functions in this class are designed to work with band structures calculations of 2D materials whose k-point sampling path was generated with ASE
(see: wiki.fysik.dtu.dk/ase/ase/dft/kpoints.html#module-ase.dft.kpoints).
        
Calculations with other k-point scheme may not work properly with the algorithms implemented here.
        
        
        ''')   
        
        self.calc_dir = calc_dir
        self.procar_file = f'{self.calc_dir}/PROCAR'
        self.poscar_file = f'{self.calc_dir}/POSCAR'
        self.potcar_file = f'{self.calc_dir}/POTCAR'

        self.vasprun = Vasprun(f'{calc_dir}/vasprun.xml', parse_potcar_file=False)
        self.outcar = Outcar(f'{calc_dir}/OUTCAR')
        
        self.rec_kpts = self.vasprun.actual_kpoints
        self.bandstructure = self.vasprun.get_band_structure(line_mode=True)
        self.get_ase_prop()
        
        self.pymatgen_gap = self.bandstructure.get_band_gap()['energy']

        if self.pymatgen_gap > 0.0:

            self.vbm = self.bandstructure.get_vbm()
            self.cbm = self.bandstructure.get_cbm()

            self.band_indexes = {'vb':self.vbm['band_index'][self.spin_up][-1],
                                 'cb':self.cbm['band_index'][self.spin_up][0]}
        
            self.vbm_band = self.bands[self.band_indexes['vb'], :]
            self.cbm_band = self.bands[self.band_indexes['cb'], :]

        else:
            warnings.warn('''Warning - This material is metalic - 
            The SS functions implemented and tested in this class are for gapped materials only''')   



        #PROCAR related

        self.nkpoints, self.nbands, self.nions, self.nchannels = self.parse_initial_parameters()
        self.norbitals = len(self.orbital_list)
        self.nlines_per_point = self.nchannels * (self.nions + 1)
        self.ncols_per_point = self.norbitals + 1
        self.raw_data, self.energy_data, self.kpoints_data, self.occupancy_data, self.old_raw_data = self.parse_data()
        self.neq_ions = len(self.eq_ions)
        self.spin_texture = self.get_spin_texture()

    # Init functions

    def get_ase_prop(self):
        '''
        Gets important properties of the calculation using ASE functions
        '''
        self.ase_calc = Vasp2(restart=True, directory=self.calc_dir)
        self.bandgap = bandgap(self.ase_calc)[0]
        self.e_fermi = self.ase_calc.get_fermi_level()
        self.ase_struc = self.ase_calc.get_atoms()
        self.ase_cell = self.ase_struc.cell
        self.ase_lat = self.ase_cell.get_bravais_lattice(pbc=[1,1,0]) # Assuming that entry is 2D

    def parse_initial_parameters(self):
        """Simple parsing of PROCAR's header (2nd line) for nkpoints, nbands and nions
        Example: '# of k-points:  100         # of bands:   84         # of ions:    4'
        It also returns the nchannels attribute (1 for non-sp, 2 for sp and 4 for ncl)

        Returns:
            (int, int, int, int): Number of kpoints, number of bands, number of ions, nchannels
        """
        nlines = 0
        with open(self.procar_file, 'r') as f:
            got_to_1st_values_line = False
            for line in f:
                header_match = re.search(self.regex_header, line)
                if header_match:
                    nkpoints, nbands, nions = (int(n) for n in header_match.groups())
                    continue
                
                if line == '\n':
                    continue

                values_match = re.search(self.regex_values, line)
                if values_match:
                    got_to_1st_values_line = True
                    nlines += 1

                elif (got_to_1st_values_line) and (not values_match) and line!='\n':
                    nchannels = int(nlines/(nions + 1))

                    return nkpoints, nbands, nions, nchannels

    def parse_data(self):
        """Parsing of all data available for all data points in PROCAR.

        Returns:
            raw_data [numpy.array]: 4D array with dimensions (nbands, nkpoints, nlines_per_point, ncols_per_point)
            It contains all tabulated data of individual orbital contributions and spin for each band x kpoint point.

            energy_data, kpoints_data, occupancy {numpy.array}: 2D array with dimensions (nbands, nkpoints).
            It contains all energies, kpoints, occupancies for each band x kpoint point.
        """
        raw_data = np.full((self.nbands, self.nkpoints, self.nlines_per_point, self.ncols_per_point), fill_value=0.0)
        energy_data = np.full((self.nbands, self.nkpoints), fill_value=0.0)
        kpoints_data = np.full((self.nbands, self.nkpoints), fill_value=None)
        occupancy_data = np.full((self.nbands, self.nkpoints), fill_value=0.0)
        nrows = (self.nions + 1) * self.nchannels

        with open(self.procar_file, 'r') as f:
            for line in f:
                k_match = re.search(self.regex_kpoint, line)
                if k_match:
                    kpoint = int(k_match.groups()[0]) - 1
                    kx, ky, kz = (float(n) for n in k_match.groups()[1:])
                    continue

                bands_match = re.search(self.regex_band, line)
                if bands_match:
                    band = int(bands_match.groups()[0]) - 1
                    energy, occupancy = (float(n) for n in bands_match.groups()[1:])
                    energy_data[band, kpoint] = energy
                    kpoints_data[band, kpoint] = (kx, ky, kz)
                    occupancy_data[band, kpoint] = occupancy
                    point = []
                    continue

                values_match = re.search(self.regex_values, line)
                if values_match:
                    values = [np.nan if v == '******' else float(v) for v in values_match.groups()[-1].split()]
                    values = np.array(values)
                    point.append(values)
                    if len(point) == nrows:
                        raw_data[band, kpoint, :, :] = np.array(point)

        corrected_raw_data = self.correct_raw_data(raw_data)

        return corrected_raw_data, energy_data, kpoints_data, occupancy_data, raw_data


    def correct_raw_data(self, raw_data):
        ''' Corrects ****** values in PROCAR, recovering from the total sum, when possible'''

        data = np.copy(raw_data)

        for i in range(self.nbands):
            for j in range(self.nkpoints):
                for n in [1, 2, 3, 4]:
                    start_line = ((n-1)*self.nions)+(n-1)
                    end_line = (n*self.nions)+(n-1)
                    for l in np.arange(start_line, end_line, 1):
                        for p in range(len(data[i,j,l,:])-1):
                            value = data[i,j,l,p]
                            if math.isnan(value):
                                line_tot = data[i,j,l,-1]
                                column_tot = data[i,j,end_line,p]

                                if (math.isnan(column_tot)) and (not math.isnan(line_tot)):
                                    line_sum = 0
                                    for pp in range(len(data[i,j,l,:])-1):
                                        if pp != p:
                                            line_sum += data[i,j,l,pp]
                                    data[i,j,l,p] = line_tot - line_sum

                                    column_sum = 0
                                    for ll in np.arange(start_line, end_line, 1):
                                        column_sum += data[i,j,ll,p]
                                    data[i,j,end_line,p] = column_sum


                                elif (not math.isnan(column_tot)) and math.isnan(line_tot):
                                    column_sum = 0
                                    for ll in np.arange(start_line, end_line, 1):
                                        if ll != l:
                                            column_sum += data[i,j,ll,p]
                                    data[i,j,l,p] = column_tot - column_sum

                                    line_sum = 0
                                    for pp in range(len(data[i,j,l,:])-1):
                                        line_sum += data[i,j,l,pp]
                                    data[i,j,l,-1] = line_sum

                                elif math.isnan(column_tot) and math.isnan(line_tot):
                                    data[i,j,l,p] = -10.0

                                    line_sum = 0
                                    for pp in range(len(data[i,j,l,:])-1):
                                        line_sum += data[i,j,l,pp]
                                    data[i,j,l,-1] = line_sum

                                    column_sum = 0
                                    for ll in np.arange(start_line, end_line, 1):
                                        column_sum += data[i,j,ll,p]
                                    data[i,j,end_line,p] = column_sum

                n = self.nions+1
                tot_lines = [n-1,2*n-1,3*n-1,4*n-1]
                for l in tot_lines:
                    for p in range(len(data[i,j,l,:])-1):
                        if math.isnan(data[i,j,l,p]):
                            start_line = l - self.nions
                            column_sum = 0
                            for ll in np.arange(start_line, l, 1):
                                column_sum += data[i,j,ll,p]
                            data[i,j,l,p] = column_sum

                for l in range(data.shape[2]):
                    if (not l in tot_lines) and math.isnan(data[i,j,l,-1]):
                        line_sum = 0
                        for pp in range(len(data[i,j,l,:])-1):
                            line_sum += data[i,j,l,pp]
                        data[i,j,l,-1] = line_sum

                for l in tot_lines:
                    if math.isnan(data[i,j,l,-1]):
                        line_sum = 0
                        for pp in range(len(data[i,j,l,:])-1):
                            line_sum += data[i,j,l,pp]
                        data[i,j,l,-1] = line_sum

        return data
    # Atributes

    @cached_property
    def orbital_list(self):
        """Parsing of all orbitals in PROCAR

        Returns:
            [list]: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2'] if there are no f-orbitals
        """
        with open(self.procar_file, 'r') as f:
            for line in f:
                if line.startswith('ion'):
                    return line.split()[1:-1] # exclude 'ion' and 'tot' strings

    @cached_property
    def cart_kpts(self):
        '''
        Gets k-points in cartesian coordinates according to pymatgen.electronic_structure module
        '''
        kpt_list = []
        for k in self.bandstructure.kpoints:
            kpt_list.append(k.cart_coords)
        return kpt_list
    
    @cached_property
    def hs_kpts(self):
        '''
        High-Symmetry k-points defined by ASE
        '''
        self.get_ase_prop()
        return self.ase_lat.get_special_points()
    
    @cached_property
    def kpt_labels(self):
        '''
        List of k-point labels implemented by ASE according to:
        
        High-throughput electronic band structure calculations: Challenges and tools
        Wahyu Setyawan, Stefano Curtarolo
        Computational Materials Science, Volume 49, Issue 2, August 2010, Pages 299–312
        https://doi.org/10.1016/j.commatsci.2010.05.010
        '''
        label_list = ['-'] * len(self.rec_kpts)
        for label in self.hs_kpts.keys():
            for kpt_index in range(len(self.rec_kpts)):
                l1 = self.hs_kpts[label]
                l2 = self.rec_kpts[kpt_index]
                for i in range(3):
                    l1[i] = round(l1[i], 6)
                    l2[i] = round(l2[i], 6)
                comp = l1 == l2
                if comp.all():
                    label_list[kpt_index] = label
        return label_list
    
    @cached_property
    def hs_path(self):
        '''
        Helper function for returning the path along high-symmetry k-points 
        e.g.: ['G', 'M', 'K', 'G']
        '''
        path_list = []
        for k in self.kpt_labels: 
            if k != '-': 
                path_list.append(k)
        return path_list
    
    @cached_property
    def hs_path_indexes(self):
        '''
        Helper function for returning the indexes of high-symmetry k-points among the list of calculated k-points
        e.g.: [0, 25, 71, 124]
        '''
        hs_indexes = []
        for i in range(len(self.kpt_labels)): 
            if self.kpt_labels[i] != '-':
                hs_indexes.append(i)
        return hs_indexes
    
    @cached_property
    def bands(self):
        '''
        Uses pymatgen.electronic_structure module for constructing an array for bandstructure data
        '''
        if len(self.bandstructure.bands.keys()) != 1:
            warnings.warn('''Warning
            Diffetent spin channels handling is not implemented in this module.
            Results may be wrong''')   
        bands = self.bandstructure.bands[self.spin_up] 
        return bands

    @cached_property
    def eq_ions(self):
        """Parsing of all equivalent ions regarding species and symmetry by wyckoff positions

        Returns:
            eq ions [list]: List of tuples. Each tuple represents one group of equivalent ions and 
            has the following structure: (specie {string}, wyckoff symbol {string}, indices of ions {list of ints})

            Example: for Bi2O2-53ac438f321b --> [('Bi_d', '2i', [0, 1]), ('O', '2h', [2, 3])]
        """
        species = Potcar.from_file(self.potcar_file).symbols
        struct = Structure.from_file(self.poscar_file)
        sym_struct = SpacegroupAnalyzer(struct).get_symmetrized_structure()
        wyckoff_symbols = sym_struct.wyckoff_symbols
        eq_indices = sym_struct.equivalent_indices
        
        eq_ions = list(zip(species, wyckoff_symbols, eq_indices))

        return eq_ions

    @cached_property
    def orbital_contributions(self):
        """Individual ion and orbital contributions taking into account equivalent ions.

        Returns:
            orbital_dict [dictionary]: A dictionary containing a numpy.array with dimensions (nbands, nkpoints)
            for each individual ion and orbital (including 'total')

            Example: for Bi2O2-53ac438f321b --> Dictionary with keys ['(Bi_d-2i)_s', '(Bi_d-2i)_py', ..., '(O-2h)_x2-y2', '(O-2h)_total']
        """
        orbital_data = np.full((self.nbands, self.nkpoints, (self.neq_ions + 1), (self.norbitals + 1)), fill_value=0.0)
        normalized_raw_data = np.copy(self.raw_data)
        for i in range(self.nbands):
            for j in range(self.nkpoints):
                total = self.raw_data[i, j, self.nions, self.norbitals] 
                normalized_raw_data[i, j, :, :] = normalized_raw_data[i, j] / total
                
                for species_index in range(self.neq_ions):
                    ion_sum = np.full((1, self.norbitals + 1), 0.0)
                    for p in self.eq_ions[species_index][-1]:
                        ion_sum += normalized_raw_data[i,j,p,:]
                    
                    orbital_data[i,j,species_index,:] = ion_sum

        orbital_dict = dict()
        for i, (ion, wyckoff, _) in enumerate(self.eq_ions):
            for j, orbital in enumerate(self.orbital_list + ['total']):
                orbital_dict[f'({ion}-{wyckoff})_{orbital}'] = orbital_data[:,:,i,j]

        return orbital_dict

    @cached_property
    def orbital_projections(self):
        if self.raw_data.shape[3] == 10:
            orb_data = np.full((self.nbands, self.nkpoints, 9), fill_value=0.0)
            for i in range(self.nbands):
                for j in range(self.nkpoints):
                    orb_list = self.raw_data[i,j,self.nions,0:-1]
                    total_contrib = self.raw_data[i,j,self.nions,-1]
                    norm_list = orb_list/total_contrib

                    for k in range(len(norm_list)):
                        orb_data[i, j, k] = norm_list[k]

            orb_keys = ['s','py','pz','px','dxy','dyz','dz2','dxz','x2-y2']
            orb_dict = {}

            for orb_index, key in zip(range(9), orb_keys):
                orb_dict[key] = orb_data[:,:, orb_index]


            return orb_dict

    #cached property
    def get_spin_texture(self):
        """Spin texture parsed accordingly to the number of channels (nchannels) that is defined by the complexity
        of the vasp calculation (i.e, spin-polarized, non-colinear).

        Returns:
            spin_data [numpy.array]: 
            Simple calculations (nchannels = 1) have the shape (nbands, nkpoints) and each point is the total magnetization for that point.
            For spin-polarized the shape is (nbands, nkpoints, 2). The third dimension is for spin-up and spin-down channels.
            For non-colinear the shape is (nbands, nkpoints, 3) and this third dimension represents the x, y, z components of spin.
        """
        if self.nchannels == 1:
            spin_data = np.full((self.nbands, self.nkpoints), fill_value=0.0)
            for i in range(self.nbands):
                for j in range(self.nkpoints):
                    spin_data[i, j] = self.raw_data[i, j, self.nions, self.norbitals]

        elif self.nchannels == 2:
            spin_data = np.full((self.nbands, self.nkpoints, 2), fill_value=0.0)
            for i in range(self.nbands):
                for j in range(self.nkpoints):
                    spin_data[i, j, 0] = self.raw_data[i, j, self.nions, self.norbitals]
                    spin_data[i, j, 1] = self.raw_data[i, j, 2*(self.nions + 1)-1, self.norbitals]

        elif self.nchannels == 4:
            spin_data = np.full((self.nbands, self.nkpoints, 3), fill_value=0.0)
            for i in range(self.nbands):
                for j in range(self.nkpoints):
                    norm = self.raw_data[i, j, self.nions, self.norbitals]
                    spin_data[i, j, 0] = self.raw_data[i, j, 2*(self.nions + 1)-1, self.norbitals] / norm
                    spin_data[i, j, 1] = self.raw_data[i, j, 3*(self.nions + 1)-1, self.norbitals] / norm
                    spin_data[i, j, 2] = self.raw_data[i, j, 4*(self.nions + 1)-1, self.norbitals] / norm

        return spin_data            

    # Helper Functions

    def energy_check(self, band, kpt):
        '''
        Method for checking if band at this k-point in accessible within a range of energy
        '''
        energy_tol = 0.03 # eV
        if band == 'vb':
            diff = abs(self.vbm['energy'] - self.vbm_band[kpt])
            if diff <= energy_tol:
                return True
        if band == 'cb':
            diff = abs(self.cbm['energy'] - self.cbm_band[kpt])
            if diff <= energy_tol:
                return True
        return False
    
    def get_blank_ss_dict(self):
        '''
        Helper function of get_ss_types()
        
        Constructs the initial SS dictionary
        '''
        ss_dict = dict.fromkeys(self.hs_path_indexes)
        for k1 in ss_dict.keys():
            ss_dict[k1] = {'label':self.kpt_labels[k1],'vb':None, 'cb':None}
            for k2 in ss_dict[k1].keys():
                if k2 == 'vb' or k2 == 'cb':
                    ss_dict[k1][k2] = {'degenerated':'not_eval',
                                       'left_path':'not_eval', 'right_path':'not_eval'}
        return ss_dict
            
    def get_degen(self, ss_dict):
        '''
        Helper function of get_spin_splittings()
        '''
        for kpt in self.hs_path_indexes:
            for band in ['vb', 'cb']:
                if band == 'vb': inc = -1
                else: inc = 1
                    
                e1 = self.bands[self.band_indexes[band], kpt]
                e2 = self.bands[self.band_indexes[band]+inc, kpt]
                
                diff = abs(e1-e2)
                
                if diff <= 1e-4:
                    ss_dict[kpt][band]['degenerated'] = True
                    
                else:
                    if band == 'vb': edge_level = self.vbm['energy']
                    else: edge_level = self.cbm['energy']
                        
                    edge_diff = abs(e1 - edge_level)
                    
                    label = self.kpt_labels[kpt]
                    if self.trim_kpts[label] == True:
                        ss_dict[kpt][band] = {'degenerated':False, 'type':'TRIM-SS', 'spin_splitting':diff, 'accessibility':edge_diff}
                    else:
                        ss_dict[kpt][band] = {'degenerated':False, 'type':'ZSS', 'spin_splitting':diff, 'accessibility':edge_diff}
                    
        return ss_dict
    
    def get_kpt_boundaries(self, ss_dict):
        '''
        Helper function of get_ss_types()
        
        Identifies which high-symmetry k-points are terminal along the band-path
        insertiing the "does_not_exist" tag on the SS dictionary in appropriate locations
        '''
        
        for kpt in self.hs_path_indexes:
            if kpt == self.hs_path_indexes[0]:
                for band in ['vb', 'cb']:
                    ss_dict[kpt][band]['left_path'] = 'does_not_exist'
            elif kpt == self.hs_path_indexes[-1]:
                for band in ['vb', 'cb']:
                    ss_dict[kpt][band]['right_path'] = 'does_not_exist'
        return ss_dict

    def get_avg_ss(self, ini_k, final_k, direction, main_band_index, neighbor_band_index):

        if direction == 'left_path': inc = -1
        else: inc = +1

        ss_sum = 0.0
        kpts = np.arange(ini_k, final_k + inc, inc)
        for k in kpts:
            ss_diff = abs(self.bands[main_band_index, k]-self.bands[neighbor_band_index, k])
            ss_sum += ss_diff
        
        return ss_sum/len(kpts)
    
    def get_ss_types(self, ss_dict, ss_threshold=0.075, ss_avg_threshold=0.05):
        '''
        Function for computing spin splitting types and measuring the Rashba parameter for cases in which it applies.
        
        Not mentioned to be directly used, called by get_spin_spplitings()
        '''
        for kpt_index in range(len(self.hs_path_indexes)):
            kpt = self.hs_path_indexes[kpt_index]
            for band_name in ['vb', 'cb']:
                if ss_dict[kpt][band_name]['degenerated'] == True:
                    for direction in ['left_path', 'right_path']:
                        if ss_dict[kpt][band_name][direction] != 'does_not_exist':
                            
                            if direction == 'left_path':
                                kpt_increment = -1
                            elif direction == 'right_path':
                                kpt_increment = +1
                                
                            main_band = self.band_indexes[band_name]
                            
                            if band_name == 'vb':
                                neighbor_band = main_band - 1
                                edge_level = self.vbm['energy']
                            elif band_name == 'cb':
                                neighbor_band = main_band + 1   
                                edge_level = self.cbm['energy']
                                
                            #check if there is SS 2 kpoints appart the high-symmetry k-point
                            e1 = self.bands[main_band, kpt+(2*kpt_increment)]
                            e2 = self.bands[neighbor_band, kpt+(2*kpt_increment)]
                            diff = abs(e1-e2)
                            
                            #if there is no splitting -> type = "NSS"
                            if diff <= 1e-3:
                                ss_dict[kpt][band_name][direction] = {'type':'NSS'}
                                
                            else:
                                e1_kpt = self.bands[main_band, kpt]
                                e2_kpt = self.bands[neighbor_band, kpt]
                                    
                                d1 = (e1 - e1_kpt)/(abs(e1 - e1_kpt))
                                d2 = (e2 - e2_kpt)/(abs(e2 - e2_kpt))
                                    
                                # if splitting have the same derivatives -> High-Order Rashba
                                if d1 == d2: linear_ss = False
                                else: linear_ss = True
                                    
                                k = kpt+(2*kpt_increment)
                                e0_main = self.bands[main_band, k]
                                ei_main = self.bands[main_band, k+kpt_increment]
                                di_main = (ei_main - e0_main)/abs(ei_main - e0_main)
                                
                                while di_main == d1 and k < len(self.cart_kpts) -2 and k > 0:
                                    k = k + kpt_increment
                                    e0_main = self.bands[main_band, k]
                                    ei_main = self.bands[main_band, k+kpt_increment]
                                    di_main = (ei_main - e0_main)/abs(ei_main - e0_main)
                                
                                if di_main != d1: 
                                    band_index = main_band
                                    e0 = e0_main
                                    e_kpt = e2_kpt
                                
                                delta_e = abs(e0-e_kpt)
                                final_kpt = k
                                
                                ss = abs(self.bands[main_band, final_kpt] - self.bands[neighbor_band, final_kpt])
                                
                                edge_diff = abs(e0-edge_level)
                                
                                delta_k = np.linalg.norm(self.cart_kpts[final_kpt]-self.cart_kpts[kpt]) # Euclidean distance
                                
                                if linear_ss == True:
                                    rashba_param = 2*(delta_e/delta_k)

                                    avg_ss = self.get_avg_ss(ini_k=kpt, final_k=final_kpt, direction=direction,
                                                             main_band_index=main_band, neighbor_band_index=neighbor_band)
                                        

                                    
                                    if final_kpt in self.hs_path_indexes: 
                                        ss_in_hs = True
                                        ss_type = 'LSS-Z'
                                    else: 
                                        ss_in_hs = False
                                        if avg_ss >= ss_avg_threshold and ss >= ss_threshold:
                                            ss_type = 'LSS'
                                        else: ss_type = 'LSS-CB'
                                        
                                    
                                    ss_dict[kpt][band_name][direction] = {'type':ss_type, 'band':band_index,
                                                                          'final_k':final_kpt,
                                                                          'delta_e':delta_e, 'delta_k':delta_k,
                                                                          'rashba_param':rashba_param,
                                                                          'avg_ss':avg_ss,
                                                                          'spin_splitting':ss,
                                                                          'accessibility':edge_diff, 
                                                                          'max_in_another_hs_kpt':ss_in_hs}
                                    
                                else:
                                        
                                    if final_kpt in self.hs_path_indexes: 
                                        ss_in_hs = True
                                        ss_type = 'HOSS-Z'
                                    else: 
                                        ss_in_hs = False
                                        ss_type = 'HOSS'
                                        
                                    
                                    ss_dict[kpt][band_name][direction] = {'type':ss_type, 'band':band_index,
                                                                          'final_k':final_kpt,
                                                                          'delta_e':delta_e, 'delta_k':delta_k,
                                                                          'spin_splitting':ss,
                                                                          'accessibility':edge_diff, 
                                                                          'max_in_another_hs_kpt':ss_in_hs}
                            
        return ss_dict

    def check_duplicated_LSS(self, ss_dict):

        L = list(ss_dict.keys())

        for left_kpt, right_kpt in zip(L, L[1:]):
            for band in ['vb', 'cb']:
                if ss_dict[left_kpt][band]['degenerated'] == True and ss_dict[right_kpt][band]['degenerated'] == True:
                    left_ss = ss_dict[left_kpt][band]['right_path']
                    right_ss = ss_dict[right_kpt][band]['left_path']
                    if left_ss['type'] == 'LSS' and right_ss['type'] == 'LSS':
                        if left_ss['final_k'] == right_ss['final_k']:
                            left_diff = abs(left_ss['final_k']-left_kpt)
                            right_diff = abs(right_ss['final_k']-right_kpt)
                            if left_diff > right_diff: ss_dict[left_kpt][band]['right_path']['type'] = 'LSS-Duplicated'
                            elif left_diff < right_diff: ss_dict[right_kpt][band]['left_path']['type'] = 'LSS-Duplicated'

        return ss_dict
        
    # Methods 

    def get_spin_splittings(self):
        '''
        Main function of the class for addressing spin splittings in high-symmetry k-points
        
        Consists in calls of different helper functions to construct the SS dictionary and fill the spin splittings information 
        for each high-symmetry k-point in the VBM/CBM level
        
        The SS dictionary has the following structure:
        
        - high-symmetry k-point index
            - k-point label
            - vbm:
                - if the eigenvalue in deegenerated
                - right_path:
                    - spin splitting type 
                        -"NSS" for no spin splitting
                        -"HOSS" for High-Order Rashba
                        -"LSS" for linear SS (Rashba/Dresselhaus)
                        -"ZSS" for Zeeman Spin Splitting
                    - Delta E (if spin splitting type is "LSS" or "ZSS")
                    - Delta K (if spin splitting type is "LSS" or "ZSS")
                    - Rashba parameter (if spin splitting type is "LSS" or "ZSS")
                    - accessibility: the difference in energy from the maximum splitting point and the VBM/CBM (if spin splitting type is "LSS" or "ZSS")
                    - The sign of the effective mass considered when evaluating the spin splittings (if spin splitting type is "LSS" or "ZSS") 
                - left_path:
                    same as "right_path"
            - cbm:
                same as "vbm"
        '''
        ss_dict = self.get_blank_ss_dict()
        ss_dict = self.get_degen(ss_dict)
        ss_dict = self.get_kpt_boundaries(ss_dict)
        ss_dict = self.get_ss_types(ss_dict)
        ss_dict = self.check_duplicated_LSS(ss_dict)
        
        return ss_dict
    
    def get_largest_ss(self):
        '''
        Method for getting the lagest difference in eigenvalues 
        for different spins in non-degenerated bands across the bandstructure
        
        Used for having a first grance of the largest spin splitting in the band edges
        not meant for identifying a SS of a particular type
        '''
        max_ss = {}
        
        for key in ['vb', 'cb']:
            band_index = self.band_indexes[key]
            
            if key == 'vb': neighbor_band = band_index-1
            else: neighbor_band = band_index+1
                
            max_diff = 0
            kpt = -1
            for i in range(len(self.rec_kpts)):
                diff = abs(self.bands[band_index, i]-self.bands[neighbor_band, i])
                if diff > max_diff: 
                    max_diff = diff
                    kpt = i
                    
            kpt_label = '-'
            for k in range(len(self.hs_path_indexes)):
                if kpt == self.hs_path_indexes[k]:
                    kpt_label = self.hs_path[k]
                    break
                if kpt < self.hs_path_indexes[k]:
                    kpt_label = f'{self.hs_path[k-1]} -> {self.hs_path[k]}'
                
            
            max_ss[key] = {'splitting':max_diff, 'loc':kpt_label}
        
        return max_ss

    def spin_plot(self, projection='z', e_range=1.5, filename='plot', save=False, save_dir = '.'):

        upper_lim = e_range + self.bandgap
        lower_lim = - e_range
        
        if projection == 'x': axis = 0
        elif projection == 'y': axis = 1
        elif projection == 'z': axis = 2

        plt.figure(dpi=1200)
        fig, ax = plt.subplots()

        cmap = cm.get_cmap('coolwarm', 500)
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        for b in np.arange(1, self.bands.shape[0]-1):
        
            spin = self.spin_texture[b, :, axis]

            for x in range(self.bands.shape[1]):
            
                y = self.bands[b,x] - self.e_fermi

                if y > lower_lim and y < upper_lim:
                
                    y_neighbors = {'+': 0.0, '-': 0.0}
                    degen = False

                    y_neighbors['+'] = self.bands[b+1,x] - self.e_fermi
                    y_neighbors['-'] = self.bands[b-1,x] - self.e_fermi

                    for key in y_neighbors.keys():
                        if abs(y_neighbors[key] - y) <= 1e-4:
                            degen = True

                    if degen:
                        if b % 2 == 0 and x % 2 == 0:
                            ax.plot(x, y, marker='o', markersize=1, color=cmap(norm(spin[x])))
                        elif b % 2 == 1 and x % 2 == 1:
                            ax.plot(x, y, marker='o', markersize=1, color=cmap(norm(spin[x])))
                    else:
                        ax.plot(x, y, marker='o', markersize=1, color=cmap(norm(spin[x])))


        plt.ylim(lower_lim, upper_lim)

        for i in range(len(self.hs_path)):
            if self.hs_path[i] == 'G':
                self.hs_path[i] = '$\Gamma$'

        plt.xticks(self.hs_path_indexes, self.hs_path)
        for kpt in self.hs_path_indexes:
            ax.axvline(x=kpt, color='black', linewidth = 1)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm)

        fig.patch.set_facecolor('xkcd:white')
        plt.margins(x=0)
        plt.tight_layout()
        plt.title(f'$S_{projection}$')
        plt.ylabel('E - E$_F$ [eV]')
        fig.set_figheight(4)
        fig.set_figwidth(5.5)
        plt.tight_layout()
        if save == True:
            plt.savefig(f'{save_dir}/{filename}.png', dpi=400, transparent=False)
        plt.show()

    def band_structure_plot(self, e_range=2, filename='plot', plot_title='Band Structure', save=False, save_dir = '.'):

        upper_lim = e_range + self.bandgap
        lower_lim = - e_range

        plt.figure(dpi=1200)
        fig, ax = plt.subplots()

        for b in range(self.bands.shape[0]):
            ax.plot(range(self.bands.shape[1]), self.bands[b,:]-self.e_fermi, color='k')

        plt.ylim(lower_lim, upper_lim)

        for i in range(len(self.hs_path)):
            if self.hs_path[i] == 'G':
                self.hs_path[i] = '$\Gamma$'

        plt.xticks(self.hs_path_indexes, self.hs_path)
        for kpt in self.hs_path_indexes:
            ax.axvline(x=kpt, color='black', linewidth = 1)

        fig.patch.set_facecolor('xkcd:white')
        plt.margins(x=0)
        plt.tight_layout()
        plt.title(plot_title)
        plt.ylabel('E - E$_F$ [eV]')
        fig.set_figheight(4)
        fig.set_figwidth(5.5)
        plt.tight_layout()
        if save == True:
            plt.savefig(f'{save_dir}/{filename}.png', dpi=1200)
        plt.show()

    def check_anti_crossing(self, test_plot=False):

        ss_dict = self.get_spin_splittings()
        ss_points = [] 


        for kpt in ss_dict.keys():
            if ss_dict[kpt]['vb']['degenerated'] == True and ss_dict[kpt]['cb']['degenerated'] == True:
                for path in ['left_path','right_path']:
                    duplicate_control = False
                    if ss_dict[kpt]['vb'][path] != 'does_not_exist':
                        band_list_1 = ['vb', 'cb']
                        band_list_2 = ['cb', 'vb']
                        for main_band, neighbor_band in zip(band_list_1, band_list_2):
                            main_band_type = ss_dict[kpt][main_band][path]['type']
                            if main_band_type == 'LSS':
                                for neighbor_type in ['LSS', 'LSS-CB', 'HOSS']:
                                    if ss_dict[kpt][neighbor_band][path]['type'] == neighbor_type:
                                        kpt_distance = abs(ss_dict[kpt][main_band][path]['final_k']-ss_dict[kpt][neighbor_band][path]['final_k'])
                                        if kpt_distance <= 30:
                                            if neighbor_type != 'HOSS':
                                                if not duplicate_control:
                                                    ss_points.append({'hs_kpt': kpt, 'label':ss_dict[kpt]['label'], 'direction': path, 'anti_crossing':False,
                                                    main_band: {'type': main_band_type, 'band_index': ss_dict[kpt][main_band][path]['band'], 'kpt_index':ss_dict[kpt][main_band][path]['final_k'], 'rashba_param':ss_dict[kpt][main_band][path]['rashba_param'], 'spin_splitting':ss_dict[kpt][main_band][path]['spin_splitting'], 'accessibility':ss_dict[kpt][main_band][path]['accessibility']},
                                                    neighbor_band: {'type': neighbor_type, 'band_index': ss_dict[kpt][neighbor_band][path]['band'], 'kpt_index':ss_dict[kpt][neighbor_band][path]['final_k'], 'rashba_param':ss_dict[kpt][neighbor_band][path]['rashba_param'], 'spin_splitting':ss_dict[kpt][neighbor_band][path]['spin_splitting'], 'accessibility':ss_dict[kpt][neighbor_band][path]['accessibility']}})
                                                if neighbor_type == 'LSS': duplicate_control = True
                                            else:
                                                ss_points.append({'hs_kpt': kpt, 'label':ss_dict[kpt]['label'], 'direction': path, 'anti_crossing':False,
                                                main_band: {'type': main_band_type, 'band_index': ss_dict[kpt][main_band][path]['band'], 'kpt_index':ss_dict[kpt][main_band][path]['final_k'], 'rashba_param':ss_dict[kpt][main_band][path]['rashba_param'], 'spin_splitting':ss_dict[kpt][main_band][path]['spin_splitting'], 'accessibility':ss_dict[kpt][main_band][path]['accessibility']},
                                                neighbor_band: {'type': neighbor_type, 'band_index': ss_dict[kpt][neighbor_band][path]['band'], 'kpt_index':ss_dict[kpt][neighbor_band][path]['final_k'], 'rashba_param':-1, 'spin_splitting':ss_dict[kpt][neighbor_band][path]['spin_splitting'], 'accessibility':ss_dict[kpt][neighbor_band][path]['accessibility']}})

        def check_direction(L):
            increasing = all(x<y for x, y in zip(L, L[1:]))
            decreasing = all(x>y for x, y in zip(L, L[1:]))
        
            if increasing: return 1
            elif decreasing: return -1
            elif not increasing and not decreasing: return 0

        orb_contrib_list = []
        for orb_name in list(self.orbital_contributions.keys()):
            if orb_name.split('_')[-1] != 'total':
                orb_contrib_list.append(orb_name)



        for ss_index in range(len(ss_points)):
            anti_crossing_result = {}
            if test_plot:
                cb_color = 'crimson'
                fig, axs = plt.subplots(len(orb_contrib_list)+1)
                fig.set_figheight(3*len(orb_contrib_list))
                fig.set_figwidth(7)
                fig.patch.set_facecolor('xkcd:white')

                for b, c in zip([-1, 0, 1, 2],['k','b',cb_color,'k']):
                    
                    axs[0].plot(range(self.nkpoints), self.bands[ss_points[ss_index]['vb']['band_index']+b, :], color=c)
                    axs[0].axvline(x=ss_points[ss_index]['vb']['kpt_index'], linestyle='dashed', color = 'b', linewidth = 1)
                    axs[0].axvline(x=ss_points[ss_index]['cb']['kpt_index'], linestyle='dashed', color = cb_color, linewidth = 1)

                    #for i in range(len(self.hs_path)):
                    #    if self.hs_path[i] == 'G':
                    #        self.hs_path[i] = '$\Gamma$'

                    axs[0].set_xticks(self.hs_path_indexes)
                    axs[0].set_xticklabels(self.hs_path)
                    for kpt in self.hs_path_indexes:
                        axs[0].axvline(x=kpt, color='black', linewidth = 1)

                    axs[0].margins(x=0)
                    axs[0].set_title('Selected pair of points in Band Structure')



            orb_index = 0
            for orb in orb_contrib_list:
                
                anti_crossing_result[orb] = False

                if ss_points[ss_index]['direction'] == 'right_path': inc = 1
                else: inc = -1

                kpts_array_vbm = np.arange(ss_points[ss_index]['hs_kpt'],ss_points[ss_index]['vb']['kpt_index']+inc, inc)
                kpts_array_cbm = np.arange(ss_points[ss_index]['hs_kpt'],ss_points[ss_index]['cb']['kpt_index']+inc, inc)

                vbm_contrib = self.orbital_contributions[orb][ss_points[ss_index]['vb']['band_index'], kpts_array_vbm]
                cbm_contrib = self.orbital_contributions[orb][ss_points[ss_index]['cb']['band_index'], kpts_array_cbm]



                vbm_direction = check_direction(vbm_contrib)
                cbm_direction = check_direction(cbm_contrib)

                if vbm_direction * cbm_direction == -1:
                    anti_crossing_result[orb] = True
                    ss_points[ss_index]['anti_crossing'] = True

                if test_plot == True:
                    orb_index += 1
                    axs[orb_index].plot(kpts_array_vbm, vbm_contrib, label='vb', color='b')
                    axs[orb_index].plot(kpts_array_cbm, cbm_contrib, label='cb', color= cb_color)
                    axs[orb_index].axvline(x=ss_points[ss_index]['vb']['kpt_index'], linestyle='dashed', color = 'b', linewidth = 1)
                    axs[orb_index].axvline(x=ss_points[ss_index]['cb']['kpt_index'], linestyle='dashed', color = cb_color, linewidth = 1)
                    if anti_crossing_result[orb]: ac_color = 'green'
                    else: ac_color = 'red'
                    axs[orb_index].tick_params(color=ac_color, labelcolor=ac_color)
                    for spine in axs[orb_index].spines.values():
                        spine.set_edgecolor(ac_color)
                    axs[orb_index].set_title(orb)
                    axs[orb_index].legend(loc='best')
                    axs[orb_index].set_xticks([])
        
            ss_points[ss_index]['orbital_dict'] = anti_crossing_result

            if test_plot == True:
                plt.tight_layout()
                plt.show()

        
        return ss_points

    def select_spin_splittings(self, access_range = 999, ss_threshold = 0.001):

        ss_dict = self.get_spin_splittings()
        ac_pair_list = self.check_anti_crossing()

    
        ss_lists = {'LSS':{}, 'HOSS':{}, 'ZSS':{}}

        for ss_prototype in ['LSS', 'HOSS', 'ZSS']:
            ss_points = {'vb':[], 'cb':[]}
            for kpt in ss_dict.keys():
                for band in ['vb', 'cb']:

                    if ss_prototype=='LSS' or ss_prototype=='HOSS':
                        if ss_dict[kpt][band]['degenerated'] == True:
                            for path in ['left_path','right_path']:
                                if ss_dict[kpt][band][path] != 'does_not_exist':
                                    if ss_dict[kpt][band][path]['type'] == ss_prototype:
                                        if (ss_dict[kpt][band][path]['accessibility'] <= access_range) and (ss_dict[kpt][band][path]['spin_splitting'] >= ss_threshold):

                                            has_ac = False
                                            for ss_pair_dict in ac_pair_list:
                                                if ss_pair_dict['anti_crossing'] == True:
                                                    if (kpt == ss_pair_dict['hs_kpt']) and (path == ss_pair_dict['direction']):
                                                        has_ac = True


                                            if ss_prototype=='LSS':
                                                ss_points[band].append({'hs_kpt': kpt, 'label':ss_dict[kpt]['label'], 'direction': path,
                                                'band_index': ss_dict[kpt][band][path]['band'], 'kpt_index':ss_dict[kpt][band][path]['final_k'],
                                                'delta_e': ss_dict[kpt][band][path]['delta_e'], 'delta_k':ss_dict[kpt][band][path]['delta_k'],
                                                'rashba_param':ss_dict[kpt][band][path]['rashba_param'], 'spin_splitting':ss_dict[kpt][band][path]['spin_splitting'],
                                                'accessibility':ss_dict[kpt][band][path]['accessibility'], 'anti_crossing': has_ac})
                                            elif ss_prototype=='HOSS':
                                                ss_points[band].append({'hs_kpt': kpt, 'label':ss_dict[kpt]['label'], 'direction': path,
                                                'band_index': ss_dict[kpt][band][path]['band'], 'kpt_index':ss_dict[kpt][band][path]['final_k'],
                                                'delta_e': ss_dict[kpt][band][path]['delta_e'], 'delta_k':ss_dict[kpt][band][path]['delta_k'],
                                                'spin_splitting':ss_dict[kpt][band][path]['spin_splitting'],
                                                'accessibility':ss_dict[kpt][band][path]['accessibility'], 'anti_crossing' : has_ac})

                    elif ss_prototype=='ZSS':
                        if ss_dict[kpt][band]['degenerated'] == False:
                            if (ss_dict[kpt][band]['accessibility'] <= access_range) and (ss_dict[kpt][band]['spin_splitting'] >= ss_threshold):
                                ss_points[band].append({'hs_kpt': kpt, 'label':ss_dict[kpt]['label'],
                                'spin_splitting':ss_dict[kpt][band]['spin_splitting'], 'accessibility':ss_dict[kpt][band]['accessibility']})
            
            ss_lists[ss_prototype] = ss_points

        return ss_lists


    def get_bandstructure_data(self, save_pickle=False, save_dir = '.', file_name='bandstructure'):

        bandstrucure_data = {}

        bandstrucure_data['nkpoints'] = self.nkpoints
        bandstrucure_data['nbands'] = self.nbands
        bandstrucure_data['efermi'] = self.e_fermi
        bandstrucure_data['cell'] = self.ase_lat.bandpath().cell
        bandstrucure_data['rec_cell'] = self.ase_lat.bandpath().icell

        labels_dict = {}
        kpt_labels = self.kpt_labels
        for i in range(len(kpt_labels)):
            if kpt_labels[i] != '-':
                labels_dict[i] = kpt_labels[i]
        bandstrucure_data['labels_dict'] = labels_dict

        bandstrucure_data['rec_kpoints_coords'] = self.rec_kpts

        bandstrucure_data['energies'] = self.bands

        bandstrucure_data['orbital_projections'] = self.orbital_contributions
        bandstrucure_data['spin_projections'] = self.get_spin_texture()

        if save_pickle == True:
            with open(f'{save_dir}/{file_name}.pickle', 'wb') as outfile:
                pickle.dump(bandstrucure_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        
        return bandstrucure_data

    def get_max_raw_ss(self):
        max_ss = {'vb':0, 'cb':0}

        for band in ['vb', 'cb']:
            if band == 'vb': inc = -1
            else: inc = 1

            for k in range(self.nkpoints):
                e1 = self.bands[self.band_indexes[band], k]
                e2 = self.bands[self.band_indexes[band]+inc, k]

                dE = abs(e1-e2)
                if (dE >= 0.001) and (dE > max_ss[band]): max_ss[band] = dE
                    
        return max_ss
                



#%%
