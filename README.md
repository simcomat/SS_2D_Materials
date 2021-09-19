# SS_2D_Materials

Python class designed to automatically identify and measure Spin Splitting (SS) effects in 2D materials, from DFT band structure calculations performed with [VASP](https://www.vasp.at/). It relies heavily on tools and functionalities implemented in [Pymatgen](https://github.com/materialsproject/pymatgen) and the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) python libraries.

## Requirements / Premises:

The code was designed to work with output files from non-collinear band structure DFT calculations performed with VASP for 2D materials. It follows the following premises:

- The calculation correspond to a compound with a finite electronic band gap: metallic materials are currently not supported by the algorithm.
- The monolayer is disposed in the $xy$ plane of the unit cell: the algorithm forces periodic boundary conditions only in these directions, necessary for the correct description of the first Brillouin Zone for each system.
- The path in the reciprocal space employed in the band structure calculation was generated with [ASE's Brillouin sampling automatic scheme](https://wiki.fysik.dtu.dk/ase/ase/dft/kpoints.html)

## Usage

The code initialization requires only the specification of the folder path where the band structure calculation was performed:

```python
from SS_2D_Materials.SSEntry import SSEntry

entry = SSEntry('path/to/calculation/folder')
```

### Implemented methods:

- `select_spin_splittings()`: Main method for identifying and measuring SS effects in the material valance and conduction bands. It returns a nested dictionary it the keys corresponding to the the different proposed SS Prototypes, namely Linear Rashba/Dresselhaus SS (`LSS`), Zeeman SS (`ZSS`) and High-Order SS (`HOSS`). Inside each dictionary key, a list of dictionaries is presented with each one corresponding to a single identified SS, following are the main SS properties computed and available inside each dictionary:
	- `label`: High-symmetry k-point label.
	- `direction`: Direction of the k-path segment analyzed by the algorithm in the vicinity of the high-symmetry k-point.
	- `rashba_param`: Measured Rashba coefficient.
	- `spin_splitting`: Measured SS magnitude.
	- `accessibility`: Energy difference from the SS to the VBM/CBM of its corresponding band.
	- `anti_crossing`: Presence of anti-crossing bands among aligned SS across valance and conduction bands.

- `band_structure_plot()`: Method for quickly plotting the materials band structure, based on the eigenvalues parsed by Pymatgen.

- `spin_plot()`: Method for plotting the materials band structure with spin texture resolution according to a specified `projection` axis, based on the values parsed from the `PROCAR` VASP output file.

- `get_max_raw_ss()`: Method for computing the maximum value for the SS magnitude in the materials valance and conduction bands.