# Green Functions with Quantum Circuits

## Overview
This repository demonstrates the computation of the Green function for the ground state of the one-site Hubbard model using quantum circuits, with 
[myQLM](https://github.com/Atos-Quantum/myqlm). It employs the Variational Quantum Eigensolver (VQE) and measures the real-time retarded Green function.

## Repository Structure
- **`green_qc.py`**: Defines variational and Green function measurement circuits.
- **`green_1site_Hubbard.ipynb`**: Runs VQE, computes, and plots the Green function.

## Installation
Install required libraries:

```bash
pip install myqlm numpy matplotlib scipy
```

## Usage
1. Open `green_1site_Hubbard.ipynb`.
2. Run cells to execute VQE and compute the Green function.
3. View plotted Green function results.


## Future Work
- Extend to the Anderson Impurity model.
- Develop a shallow circuit for measuring the Green function in the frequency space.


## License
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

---
This repository explores quantum computing for Green function calculations.





