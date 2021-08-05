# Surrogate Beamline

## Files
- shadowSurrogate.yml : Conda environment in use  
- torchTools.py : extra tools for simplifying PyTorch use

### Beamline files
- beamline_29.py                : Shadow3 implementation of simplified 29-ID (branch?)
- IEX_OO_Shadow_tests.ipynb     : Several runs with differing windows for sampling DOFs; also checked timing of simulation
- IEX_surrogate_NN.ipynb        : Training/validation of surrogate model to replace Shadow3 code
- IEX_shadow_vs_surrogate.ipynb : TODO compare timing of NN-surrogate vs Shadow3 simulation
- IEX_100k_04w.pkl              : Pickle of Shadow3 simulated data from IEX_OO_Shadow_tests.ipynb
- IEX_100k_04w_scaling.pkl      : Pickle of feature scaling used in IEX_OO_Shadow_tests.ipynb
- IEX_100k_04w_NN_results.pkl   : Pickle of models and results from IEX_surrogate_NN.ipynb

