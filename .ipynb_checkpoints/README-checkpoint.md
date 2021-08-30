# Surrogate Beamline

### Code
- beamline_29.py
    - shadow implmentation fo 29ID (don't recall which branch)
- torchTools.py 
    - homebrew tools to hasten NN build/training/testing

### Data files
- IEX_100k_04w.pkl
    - Pickle of Shadow3 simulated data from IEX_OO_Shadow_tests.ipynb
- IEX_100k_04w_scaling.pkl
    - Pickle of feature scaling used in IEX_OO_Shadow_tests.ipynb
- IEX_100k_04w_NN_results.pkl
    - Pickle of models and results from IEX_surrogate_NN.ipynb

### Notebooks
- IEX_OO_Shadowtests.ipynb 
    - Testing out beamline_29.py
    - playing with ranges of each DOF
    - build data set for surrogate
- IEX_surrogate_NN.ipynb
    - train and test NN on shadow generated data
- IEX_shadow_vs_surrogate.ipynb
    - test time usage of both beamlines
- GA_vs_beamline.ipynb
    - __in progress__
    - goal to test out simple genetic algorithm to find _optimal_ beamline configuration

### Upcoming
- Gaussian process implementation (variation of MG-GPO)
- Bluesky version of beamline 
    - Either adapt beamline_29.py or surrogate model
    - Construct plan that can implement optimization technqiue
        - GA
        - MP-GPO
        - ???
