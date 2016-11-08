# Scikit-learn with Charged Higgs simulation data
How do different machine-learning classifiers fare with data from simulations of a charged Higgs boson mixed up with data from simulation of one of its background processes in the context of proton-proton scattering experiments (think, Large hadron collider type of experiments). 

The data is from simulations of proton collisions and the ATLAS detector. In a real life scenario, the background processes are much more complicated. This project studies a very simplified case where the only background contribution is from events that produce two top quarks (a top and an anti-top quark combination). The data is stored in a binary ROOT format, with more than 80 fields. 

This project shows how to convert this binary data to a format that scipy handles well -- pandas dataframes! It also shows how some feature engineering is done to narrow down on more impactful fields of the data. In addition, some distributions of the fields from the signal are overlayed with background to explore any glaring differences. Finally, classifiers are run from scipy and keras (for deep neural networks). The final models are then assessed using roc_curves, etc. 

Shoot me a message if you find this interesting... lastdylan@gmail.com
