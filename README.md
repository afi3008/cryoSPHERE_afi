# cryoSPHERE: Single-particle heterogeneous reconstruction from cryo EM

CryoSPHERE is a structural heterogeneous reconstruction software of cryoEM data.

## Installation

CryoSPHERE is available as a python package named `cryosphere`. Create a conda environment, install cryosphere with `pip` and then ``pytorch3d`:
```
conda create -n cryosphere python==3.9.20
conda activate cryosphere
pip install cryosphere
conda install pytorch3d -c pytorch3d
```

## Training
The first step before running cryoSPHERE on a dataset is to run a homogeneous reconstruction software such as RELION or cryoSparc. This should yield a star file containing the poses of each image, the CTF and information about the images as well as one or several mrcs file(s) containing the actual images. You should also obtain one or several mrc files corresponding to consensus reconstruction(s). For example, you obtained a `conensus_map.mrc`
The second step is to fit a good atomic structure of the protein of interest into the volume obtained at step one (`consensus_map.mrc`), using e.g ChimeraX. Save this structure in pdb format: `fitted_structure.pdb`. You can now use cryopshere command line tools to center the structure:
```
cryosphere_center_origin --pdb_file_path fitted_structure.pdb --mrc_file_path consensus_map.mrc
```
This yields another pdb file `fitted_structure_centered.pdb".

The third step is to run cryoSPHERE. To run it, you need  two yaml files: a `parameters.yaml` file, defining all the parameters of the training run and a `image.yaml` file, containing informations about the images. You need to set the `folder_experiment` entry of the paramters.yaml to the path of the folder containing your data. You also need to change the `base_structure` entry to `fitted_structure_centered.pdb`. You can then run cryosphere using the command line tool:
```
cryosphere_train --experiment_yaml /path/to/parameters.yaml
```



 
