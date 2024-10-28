# cryoSPHERE: Single-particle heterogeneous reconstruction from cryo EM

CryoSPHERE is a structural heterogeneous reconstruction software of cryoEM data.

## Installation

CryoSPHERE is available as a python package named `cryosphere`. Create a conda environment, install cryosphere with `pip` and then `pytorch3d`:
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

## Analysis

You can first get the latent variables corresponding to the imagaes and generate a PCA analysis of the latent space, with latent traversal of first principal components::
```
cryosphere_analyze --experiment_yaml /path/to/parameters.yaml --model /path/to/model.pt --output_path /path/to/outpout_folder --no-generate_structures
```
where `model.pt` is the saved torch model you want to analyze and output_folder is the folder where you want to save the results of the analysis.
This will create the following directory structure:
```
analysis
   |	z.npy
   |	pc0
	   |   structure_z_1.pdb
	   .
	   .
	   .
	   |   structure_z_10.pdb
           |   pca.png

	pc1
	   |   structure_z_1.pdb
	   .
	   .
           .
```
 If you want to generate all structures (one for each images), you can set `--generate_structures` instead. This will skip the PCA step.

It is also possible to get the structures corresponding to specific images. Save the latent variables corresponding to the images of interest into a `z_interest.npy`. You can then run:
```
cryosphere_analyze --experiment_yaml /path/to/parameters.yaml --model /path/to/model.pt --output_path /path/to/outpout_folder --z /path/to/z_interest.npy --generate_structures
``` 
Setting the `--z /path/to/z_interest.npy` argument will directly decode the latent variables in `z_interest.npy` into structures.
 
