# cryoSPHERE: Single-particle heterogeneous reconstruction from cryo EM

CryoSPHERE is a structural heterogeneous reconstruction software of cryoEM data. It requires an estimate of the CTF and poses of each image. This can be obtained using other softwares.
CryoSPHERE works with two yaml files: one `parameters.yaml` describing the hyperparameters used to train cryoSPHERE and a `image.yaml` file, describing the images in the dataset. You can find a commented example of these files in the repository.  

The file `parameters.yaml` describes how to fit a single segmentation for the entire protein.
The file `parameters_two_segmentation.yaml` describes how you can segment chain A and chain C (for example) only, and how to create a custom starting segmentation and define a custom prior on the segmentation. Note that if these custom starting segmentation and prior are not defined, the values of the segmentation are taken so that it is uniform. See the class Segmentation in the file segmentation.py.

## Installation

CryoSPHERE is available as a python package named `cryosphere`. Create a conda environment, install cryosphere with `pip` and then `pytorch3d`:
```
conda create -n cryosphere python==3.9.20
conda activate cryosphere
pip install cryosphere
conda install pytorch3d -c pytorch3d -c conda-forge
```

## Training
### Preliminary: consensus reconstruction.
Before running cryoSPHERE on a dataset you need  to run a homogeneous reconstruction software such as RELION or cryoSparc. This should yield a star file containing the poses of each image, the CTF and information about the images as well as one or several mrcs file(s) containing the actual images. You should also obtain one or several mrc files corresponding to consensus reconstruction(s). For this tutorial, we assume your images are in a file called `particles.mrcs` and after a consensus reconstruction, you obain a star file named `particles.star` and a consensus reconstruction file called `consensus_map.mrc`. This naming is not mandatory, your files can have arbitrary names as long as the extension is correct. CryoSPHERE would also work with data preprocessed by cryoSparc. In that case you can directly use the `particles.cs` file.

This step is important to obtain an estimation of the CTF and the pose of each image. 

### First step: centering the structure
Fit a good atomic structure of the protein of interest into the volume obtained in the prelimiary step (`consensus_map.mrc`), using e.g ChimeraX. Save this structure in pdb format: `fitted_structure.pdb`. You can now use cryoSPHERE's command line tools to center the structure and volume:
```
cryosphere_center_origin --pdb_file_path fitted_structure.pdb --mrc_file_path consensus_map.mrc
```
This yields a pdb file `fitted_structure_centered.pdb` of the centered structure and a mrc file `consensus_map_centered.mrc` of the centered consensus volume.

### First step bis (optional)
Since the dataset is usually very noisy, it might be helpful to apply a low pass filter to the images. To determine the bandwith cutoff, first turn the centered structure into a volume, using the same GMM representation of the protein as is used during training cryoSPHERE:
```
cryosphere_structure_to_volume --image_yaml /path/to/image.yaml --structure_path/path/to/fitted_structure_centered.pdb --output_path /path/to/fitted_structure_centered_volume.mrc
```
You can now compute the Fourier Shell Correlation (FSC) between `fitted_structure_centered_volume.mrc` and `consensus_map_centered.mrc` using available softwares: e.g the e2proc3d command of EMAN2 or online [here](https://www.ebi.ac.uk/emdb/validation/fsc/). 
Next, find the frequency `cutoff_freq` for which the FSC is equal to 0.5, and set `lp_bandwidth: 1/cutoff_freq` in the `parameters.yaml`. This means that the in the images, the frequencies such that `freq > 1/lp_bandwidth` are set to 0.

### Second step

The second step is to run cryoSPHERE. To run it, you need  two yaml files: a `parameters.yaml` file, defining all the parameters of the training run and a `image.yaml` file, containing informations about the images. You need to set the `folder_experiment` entry of the paramters.yaml to the path of the folder containing your data. You also need to change the `base_structure` entry to `fitted_structure_centered.pdb`. You can then run cryosphere using the command line tool:
```
cryosphere_train --experiment_yaml /path/to/parameters.yaml
```
This command creates a folder named `cryoSPHERE` which contains the PyTorch models `ckpt.pt`, one at the end of each epoch. It also copies the `parameters.yaml` and `image.yaml` files in this directory and creates a `run.log` to log training data.

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
 If you want to generate all structures (one for each images), you can set `--generate_structures` instead. This will skip the PCA step. The file `z.npy` contains the latent variable associated to each image (in the same order as the images in the star file), the `.pdb` files are the structures sampled along the principal component (from lowest to highest values along that PC) and the `.png` files are images of the PCA decompisitions.

It is also possible to get the structures corresponding to specific images. Save the latent variables corresponding to the images of interest into a `z_interest.npy`. You can then run:
```
cryosphere_analyze --experiment_yaml /path/to/parameters.yaml --model /path/to/model.pt --output_path /path/to/outpout_folder --z /path/to/z_interest.npy --generate_structures
``` 
Setting the `--z /path/to/z_interest.npy` argument will directly decode the latent variables in `z_interest.npy` into structures.
 
