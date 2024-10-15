import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import mrc
import yaml
import torch
import utils
import mrcfile
import argparse
import starfile
import numpy as np
from ctf import CTF
import seaborn as sns
from time import time
from tqdm import tqdm
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from polymer import Polymer
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from dataset import ImageDataSet
from gmm import Gaussian, EMAN2Grid
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from pytorch3d.transforms import quaternion_to_axis_angle, quaternion_to_matrix


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True, help="path to the yaml defining the experimentation")
parser_arg.add_argument("--model", type=str, required=True, help="path to the model we want to analyze")
parser_arg.add_argument("--output_path", type=str, required=True, help="path of the directory to save the results")
parser_arg.add_argument("--z", type=str, required=False, help="path of the latent variables in npy format, if we already have them")
parser_arg.add_argument("--thinning", type=int, required=False, default= 1,  help="""thinning to apply on the latent variables to perform the PCA analysis: if there are too many images,
                        the PCA may take a long time, hence thinning might be needed. For example, thinning = 10 takes one latent variable out of ten for the PCA analysis.""")
parser_arg.add_argument("--num_points", type=int, required=False, default= 20, help="Number of points to generate for the PC traversals")
parser_arg.add_argument('--dimensions','--list', nargs='+', type=int, default= [0, 1, 2], help='<Required> PC dimensions along which we compute the trajectories. If not set, use pc 1, 2, 3', required=False)
parser_arg.add_argument('--generate_structures', action=argparse.BooleanOptionalAction, default= False, help="""If False: run a PCA analysis with PCA traversal. If True,
                            generates the structures corresponding to the latent variables given in z.""")


def concat_and_save(tens, path):
    """
    Concatenate the lsit of tensor along the dimension 0
    :param tens: list of tensor with batch size as dim 0
    :param path: str, path to save the torch tensor
    :return: tensor of concatenated tensors
    """
    concatenated = torch.concat(tens, dim=0)
    np.save(path, concatenated.detach().numpy())
    return concatenated


def compute_traversals(z, dimensions = [0, 1, 2], numpoints=10):
    pca = PCA()
    z_pca = pca.fit_transform(z)
    all_trajectories = []
    all_trajectories_pca = []
    for dim in dimensions:
            traj_pca = graph_traversal(z_pca, dim, numpoints)
            ztraj_pca = pca.inverse_transform(traj_pca)
            nearest_points, _ = get_nearest_point(z, ztraj_pca)
            all_trajectories.append(nearest_points)
            all_trajectories_pca.append(traj_pca)
        
    return all_trajectories, all_trajectories_pca, z_pca, pca



def get_nearest_point(data, query):
    """
    Find closest point in @data to @query
    Return datapoint, index
    """
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind

def graph_traversal(z_pca, dim, numpoints=10):
    z_pca_dim = z_pca[:, int(dim)]
    start = np.percentile(z_pca_dim, 5)
    stop = np.percentile(z_pca_dim, 95)
    traj_pca = np.zeros((numpoints, z_pca.shape[1]))
    traj_pca[:, dim] = np.linspace(start, stop, numpoints)
    return traj_pca


def sample_latent_variables(vae, dataset, batch_size, output_path, num_workers=4):
    """
    Sample all the latent variables of the dataset and save them in a .npy file
    :param vae: object of class VAE corresponding to the model we want to analyze.
    :param dataset: object of class dataset: data on which to analyze the model
    :param batch_size: integer, batch size
    :param output_path: str, path where we want to register the latent variables
    :param num_workers: integer, number of workers
    return 
    """
    for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
        batch_images = batch_images.to(device)
        batch_poses = batch_poses.to(device)
        batch_poses_translation = batch_poses_translation.to(device)
        indexes = indexes.to(device)

        batch_images = batch_images.flatten(start_dim=-2)
        latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images, indexes)
        all_latent_variables.append(latent_variables)


    all_latent_variables = torch.concat(all_latent_variables, dim=0)
    latent_path = os.path.join(output_path, "z.npy")
    np.save(latent_path, all_latent_variables.detach().cpu().numpy())
    return all_latent_variables


def plot_pca(output_path, dim, all_trajectories_pca):
    """
    Function in charge of plotting the PCA of the latent space with the PC traversals
    :param output_path: str, path to the output directory
    :param dim: intger, dimension along which we generate a traversal.
    :param all_trajectories_pca: np.array(num_points, N_pc) coordinates of the points sampled during the traversal.
    """
    os.makedirs(os.path.join(output_path, f"pc{dim}/"), exist_ok=True)
    sns.kdeplot(x=z_pca[:, dim], y=z_pca[:, dim+1], fill=True, clip= (-5, 5))
    print("TRJACTORIES", all_trajectories_pca[dim][:, :])
    plt.scatter(x=all_trajectories_pca[dim][:, dim], y=all_trajectories_pca[dim][:, dim+1], c="red")
    plt.title("PCA of the latent space")
    plt.xlabel(f"PC {dim+1}, variance {pca.explained_variance_ratio_[dim]} ")
    plt.ylabel(f"PC {dim+2}, variance variance {pca.explained_variance_ratio_[dim+1]}")
    plt.savefig(os.path.join(output_path, f"pc{dim}/pca.png"))
    plt.close()

def predict_structures(z_dim):
    """
    Function predicting the structures for a PC traversal along a specific PC.
    :param z_dim: np.array(num_points, latent_dim) coordinates of the sampled structures for the PC traversal
    :param predicted_structures: torch.tensor(num_points, N_residues, 3), predicted structutres for each one of the sampled points of the PC traversal.
    """
    z_dim = torch.tensor(z_dim, dtype=torch.float32, device=device)
    segmentation = vae.sample_segmentation(z_dim.shape[0])
    quaternions_per_domain, translations_per_domain = vae.decode(z_dim)
    rotation_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, segmentation, device)
    translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, segmentation)
    predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                       rotation_per_residue)

    return predicted_structures


def save_structures(predicted_structures, dim, output_path):
    """
    Save a set of structures given in a torch tensor in different pdb files.
    :param predicted_structures: torch.tensor(N_predicted_structures, N_residues, 3), et of structures
    :param dim: integer, dimension along which we sample
    :param output_path: str, path to the directory in which we save the structures.
    """
    for i, pred_struct in enumerate(predicted_structures):
        print("Saving structure", i+1, "from pc", dim)
        base_structure.coord = pred_struct.detach().cpu().numpy()
        base_structure.to_pdb(os.path.join(output_path, f"pc{dim}/structure_z_{i}.pdb"))


def run_pca_analysis(z, dimensions, num_points, output_path):
    """
    Runs a PCA analysis of the latent space and return PC traversals and plots of the PCA of the latent space
    :param z: torch.tensor(N_latent, latent_dim) containing all the latent variables
    :param dimensions: list of integer, list of PC dimensions we want to traverse
    :param num_points: integer, number of points to sample along a PC for the PC traversals
    :param output_path: str, path to the directory where we want to save the PCA resuls
    """
    if z.shape[-1] > 1:
        all_trajectories, all_trajectories_pca, z_pca, pca = compute_traversals(z[::thinning], dimensions=dimensions, numpoints=numpoints)
        sns.set_style("white")
        for dim in dimensions[:-1]:
            plot_pca(output_path, dim, all_trajectories_pca)
            predicted_structures = predict_structures(all_trajectories[dim])
            save_structures(predicted_structures, dim, output_path)

    else:
            os.makedirs(os.path.join(output_path, f"pc0/"), exist_ok=True)
            all_trajectories = graph_traversal(z, 0, numpoints=numpoints)
            z_dim = torch.tensor(all_trajectories, dtype=torch.float32, device=device)
            predicted_structures = predicted_structures(all_trajectories)
            save_structures(predicted_structures, 0, output_path)


def analyze(yaml_setting_path, model_path, output_path, z, thinning=1, dimensions=[0, 1, 2], numpoints=10, generate_structures=False):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment.
    :param model_path: str, path to the model we want to analyze.
    :param structures_path: 
    :return:
    """
    (vae, image_translator, ctf_experiment, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device,
    scheduler, base_structure, lp_mask2d, mask, amortized, path_results, structural_loss_parameters)  = utils.parse_yaml(yaml_setting_path)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()
    if z is None:        
        z = sample_latent_variables(vae, dataset, batch_size, output_path)

    if not generate_structures:
            run_pca_analysis(z, dimensions, num_points, output_path)

    else:
            z = torch.tensor(z, dtype=torch.float32, device=device)
            latent_variables_loader = iter(DataLoader(z, shuffle=False, batch_size=batch_size))
            for batch_num, z in enumerate(latent_variables_loader): 
                predicted_structures = predicted_structures(z) 
                save_structures(predicted_structures)


if __name__ == '__main__':
    args = parser_arg.parse_args()
    output_path = args.output_path
    thinning = args.thinning
    model_path = args.model
    num_points = args.num_points
    path = args.experiment_yaml
    dimensions = args.dimensions
    z = None
    if args.z is not None:
        z = np.load(args.z)
        
    generate_structures = args.generate_structures
    analyze(path, model_path, output_path, z, dimensions=dimensions, generate_structures=generate_structures, thinning=thinning, numpoints=num_points)






