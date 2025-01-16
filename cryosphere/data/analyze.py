import sys
import os
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import torch
from cryosphere.model import utils
import argparse
import starfile
import numpy as np
import seaborn as sns
from time import time
from tqdm import tqdm
from cryosphere.model.polymer import Polymer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True, help="path to the yaml defining the experimentation")
parser_arg.add_argument("--model", type=str, required=True, help="path to the model we want to analyze")
parser_arg.add_argument("--segmenter", type=str, required=True, help="path to the segmenter we want to analyze")
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


def compute_traversals(z, dimensions = [0, 1, 2], num_points=10):
    pca = PCA()
    z_pca = pca.fit_transform(z)
    all_trajectories = []
    all_trajectories_pca = []
    for dim in dimensions:
            traj_pca = graph_traversal(z_pca, dim, num_points)
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

def graph_traversal(z_pca, dim, num_points=10):
    z_pca_dim = z_pca[:, int(dim)]
    start = np.percentile(z_pca_dim, 5)
    stop = np.percentile(z_pca_dim, 95)
    traj_pca = np.zeros((num_points, z_pca.shape[1]))
    traj_pca[:, dim] = np.linspace(start, stop, num_points)
    return traj_pca


def sample_latent_variables(vae, dataset, batch_size, output_path, device, num_workers=4):
    """
    Sample all the latent variables of the dataset and save them in a .npy file
    :param vae: object of class VAE corresponding to the model we want to analyze.
    :param dataset: object of class dataset: data on which to analyze the model
    :param batch_size: integer, batch size
    :param output_path: str, path where we want to register the latent variables
    :param num_workers: integer, number of workers
    return 
    """
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
    all_latent_variables = []
    for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
        batch_images = batch_images.to(device)
        batch_poses = batch_poses.to(device)
        batch_poses_translation = batch_poses_translation.to(device)
        indexes = indexes.to(device)

        batch_images = batch_images.flatten(start_dim=-2)
        latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images, indexes)
        all_latent_variables.append(latent_variables)


    all_latent_variables = torch.concat(all_latent_variables, dim=0).detach().cpu().numpy()
    latent_path = os.path.join(output_path, "z.npy")
    np.save(latent_path, all_latent_variables)
    return all_latent_variables


def plot_pca(output_path, dim, all_trajectories_pca, z_pca, pca):
    """
    Function in charge of plotting the PCA of the latent space with the PC traversals
    :param output_path: str, path to the output directory
    :param dim: intger, dimension along which we generate a traversal.
    :param all_trajectories_pca: np.array(num_points, N_pc) coordinates of the points sampled during the traversal.
    :param z_pca: np.array(N_latent_variables, PCA_latent_dim) all the latent variables in the PCA coordinate system.
    :param pca: scikit-learn PCA object.
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

def predict_structures(vae, z_dim, gmm_repr, segmenter, device):
    """
    Function predicting the structures for a PC traversal along a specific PC.
    :param vae: object of class VAE.
    :param z_dim: np.array(num_points, latent_dim) coordinates of the sampled structures for the PC traversal
    :param gmm_repr: Gaussian representation. Object of class Gaussian.
    :param predicted_structures: torch.tensor(num_points, N_residues, 3), predicted structutres for each one of the sampled points of the PC traversal.
    :param segmenter: object of class Segmentation
    :param device: torch device
    """
    z_dim = torch.tensor(z_dim, dtype=torch.float32, device=device)
    segmentation = segmenter.sample_segments(z_dim.shape[0])
    quaternions_per_domain, translations_per_domain = vae.decode(z_dim)
    translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, segmentation, gmm_repr.mus.shape[0],z_dim.shape[0], device)
    predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue, quaternions_per_domain, segmentation, device)
    return predicted_structures


def save_structure(base_structure, path):
    """
    Save one structure in a PDB file, saved at path
    :param base_structure: object of class Polymer
    :param path: str, path to which we save the PDB
    """
    base_structure.to_pdb(path)


def save_structures_pca(predicted_structures, dim, output_path, base_structure):
    """
    Save a set of structures given in a torch tensor in different pdb files.
    :param predicted_structures: torch.tensor(N_predicted_structures, N_residues, 3), et of structures
    :param dim: integer, dimension along which we sample
    :param output_path: str, path to the directory in which we save the structures.
    :param base_structrue: object of class Polymer
    """
    for i, pred_struct in enumerate(predicted_structures):
        print("Saving structure", i+1, "from pc", dim)
        base_structure.coord = pred_struct.detach().cpu().numpy()
        save_structure(base_structure, os.path.join(output_path, f"pc{dim}/structure_z_{i}.pdb"))

def save_structures(predicted_structures, base_structure, batch_num, output_path, batch_size):
    """
    Save structures in batch, with the correct numbering .
    :param predicted_structures: torch.tensor(N_batch, N_residues, 3) of predicted structures
    :param base_structure: object of class Polymer.
    :param batch_num: integer, batch number
    :param output_path: str, path where we want to save the structures
    """
    for i, pred_struct in enumerate(predicted_structures):
        print("Saving structure", batch_num*batch_size + i)
        base_structure.coord = pred_struct.detach().cpu().numpy()
        base_structure.to_pdb(os.path.join(output_path, f"structure_z_{batch_num*batch_size + i}.pdb"))

def run_pca_analysis(vae, z, dimensions, num_points, output_path, gmm_repr, base_structure, thinning, segmenter, device):
    """
    Runs a PCA analysis of the latent space and return PC traversals and plots of the PCA of the latent space
    :param vae: object of class VAE.
    :param z: torch.tensor(N_latent, latent_dim) containing all the latent variables
    :param dimensions: list of integer, list of PC dimensions we want to traverse
    :param num_points: integer, number of points to sample along a PC for the PC traversals
    :param output_path: str, path to the directory where we want to save the PCA resuls
    :param gmm_repr: object of class Gaussian.
    :param segmenter: object of class segmenter.
    :param device: torch device on which we perform the computations
    """
    if z.shape[-1] > 1:
        all_trajectories, all_trajectories_pca, z_pca, pca = compute_traversals(z[::thinning], dimensions=dimensions, num_points=num_points)
        sns.set_style("white")
        for dim in dimensions:
            plot_pca(output_path, dim, all_trajectories_pca, z_pca, pca)
            predicted_structures = predict_structures(vae, all_trajectories[dim], gmm_repr, segmenter, device)
            save_structures_pca(predicted_structures, dim, output_path, base_structure)

    else:
            os.makedirs(os.path.join(output_path, f"pc0/"), exist_ok=True)
            all_trajectories = graph_traversal(z, 0, num_points=num_points)
            z_dim = torch.tensor(all_trajectories, dtype=torch.float32, device=device)
            predicted_structures = predicted_structures(all_trajectories)
            save_structures_pca(predicted_structures, 0, output_path, base_structure)


def analyze(yaml_setting_path, model_path, segmenter_path, output_path, z, thinning=1, dimensions=[0, 1, 2], num_points=10, generate_structures=False):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment.
    :param model_path: str, path to the model we want to analyze.
    :param segmenter_path: str, path to the segmenter used for the analysis.
    :param structures_path: 
    :return:
    """
    (vae, image_translator, ctf_experiment, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device,
    scheduler, base_structure, lp_mask2d, mask, amortized, path_results, structural_loss_parameters, segmenter)  = utils.parse_yaml(yaml_setting_path, analyze=True)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()
    segmenter.load_state_dict(torch.load(segmenter_path))
    segmenter.eval()
    if not os.path.exists(output_path):
            os.makedirs(output_path)

    if z is None:        
        z = sample_latent_variables(vae, dataset, batch_size, output_path, device)

    if not generate_structures:
            run_pca_analysis(vae, z, dimensions, num_points, output_path, gmm_repr, base_structure, thinning, segmenter, device=device)

    else:
        path_structures = os.path.join(output_path, "predicted_structures")
        if not os.path.exists(path_structures):
            os.makedirs(path_structures)

        z = torch.tensor(z, dtype=torch.float32, device=device)
        latent_variables_loader = iter(DataLoader(z, shuffle=False, batch_size=batch_size))
        for batch_num, z in enumerate(latent_variables_loader): 
            predicted_structures = predict_structures(vae, z, gmm_repr, segmenter, device)
            save_structures(predicted_structures, base_structure, batch_num, path_structures, batch_size)


def analyze_run():
    args = parser_arg.parse_args()
    output_path = args.output_path
    thinning = args.thinning
    model_path = args.model
    num_points = args.num_points
    path = args.experiment_yaml
    dimensions = args.dimensions
    segmenter_path = args.segmenter
    z = None
    if args.z is not None:
        z = np.load(args.z)
        
    generate_structures = args.generate_structures
    analyze(path, model_path, segmenter_path, output_path, z, dimensions=dimensions, generate_structures=generate_structures, thinning=thinning, num_points=num_points)


if __name__ == '__main__':
    analyze_run()






