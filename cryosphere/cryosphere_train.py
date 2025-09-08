import sys
import torch
import wandb
import logging
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from cryosphere import model
import torch.nn.functional as F
import torch.multiprocessing as mp
from cryosphere.model import renderer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from cryosphere.model.utils import low_pass_images, ddp_setup
from torch.distributed import destroy_process_group
from cryosphere.model.loss import compute_loss, find_range_cutoff_pairs, remove_duplicate_pairs, find_continuous_pairs, calc_dist_by_pair_indices


import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True, help="path to the yaml containing all the parameters for the cryoSPHERE run.")

def train(rank, world_size, yaml_setting_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    """
    ddp_setup(rank, world_size)
    (vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device, scheduler, 
    base_structures, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters, segmenters) = model.utils.parse_yaml(yaml_setting_path, rank)
    start_training(vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, scheduler, 
    base_structures, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters, segmenters, rank)
    destroy_process_group()

def start_training(vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, scheduler, 
    base_structures, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters, segmenters, gpu_id):
    vae = DDP(vae, device_ids=[gpu_id])
    for pdb_name in segmenters:
        segmenters[pdb_name] = DDP(segmenters[pdb_name], device_ids=[gpu_id])
    for epoch in range(N_epochs):
        tracking_metrics = {"wandb":experiment_settings["wandb"], "epoch": epoch, "path_results":path_results ,"correlation_loss":[], "kl_prior_latent":[], 
                            "kl_prior_segmentation_mean":[], "kl_prior_segmentation_std":[], "kl_prior_segmentation_proportions":[], "l2_pen":[], "continuity_loss":[], 
                            "clashing_loss":[]}

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = experiment_settings["num_workers"], drop_last=True, sampler=DistributedSampler(dataset, drop_last=True))
        start_tot = time()
        data_loader.sampler.set_epoch(epoch) 
        data_loader = tqdm(iter(data_loader))
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
            batch_images = batch_images.to(gpu_id)
            batch_poses = batch_poses.to(gpu_id)
            batch_poses_translation = batch_poses_translation.to(gpu_id)
            indexes = indexes.to(gpu_id)
            flattened_batch_images = batch_images.flatten(start_dim=-2)
            batch_translated_images = image_translator.transform(batch_images, batch_poses_translation[:, None, :])
            lp_batch_translated_images = low_pass_images(batch_translated_images, lp_mask2d)
            if amortized:
                latent_variables, latent_mean, latent_std = vae.module.sample_latent(flattened_batch_images)
            else:
                latent_variables, latent_mean, latent_std = vae.module.sample_latent(None, indexes)

            # classifier!!!
            
            segmentation = {}
            for pdb_name, seg in segmenters.items():
                segmentation[pdb_name] = seg.module.sample_segments(batch_images.shape[0])
            #segmentation = segmenter.module.sample_segments(batch_images.shape[0])
            quaternions_per_domain, translations_per_domain = vae.module.decode(latent_variables)
            quaternions_per_pdb = {}
            translations_per_pdb = {}
            for pdb_name, parts in pdb_to_parts_mapping.items():  # some dict mapping pdb_name -> list of parts
                quaternions_per_pdb[pdb_name] = {part: quaternions_per_domain[part] for part in parts}
                translations_per_pdb[pdb_name] = {part: translations_per_domain[part] for part in parts}
            N_residues = {pdb_name: len(residues_indexes[pdb_name]) for pdb_name in segmenters.keys()}
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, segmentation, N_residues, batch_size, gpu_id)
            #predicted_structures = model.utils.deform_structure(gmm_repr.mus, translation_per_residue, quaternions_per_domain, segmentation, gpu_id)
            predicted_structures = {}
            posed_predicted_structures = {}
            predicted_images = {}
            for pdb_name, atom_positions in base_structures.items():
                predicted_structures[pdb_name] = model.utils.deform_structure(gmm_repr.mus[pdb_name], translation_per_residue[pdb_name], quaternions_per_domain[pdb_name], segmentation[pdb_name], gpu_id)
                posed_predicted_structures[pdb_name] = renderer.rotate_structure(predicted_structures[pdb_name], batch_poses)
                predicted_images[pdb_name] = renderer.project(posed_predicted_structures[pdb_name], gmm_repr.sigmas[pdb_name], gmm_repr.amplitudes[pdb_name], grid)
                #posed_predicted_structures = renderer.rotate_structure(predicted_structures, batch_poses)
                #predicted_images  = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
                batch_predicted_images = renderer.apply_ctf(predicted_images[pdb_name], ctf, indexes)/dataset.f_std
                loss[pdb_name] = compute_loss(batch_predicted_images[pdb_name], lp_batch_translated_images, None, latent_mean, latent_std, vae.module, segmenter.module, experiment_settings, tracking_metrics, 
                structural_loss_parameters= structural_loss_parameters, epoch=epoch, predicted_structures=predicted_structures[pdb_name], device=gpu_id)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        model.utils.monitor_training(segmentation, segmenters, tracking_metrics, experiment_settings, vae.module, optimizer, predicted_images, batch_images, gpu_id)


def cryosphere_train():
    """
    This function serves as an entry point to be called from the command line 
    """
    args = parser_arg.parse_args()
    path = args.experiment_yaml

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, path), nprocs=world_size)


if __name__ == '__main__':
    cryosphere_train()

