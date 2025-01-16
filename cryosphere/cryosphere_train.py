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
from cryosphere.model import renderer
from torch.utils.data import DataLoader
from cryosphere.model.utils import low_pass_images
from cryosphere.model.loss import compute_loss, find_range_cutoff_pairs, remove_duplicate_pairs, find_continuous_pairs, calc_dist_by_pair_indices


import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True, help="path to the yaml containing all the parameters for the cryoSPHERE run.")

def train(yaml_setting_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    """
    (vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device, scheduler, 
    base_structure, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters, segmenter) = model.utils.parse_yaml(yaml_setting_path)

    for epoch in range(N_epochs):
        tracking_metrics = {"wandb":experiment_settings["wandb"], "epoch": epoch, "path_results":path_results ,"correlation_loss":[], "kl_prior_latent":[], 
                            "kl_prior_segmentation_mean":[], "kl_prior_segmentation_std":[], "kl_prior_segmentation_proportions":[], "l2_pen":[], "continuity_loss":[], 
                            "clashing_loss":[]}

        data_loader = tqdm(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = experiment_settings["num_workers"], drop_last=True)))
        start_tot = time()
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)
            flattened_batch_images = batch_images.flatten(start_dim=-2)
            batch_translated_images = image_translator.transform(batch_images, batch_poses_translation[:, None, :])
            lp_batch_translated_images = low_pass_images(batch_translated_images, lp_mask2d)
            if amortized:
                latent_variables, latent_mean, latent_std = vae.sample_latent(flattened_batch_images)
            else:
                latent_variables, latent_mean, latent_std = vae.sample_latent(None, indexes)

            segmentation = segmenter.sample_segments(batch_images.shape[0])
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, segmentation, base_structure.coord.shape[0], batch_size, device)
            predicted_structures = model.utils.deform_structure(gmm_repr.mus, translation_per_residue, quaternions_per_domain, segmentation, device)
            posed_predicted_structures = renderer.rotate_structure(predicted_structures, batch_poses)
            predicted_images  = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)/dataset.f_std
            loss = compute_loss(batch_predicted_images, lp_batch_translated_images, None, latent_mean, latent_std, vae, segmenter, experiment_settings, tracking_metrics, 
                structural_loss_parameters= structural_loss_parameters, epoch=epoch, predicted_structures=predicted_structures, device=device)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        model.utils.monitor_training(segmentation, segmenter, tracking_metrics, experiment_settings, vae, optimizer, predicted_images, batch_images)


def cryosphere_train():
    """
    This function serves as an entry point to be called from the command line 
    """
    args = parser_arg.parse_args()
    path = args.experiment_yaml
    train(path)


if __name__ == '__main__':
    cryosphere_train()

