import torch
import wandb
import einops
import logging
import argparse
import model.utils
import numpy as np
from tqdm import tqdm
from time import time
from model import renderer
import torch.nn.functional as F
from model.utils import low_pass_images
from torch.utils.data import DataLoader
from model.loss import compute_loss, find_range_cutoff_pairs, remove_duplicate_pairs, find_continuous_pairs, calc_dist_by_pair_indices


import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True)
parser_arg.add_argument('--debug', type=bool, required=False)


def train(yaml_setting_path, debug_mode):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    (vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, device, scheduler, 
    base_structure, lp_mask2d, mask_images, amortized, path_results, structural_loss_parameters) = model.utils.parse_yaml(yaml_setting_path)

    if experiment_settings["wandb"] == True:
        if experiment_settings["resume_training"]["model"] != "None":
            name = f"experiment_{experiment_settings['name']}_resume"
        else:
            name = f"experiment_{experiment_settings['name']}"
        if not debug_mode:
            wandb.init(
                # Set the project where this run will be logged
                project=experiment_settings['wandb_project'],
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                    name=name,


                # Track hyperparameters and run metadata
                config={
                    "learning_rate": experiment_settings["optimizer"]["learning_rate"],
                    "architecture": "VAE",
                    "dataset": experiment_settings["star_file"],
                    "epochs": experiment_settings["N_epochs"],
                })

    N_residues = base_structure.coord.shape[0]

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

            mask = vae.sample_segmentation(batch_images.shape[0])
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
            predicted_structures = model.utils.deform_structure(gmm_repr.mus, translation_per_residue, quaternions_per_domain, mask, device)
            posed_predicted_structures = renderer.rotate_structure(predicted_structures, batch_poses)
            predicted_images  = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)/dataset.f_std
            loss = compute_loss(batch_predicted_images, lp_batch_translated_images, None, latent_mean, latent_std, vae,
                                experiment_settings["loss"]["loss_weights"], experiment_settings, tracking_metrics, structural_loss_parameters= structural_loss_parameters,
                                 predicted_structures=predicted_structures, device=device)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            break
        if scheduler:
            scheduler.step()

        if not debug_mode:
            model.utils.monitor_training(mask, tracking_metrics, experiment_settings, vae, optimizer, predicted_images, batch_images)


if __name__ == '__main__':
    wandb.login()

    args = parser_arg.parse_args()
    path = args.experiment_yaml
    debug_mode = args.debug
    from torch import autograd
    with autograd.detect_anomaly():
        train(path, debug_mode)

