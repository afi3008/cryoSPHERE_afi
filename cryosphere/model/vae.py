import torch
import numpy as np


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, segmentation_start_values, N_segments=6, N_residues=1006, tau_segmentation=0.05,
                 latent_dim = None, amortized=True, N_images=None):
        """
        VAE class. This defines all the parameters needed and perform the reparametrization trick.
        :param encoder: object of type MLP, with type "encoder"
        :param decoder: object of type MLP, with type "decoder"
        :param device: torch device on which we want to perform the computations.
        :param segmentation_start_values: "uniform" for starting uniformly, otherwise dictionnary containing the mean, std for each of the GMM segmentation parameters (mean, std, proportion)
        :param N_segments: integer, number of segments
        :param N_residues: integer, total number of residues in the base structure
        :param tau_segmentation: float, the probabilities of belonging to each segment is annealed by 1/tau
        :param latent_dim: integer, latent dimension
        :param amortized: bool, whether to perform amortized inference or not
        :param N_images: integer, number of images in the dataset
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.N_segments= N_segments
        self.N_residues = N_residues
        self.tau_segmentation = tau_segmentation
        self.latent_dim = latent_dim
        self.N_images = N_images
        self.amortized = amortized

        self.residues = torch.arange(0, self.N_residues, 1, dtype=torch.float32, device=device)[:, None]

        if segmentation_start_values.get("type") == "uniform":
            bound_0 = self.N_residues/N_segments
            self.segmentation_means_mean = torch.nn.Parameter(data=torch.tensor(np.array([bound_0/2 + i*bound_0 for i in range(N_segments)]), dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)
            self.segmentation_means_std = torch.nn.Parameter(data= torch.tensor(np.ones(N_segments)*10.0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)
            self.segmentation_std_mean = torch.nn.Parameter(data= torch.tensor(np.ones(N_segments)*bound_0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)

            self.segmentation_std_std = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.segmentation_proportions_mean = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_segments) * 0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.segmentation_proportions_std = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_segments), dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

        else:
            self.segmentation_means_mean = torch.nn.Parameter(data=torch.tensor(segmentation_start_values["clusters_means"]["mean"], dtype=torch.float32,device=device)[None, :],
                                                    requires_grad=True)

            self.segmentation_means_std = torch.nn.Parameter(data=torch.tensor(segmentation_start_values["clusters_means"]["std"], dtype=torch.float32, device=device)[None, :],
                                                  requires_grad=True)

            self.segmentation_std_mean = torch.nn.Parameter(data=torch.tensor(segmentation_start_values["clusters_stds"]["mean"], dtype=torch.float32, device=device)[None, :],
                                                  requires_grad=True)

            self.segmentation_std_std = torch.nn.Parameter(data=torch.tensor(segmentation_start_values["clusters_stds"]["std"], dtype=torch.float32, device=device)[None, :],
                                                  requires_grad=True)

            self.segmentation_proportions_mean = torch.nn.Parameter(torch.tensor(segmentation_start_values["clusters_proportions"]["mean"], dtype=torch.float32, device=device)[None, :],
                                                          requires_grad=True)

            self.segmentation_proportions_std = torch.nn.Parameter(torch.tensor(segmentation_start_values["clusters_proportions"]["std"], dtype=torch.float32, device=device)[None, :],
                               requires_grad=True)



        self.segmentation_parameters = {"means":{"mean":self.segmentation_means_mean, "std":self.segmentation_means_std},
                               "stds":{"mean":self.segmentation_std_mean, "std":self.segmentation_std_std},
                               "proportions":{"mean":self.segmentation_proportions_mean, "std":self.segmentation_proportions_std}}
        self.elu = torch.nn.ELU()
        if not amortized:
            assert N_images, "If using a non amortized version of the code, the number of images must be specified"
            self.latent_variables_mean = torch.nn.Parameter(torch.zeros(N_images, self.latent_dim, dtype=torch.float32, device=device), requires_grad=True)
            self.latent_variables_std = torch.nn.Parameter(torch.ones(N_images, self.latent_dim, dtype=torch.float32, device=device)*0.5, requires_grad=False)

    def sample_segmentation(self, N_batch):
        """
        Samples a segmantion
        :param N_batch: integer: size of the batch.
        :return: torch.tensor(N_batch, N_residues, N_segments) values of the segmentation
        """
        cluster_proportions = torch.randn((N_batch, self.N_segments),
                                          device=self.device) * self.segmentation_proportions_std+ self.segmentation_proportions_mean
        cluster_means = torch.randn((N_batch, self.N_segments), device=self.device) * self.segmentation_means_std+ self.segmentation_means_mean
        cluster_std = self.elu(torch.randn((N_batch, self.N_segments), device=self.device)*self.segmentation_std_std + self.segmentation_std_mean) + 1
        proportions = torch.softmax(cluster_proportions, dim=-1)
        log_num = -0.5*(self.residues[None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
              torch.log(proportions[:, None, :])

        segmentation = torch.softmax(log_num / self.tau_segmentation, dim=-1)
        return segmentation

    def sample_latent(self, images, indexes=None):
        """
        Samples latent variables given an image or given an image index if non amortized inference is performed. Apply the reparameterization trick
        :param images: torch.tensor(N_batch, N_pix**2) of input images
        :param indexes: torch.tensor(N_batch, dtype=torch.int) the indexes of images in the batch
        :return: torch.tensor(N_batch, latent_dim) sampled latent variables,
                torch.tensor(N_batch, latent_dim) latent_mean,
                torch.tensor(N_batch, latent_dim) latent std

        """
        if not self.amortized:
            assert indexes is not None, "If using a non-amortized version of the code, the indexes of the images must be provided"
            latent_variables = torch.randn_like(self.latent_variables_mean[indexes, :], dtype=torch.float32, device=self.device)*self.latent_variables_std[indexes, :] + self.latent_variables_mean[indexes, :]
            return latent_variables, self.latent_variables_mean[indexes, :], self.latent_variables_std[indexes, :] 
        else:
            latent_mean, latent_std = self.encoder(images)
            latent_variables = latent_mean + torch.randn_like(latent_mean, dtype=torch.float32, device=self.device)\
                                *latent_std

            return latent_variables, latent_mean, latent_std


    def decode(self, latent_variables):
        """
        Decode the latent variables into a rigid body transformation (rotation and translation) per segment.
        :param latent_variables: torch.tensor(N_batch, latent_dim)
        :return: torch.tensor(N_batch, N_segments, 4) quaternions, torch.tensor(N_batch, N_segments, 3) translations.
        """
        N_batch = latent_variables.shape[0]
        transformations = self.decoder(latent_variables)
        transformations_per_segments = torch.reshape(transformations, (N_batch, self.N_segments, 6))
        ones = torch.ones(size=(N_batch, self.N_segments, 1), device=self.device)
        quaternions_per_segments = torch.concat([ones, transformations_per_segments[:, :, 3:]], dim=-1)
        translations_per_segments= transformations_per_segments[:, :, :3]
        return quaternions_per_segments, translations_per_segments



