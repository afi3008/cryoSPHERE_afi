


def compute_segmentation_prior(N_residues, N_segments, start_residue, device):
    """
    Computes the segmentation prior if "uniform" is set in the yaml file
    :param N_residues: integer, number of residues
    :param N_segments: integer, number of domains
    :param start_residue: integer, starting residue of the segments.
    :param device: str, device to use
    :return: dict of means and std for each prior over the parameters of the GMM.
    """
    bound_0 = N_residues / N_segments
    segmentation_means_mean = torch.tensor(np.array([bound_0 / 2 + i * bound_0 for i in range(N_segments)]), dtype=torch.float32,
                          device=device)[None, :]

    segmentation_means_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]

    segmentation_stds_mean = torch.tensor(np.ones(N_segments) * bound_0, dtype=torch.float32, device=device)[None, :]

    segmentation_stds_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]

    segmentation_proportions_mean = torch.tensor(np.ones(N_segments) * 0, dtype=torch.float32, device=device)[None, :]

    segmentation_proportions_std = torch.tensor(np.ones(N_segments), dtype=torch.float32, device=device)[None, :]

    segmentation_prior = {}
    segmentation_prior["means"] = {"mean":segmentation_means_mean, "std":segmentation_means_std}
    segmentation_prior["stds"] = {"mean":segmentation_stds_mean, "std":segmentation_stds_std}
    segmentation_prior["proportions"] = {"mean":segmentation_proportions_mean, "std":segmentation_proportions_std}

    return segmentation_prior


class Segmentation():
	def __init__(self, segmentation_config):
		"""
		Creates a GMM used for segmentation purposes.
		:param segmentation_config: dictionnary, containing, for each part of the protein we want to segment, a dictionnary
									 {part_i:{N_segm:x, start_res:res, end_res:res}}
		"""
		self.segmentation_config = segmentation_config
        self.segments_means_means = torch.nn.ParameterDict({})
        self.segments_means_stds = torch.nn.ParameterDict({})
        self.segments_stds_means = torch.nn.ParameterDict({})
        self.segments_stds_stds = torch.nn.ParameterDict({})
        self.segments_proportions_means = torch.nn.ParameterDict({})
        self.segments_proportions_stds = torch.nn.ParameterDict({})

		for part in segmentation_config:
			N_segm = part["N_segm"]
			start_res = part["start_res"]
			end_res = part["end_res"]
			N_res = end_res - start_res
			if "segmentation_start_values" not in part:
				#Initialize the segmentation in a uniform way
				bound_0 = N_res/N_segm
	            self.segments_means_means[part]= torch.nn.Parameter(data=torch.tensor(np.array([start_res + bound_0/2 + i*bound_0 for i in range(N_segments)]), dtype=torch.float32, device=device)[None, :],
	                                                      requires_grad=True)
	            self.segments_means_stds[part] = torch.nn.Parameter(data= torch.tensor(np.ones(N_segments)*10.0, dtype=torch.float32, device=device)[None,:],
	                                                    requires_grad=True)

	            self.segments_stds_means[part] = torch.nn.Parameter(data= torch.tensor(np.ones(N_segments)*bound_0, dtype=torch.float32, device=device)[None,:],
	                                                    requires_grad=True)

	            self.segments_stds_stds[part] = torch.nn.Parameter(
	                data=torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :],
	                requires_grad=True)

	            self.segments_proportions_means[part] = torch.nn.Parameter(
	                data=torch.tensor(np.ones(N_segments) * 0, dtype=torch.float32, device=device)[None, :],
	                requires_grad=True)

	            self.segments_proportions_stds[part] = torch.nn.Parameter(
	                data=torch.tensor(np.ones(N_segments), dtype=torch.float32, device=device)[None, :],
	                requires_grad=True)

	        else:
	        	#Otherwise take the definitions of the segments
	        	self.segments_means_means[part] = part["segmentation_start_values"]["means_means"]
	        	self.segments_means_stds[part] = part["segmentation_start_values"]["means_stds"]
	        	self.segments_stds_means[part] = part["segmentation_start_values"]["stds_means"]
	        	self.segments_stds_stds[part] = part["segmentation_start_values"]["stds_stds"]
	        	self.segments_proportions_means[part] = part["segmentation_start_values"]["proportions_means"]
	        	self.segments_proportions_stds[part] = part["segmentation_start_values"]["proportions_stds"]

	        segmentation_prior = {}
	        if "segmentation_prior" not in part:
	        	#Create a prior with values taken uniformly
	        	for part in segmentation_config:
		        	N_segm = part["N_segm"]
					start_res = part["start_res"]
					end_res = part["end_res"]
					N_res = end_res - start_res
	    			bound_0 = N_res / N_segm
	    			segmentation_means_mean = torch.tensor(np.array([start_res + bound_0 / 2 + i * bound_0 for i in range(N_segments)]), dtype=torch.float32,
	                          device=device)[None, :]
	    			segmentation_means_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]
	    			segmentation_stds_mean = torch.tensor(np.ones(N_segments) * bound_0, dtype=torch.float32, device=device)[None, :]
	    			segmentation_stds_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]
	    			segmentation_proportions_mean = torch.tensor(np.ones(N_segments) * 0, dtype=torch.float32, device=device)[None, :]
	    			segmentation_proportions_std = torch.tensor(np.ones(N_segments), dtype=torch.float32, device=device)[None, :]
					self.segmentation_prior[part]["means"] = {"mean":segmentation_means_mean, "std":segmentation_means_std}
					self.segmentation_prior[part]["stds"] = {"mean":segmentation_stds_mean, "std":segmentation_stds_std}
					self.segmentation_prior[part]["proportions"] = {"mean":segmentation_proportions_mean, "std":segmentation_proportions_std}

			else:
				# Otherwise just take the prior values input by the user.
				for part in segmentation_config:
					for type_value in ["means", "stds", "proportions"]
						self.segmentation_prior[part][type_value] = {"mean":part["segmentation_prior"][f"{type_value}_means"], "std":part["segmentation_prior"][f"{type_value}_stds"]}






