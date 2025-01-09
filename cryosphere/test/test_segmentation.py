import os
import sys
import torch
import unittest
import pytorch3d
import numpy as np
path = os.path.abspath("model")
sys.path.append(path)
sys.path.insert(1, '../model')
from segmentation import Segmentation
from utils import compute_translations_per_residue, deform_structure, parse_yaml



class TestSegmentation(unittest.TestCase):
	"""
	Class for testing the segmentation of different parts.
	"""
	def setUp(self):
		self.N_residues = 1000
		self.residues_chain = np.array(["A" for _ in range(100)] + ["B" for _ in range(500)] + ["C" for _ in range(400)])
		self.residues_indexes = np.array([i for i in range(1000)])
		self.segmentation_config = {"part1":{"N_segm":6, "start_res":0, "end_res":80, "chain":"A"}, "part2":{"N_segm":15, "start_res":300, "end_res":499, "chain":"B"},
									"part3":{"N_segm":10, "start_res":300, "end_res":399, "chain":"C"}}
		self.segmenter = Segmentation(self.segmentation_config, self.residues_indexes, self.residues_chain, tau_segmentation=0.05)


	def test_segmentation(self):
		"""
		Test the actual segmentation function of the Segmentation class
		"""
		batch_size = 10
		segmentation = self.segmenter.sample_segments(batch_size)
		total_moving_residues = 81 + 200 + 100
		total_start = 0
		for part, segm in self.segmentation_config.items():
			segm_mat = segmentation[part]["segmentation"]
			segm_mask = segmentation[part]["mask"]

			self.assertEqual(segm_mat.shape[0], batch_size)
			self.assertEqual(segm_mat.shape[1], segm["end_res"] - segm["start_res"]+1)
			self.assertEqual(segm_mat.shape[2], segm["N_segm"])
			self.assertEqual(np.sum(segm_mask), segm["end_res"] - segm["start_res"]+1)
			self.assertEqual(np.sum(segm_mask[total_start+ segm["start_res"]:total_start+segm["end_res"]+1] == 0), 0)
			total_start += np.sum(self.residues_chain == segm["chain"])



class TestMovingResidues(unittest.TestCase):
	"""
	Class for testing the displacement of the residues based on the segmentation by parts.
	"""
	def setUp(self):
		#torch.manual_seed(1)
		self.device="cpu"
		self.batch_size = 10
		self.N_residues = 1000
		self.residues_chain = np.array(["A" for _ in range(100)] + ["B" for _ in range(500)] + ["C" for _ in range(400)])
		self.residues_indexes = np.array([i for i in range(1000)])
		self.segmentation_config = {"part1":{"N_segm":6, "start_res":0, "end_res":80, "chain":"A"}, "part2":{"N_segm":15, "start_res":300, "end_res":499, "chain":"B"},
									"part3":{"N_segm":10, "start_res":300, "end_res":399, "chain":"C"}}
		self.segmenter = Segmentation(self.segmentation_config, self.residues_indexes, self.residues_chain, tau_segmentation=0.05)
		self.atom_positions = torch.randn((self.batch_size, self.N_residues, 3), dtype=torch.float32, device=self.device)
		self.translation_per_segments = {}
		self.rotation_per_segments = {}
		for part, part_config in self.segmentation_config.items():
			self.translation_per_segments[part] = torch.randn((self.batch_size, self.segmentation_config[part]["N_segm"], 3), dtype=torch.float32, device=self.device)
			self.rotation_per_segments[part] = pytorch3d.transforms.random_quaternions(
										part_config["N_segm"]*self.batch_size, device=self.device).reshape(self.batch_size, part_config["N_segm"], -1)

	def test_translations(self):
		"""
		Test that we translate the right atoms and leave the others untouched.
		"""
		segmentation = self.segmenter.sample_segments(self.batch_size)
		translations_per_residue = compute_translations_per_residue(self.translation_per_segments, segmentation, self.N_residues, self.batch_size, self.device)
		mask = np.zeros(self.N_residues)
		for part, segm in segmentation.items():
			mask += segm["mask"]

		max_trans_non_moving = np.max(torch.abs(translations_per_residue[:, mask == 0]).detach().cpu().numpy())
		self.assertEqual(max_trans_non_moving, 0.0)

	def test_rotations(self):
		"""
		Testing that we rotate the right residues
		"""
		segmentation = self.segmenter.sample_segments(self.batch_size)
		translations_per_residue = compute_translations_per_residue(self.translation_per_segments, segmentation, self.N_residues, self.batch_size, self.device)
		new_atom_positions = deform_structure(self.atom_positions, translations_per_residue, self.rotation_per_segments, segmentation, self.device)
		distances = (new_atom_positions - self.atom_positions)**2
		mask = np.zeros(self.N_residues)
		for part, segm in segmentation.items():
			mask += segm["mask"]


		self.assertEqual(np.max(distances[:, mask==0].detach().cpu().numpy()), 0.0)


	def test_yaml_parsing(self):
		"""
		Tests if the yaml parsing still works well
		"""
		try:
			parse_yaml("test_apoferritin/parameters_package_segmentation.yaml")
			self.assertEqual(0.0, 0.0)
		except:
			self.assertEqual(0.0, 1.0)






