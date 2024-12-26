import os
import sys
path = os.path.abspath("model")
sys.path.append(path)
import torch
import unittest
import sys
sys.path.insert(1, '../model')
from torch.utils.data import DataLoader
from dataset import ImageDataSet
import numpy as np

class TestCsStarEquivalence(unittest.TestCase):
	"""
	This class takes as input a cryoSparc dataset and the same dataset converted to star (RELION) format. 
	It tests if the CTF, images and poses are the same
	"""
	def setUp(self):
		self.star_config = {"file": "test_apoferritin/particles/particles.star"}
		self.cs_file = {"file": "test_apoferritin/J25_split_0_exported.cs"}
		print(ImageDataSet.__init__.__code__.co_varnames)
		self.cs_dataset = ImageDataSet(apix=1.428, side_shape=256, star_cs_file_config=self.cs_file,particles_path="test_apoferritin")
		self.star_dataset = ImageDataSet(apix=1.428, side_shape=256, star_cs_file_config=self.star_config,particles_path="test_apoferritin/particles/")

	def test_images(self):
		torch.manual_seed(0)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = np.max(torch.abs(batch_images_cs - batch_images_star).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_rotations(self):
		torch.manual_seed(1)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=False, num_workers = 4, drop_last=True))
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=False, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = np.max(torch.abs(batch_poses_cs - batch_poses_star).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_translations(self):
		torch.manual_seed(2)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = torch.max(torch.abs(batch_poses_translation_cs - batch_poses_translation_star))
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_fproj(self):
		torch.manual_seed(3)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = torch.max(torch.abs(fproj_cs - fproj_star)).detach().cpu().numpy()
		self.assertAlmostEqual(diff, 0.0, 5)



if __name__ == '__main__':
    unittest.main()



