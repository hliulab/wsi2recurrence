from __future__ import print_function, division
import os
import openslide
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F
from PIL import Image
import h5py

from random import randrange

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1,
		contrastive=True,
		transform = None
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi_path = wsi
		self.wsi = None if contrastive else wsi
		self.contrastive = contrastive
		self.transforms = transform
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords'] if not self.contrastive else f['coords'][:100]
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		#self.summary()
		self.h5py_file = None
		
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		if self.contrastive:
			self.wsi = openslide.open_slide(self.wsi_path)
			self.h5py_file = h5py.File(self.file_path, 'r')
			all_coord = self.h5py_file['coords'][:100]
			coord = all_coord[idx]
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

			if self.target_patch_size is not None:
				img = img.resize(self.target_patch_size)
			if self.transforms is not None:
				img = self.transforms(img)
			self.h5py_file.close()
			self.wsi.close()
		else:
			self.h5py_file = h5py.File(self.file_path, 'r')
			coord = self.h5py_file['coords'][idx]
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
			
			if self.target_patch_size is not None:
				img = img.resize(self.target_patch_size)
			img = self.roi_transforms(img).unsqueeze(0)
			self.h5py_file.close()
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]

	def set_result(self,idx,result):
		if 'result' not in self.df.columns:
			self.df.loc[:, 'result'] = 0
		self.df.loc[idx,'result'] = result

	def save_result(self,csv_path):
		self.df.to_csv(csv_path,index=False)