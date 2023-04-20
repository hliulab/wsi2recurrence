import argparse
import os
import multiprocessing as mp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from wsi_core.wsi_utils import isWhitePatch_S, isBlackPatch, isBlackPatch_S, isWhitePatch

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type=str, default="data")
parser.add_argument('--save_dir', type=str,
					default="../tiles_result")  # /data/data_zy/tcga_tiles/breast_ts
parser.add_argument('--patch_size', type=int, default=256,
					help='patch_size')
parser.add_argument('--sample_number', type=int, default=100,
					help='sample_number')

def filter_block(coord, patch_level, patch_size, file_name):
	file_path = file_name + '.svs'
	wsi = openslide.open_slide(file_path)
	
	patch = wsi.read_region((coord[0], coord[1]), patch_level, (patch_size, patch_size)).convert('RGB')
	
	if isWhitePatch_S(np.array(patch), rgbThresh=210, percentage=0.5) or isBlackPatch(np.array(patch)):
		coord = None
	wsi.close()
	return coord

def save(file, patch_save_dir, tiles_save_dir_40x, tiles_save_dir_20x):
	h5py_file = h5py.File(os.path.join(patch_save_dir, file), "r")
	dset = h5py_file['coords']

	attr = {}
	for name, value in dset.attrs.items():
		attr[name] = value
	attr_dict = {'coords': attr}

	length = dset[:].shape[0]

	small_length = args.sample_number if length > args.sample_number else length

	results = dset[:]
	np.random.shuffle(results)
	results = results[:small_length]

	file_path = attr['name'] + '.svs'
	wsi = openslide.open_slide(file_path)
	for i, coord in enumerate(results):
		if attr['patch_size'] == 512:#在40倍放大下等比缩放保存20倍放大的图块
			patch = wsi.read_region((coord[0], coord[1]), attr['patch_level'],
			                        (attr['patch_size'], attr['patch_size'])).convert('RGB')
			patch1 = wsi.read_region((coord[0], coord[1]), attr['patch_level'],
			                         (256, 256)).convert('RGB')
			png_name = os.path.join(tiles_save_dir_40x,
			                        os.path.basename(file_path)[:-4] + '_(' + str(coord[0]) + ',' + str(
				                        coord[1]) + ')_40X.png')
			patch1.save(png_name)
			patch = patch.resize(size=(256, 256))
			png_name = os.path.join(tiles_save_dir_20x,
			                        os.path.basename(file_path)[:-4] + '_(' + str(coord[0]) + ',' + str(
				                        coord[1]) + ').png')
			patch.save(png_name)
		else:#仅用40倍放大数据
			patch = wsi.read_region((coord[0], coord[1]), attr['patch_level'],
			                        (attr['patch_size'], attr['patch_size'])).convert('RGB')
			png_name = os.path.join(tiles_save_dir_40x,
			                        os.path.basename(file_path)[:-4] + '_(' + str(coord[0]) + ',' + str(
				                        coord[1]) + ')_40X.png')
			patch.save(png_name)
	wsi.close()

if __name__ == '__main__':
	args = parser.parse_args()
	
	save_path =  args.save_dir
	patch_save_dir = os.path.join(args.save_dir, 'patches')#filter_block
	tiles_save_dir_20x = os.path.join(save_path, 'tiles_20x')
	tiles_save_dir_40x = os.path.join(save_path, 'tiles_40x')
	if not os.path.exists(tiles_save_dir_20x):
		os.mkdir(tiles_save_dir_20x)
	if not os.path.exists(tiles_save_dir_40x):
		os.mkdir(tiles_save_dir_40x)

	num_workers = mp.cpu_count()
	if num_workers > 32:
		num_workers = 32
	pool = mp.Pool(num_workers)

	iterable = [(file, patch_save_dir, tiles_save_dir_40x, tiles_save_dir_20x) for file in os.listdir(patch_save_dir)]
	pool.starmap(save, iterable)
	pool.close()
