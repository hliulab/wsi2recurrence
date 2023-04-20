import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, transforms
def default_transforms(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = None
	#transforms.Compose(=[transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
    return t

class FolderDataset(Dataset):
	def __init__(self,data_path,data_path1= None,transform = None):
		self.data_path = data_path
		self.file_list = list(map(lambda x:os.path.join(data_path,x),os.listdir(data_path)))
		if data_path1:
			self.file_list.extend(list(map(lambda x:os.path.join(data_path1,x),os.listdir(data_path1))))
		self.transform = transform if transform is not None else default_transforms()
		self.length = len(self.file_list)
		print('一共 {}张图块'.format(self.length))

	def __len__(self):
		return self.length
	
	def __getitem__(self, item):
		path = self.file_list[item]
		image = Image.open(os.path.join(self.data_path,path))  # 读取到的是RGB， W, H, C
		# plt.imshow(image)
		# plt.show()
		if self.transform is not None:
			image = self.transform(image)
		return image,path
	
