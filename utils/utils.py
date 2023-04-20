import collections
import math
import os
from itertools import islice, cycle
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch import optim
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convlog10(a):
    
    return 10 ** a -1

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor(np.array([item[1] for item in batch]))
    return [img, label]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 0, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset),
                        collate_fn=collate_MIL, **kwargs)
    return loader


def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    """
        return either the validation loader or training loader
    """
    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    print(kwargs)
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)),
                                    collate_fn=collate_MIL, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=1, collate_fn=collate_MIL, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, collate_fn=collate_MIL, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate_MIL,
                            **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


# def make_weights_for_balanced_classes_split(dataset):
#     N = float(len(dataset))
#     weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
#     weight = [0] * int(N)
#     for idx in range(len(dataset)):
#         y = dataset.getlabel(idx)
#         weight[idx] = weight_per_class[y]
#
#     return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def cal_roc_auc_score(labels,prob,n_classes):
	if n_classes == 2:
		return roc_auc_score(labels, prob[:, 1])
	else:
		return roc_auc_score(labels, prob, multi_class='ovr')

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))
	weight_per_class = [N / len(dataset.tiles_cls_ids[c]) for c in range(len(dataset.tiles_cls_ids))]
	print(weight_per_class)
	weight = [0] * int(N)
	for idx in range(len(dataset)):
		y = dataset.getlabel(idx)
		weight[idx] = weight_per_class[y]

	return torch.DoubleTensor(weight)

def plot_auc(label_dict,test_labelAll,test_probAll,dir):
	n_classes = len(label_dict)
	new_d = {v: k for k, v in label_dict.items()}
	y_label = label_binarize(test_labelAll, classes=[i for i in range(n_classes)])
	print(n_classes)
	print(y_label.shape)
	if n_classes >2 :
		# 计算每一类的ROC
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(y_label[:, i], np.array(test_probAll)[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])
		
		# micro（方法二）
		fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), np.array(test_probAll).ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
		
		# macro（方法一）
		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes):
			mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		
		mean_tpr /= n_classes
		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
		
		# Plot all ROC curves
		lw = 2
		plt.figure(figsize=(10, 8))
		plt.plot(fpr["micro"], tpr["micro"],
		         label='micro-average ROC curve (area = {0:0.2f})'
		               ''.format(roc_auc["micro"]),
		         color='deeppink', linestyle=':', linewidth=4)
		
		plt.plot(fpr["macro"], tpr["macro"],
		         label='macro-average ROC curve (area = {0:0.2f})'
		               ''.format(roc_auc["macro"]),
		         color='navy', linestyle=':', linewidth=4)
		
		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'peru', 'pink'])
		for i, color in zip(range(n_classes), colors):
			plt.plot(fpr[i], tpr[i], color=color, lw=lw,
			         label='ROC curve of class {0} (area = {1:0.2f})'
			               ''.format(new_d.get(i), roc_auc[i]))
	else:
		fpr, tpr, thresholds = roc_curve(test_labelAll, test_probAll[:,1])
		roc_auc = auc(fpr, tpr)
		# print(fpr)
		# print(tpr)
		# print(roc_auc)
		lw = 2
		plt.figure(figsize=(10, 10))
		plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('multi-class ROC')
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(dir, 'roc.png'))
	plt.show()

def plot_pr(label_dict,test_labelAll,test_probAll,dir):
	from sklearn.metrics import precision_recall_curve
	plt.figure("P-R Curve")
	plt.title('Precision/Recall Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	#y_true为样本实际的类别，y_scores为样本为正例的概率
	precision, recall, thresholds = precision_recall_curve(test_labelAll, test_probAll[:,1])
	#print(precision)
	#print(recall)
	#print(thresholds)
	plt.plot(recall,precision,color='darkorange')
	plt.savefig(os.path.join(dir, 'pr.png'))
	plt.show()

import numpy as np
import pandas as pd
from sklearn.utils import shuffle as reset


def train_test_split(data, test_size=0.3, shuffle=True, random_state=None):
	'''Split DataFrame into random train and test subsets

	Parameters
	----------
	data : pandas dataframe, need to split dataset.

	test_size : float
		If float, should be between 0.0 and 1.0 and represent the
		proportion of the dataset to include in the train split.

	random_state : int, RandomState instance or None, optional (default=None)
		If int, random_state is the seed used by the random number generator;
		If RandomState instance, random_state is the random number generator;
		If None, the random number generator is the RandomState instance used
		by `np.random`.

	shuffle : boolean, optional (default=None)
		Whether or not to shuffle the data before splitting. If shuffle=False
		then stratify must be None.
	'''
	
	if shuffle:
		data = reset(data, random_state=random_state)
	
	train = data[int(len(data) * test_size):].reset_index(drop=True)
	test = data[:int(len(data) * test_size)].reset_index(drop=True)
	
	return train, test

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_test_split1(data, test_size=0.2,val_size = 0.2, shuffle=True, random_state=None):
	if shuffle:
		data = reset(data, random_state=random_state)

	train = data[int(len(data) * (test_size+val_size)):].reset_index(drop=True)
	val = data[int(len(data) * test_size):int(len(data) * (test_size+val_size)):].reset_index(drop=True)
	test = data[:int(len(data) * test_size)].reset_index(drop=True)

	return train, val, test

def plot_confusion_matrix(cm, classes):
	sns.heatmap(cm, annot=True, fmt='g', xticklabels=classes, yticklabels=classes)
	plt.ylabel('Real label')
	plt.xlabel('Prediction')
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	df =pd.read_csv(r'E:\project\bioProject_breast\precess_data1\czsy_test.csv')
	df['age'] = df['年龄'].map(lambda x:int(x[:-1])/100)
	print(df['age'])
	df.to_csv(r'E:\project\bioProject_breast\precess_data1\czsy_test.csv')