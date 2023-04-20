import os

import h5py
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import Dataset
from models.att_model import CLAM_SB_Reg_NN_Pool

# 返回一个1×1000的numpy数组
def read_h5file(path):
    with h5py.File(path, 'r') as hdf5_file:
        features = hdf5_file['features'][:]
    return features

def normalized(data):
    data = np.array(data)
    scale = MinMaxScaler(feature_range=(0, 1))
    return np.squeeze(scale.fit_transform(data.reshape((-1, 1))))

class FeatureDataset(Dataset):
    def __init__(self, feature_path=None, data_path=None, label='all', is_normalized=False, is_mean=False,
                 is_exp=False, is_max=False):
        super(FeatureDataset, self).__init__()
        self.path = feature_path
        self.data_path = data_path
        self.label = label
        self.is_normalized = is_normalized
        self.is_mean = is_mean
        self.is_max = is_max
        self.is_exp = is_exp
        self.feature = []
        df = pd.read_csv(data_path, sep=',', header=0)

        colums_list = df.columns.tolist()
        for idx, row in df.iterrows():
            y = row['path']
            if label == 'all':
                self.feature.append((os.path.join(self.path, str(y)),list(row[colums_list.index('MKI67'):colums_list.index('TFRC') + 1])))
            else:
                self.feature.append((os.path.join(self.path, str(y)),row[colums_list.index(label)]))

    def __getitem__(self, item) -> tuple:
        feature_h5path, data = self.feature[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            hdf5_file.close()
        features = torch.from_numpy(features)
        if self.is_mean:
            features = torch.mean(features, dim=0)
        elif self.is_max:
            features, _ = torch.max(features, dim=0)
        if self.is_normalized:  # 是否归一化
            data = torch.Tensor(normalized(data))
            #features = torch.Tensor(features)
        elif self.is_exp:
            data = torch.Tensor(data)
            data = torch.exp(data)
        else:
            data = torch.as_tensor(data)
            #data = torch.log10(4 + data)
        return features, data

    def __len__(self) -> int:
        return len(self.feature)

class FeatureDataset_class_mix(Dataset):
    def __init__(self, feature_path=None, data_path=None, label='TCGA Subtype', label_dict=None, is_mean=False,
                 is_max=False,train_sample =False,is_oversample = False,is_undersample = False):
        super(FeatureDataset_class_mix, self).__init__()
        self.path = feature_path
        self.train_sample = train_sample
        self.data_path = data_path
        self.label = label
        self.label_dict = label_dict
        self.num_classes = len(label_dict)
        self.is_mean = is_mean
        self.is_max = is_max
        self.is_oversample = is_oversample
        self.is_undersample = is_undersample
        self.wsi_paths = []
        self.gene_data = []
        self.labels = []
        self.classes_for_all_imgs = []
        df = pd.read_csv(data_path, sep=',', header=0)
        df = self.df_prep(df, self.label_dict, label)
        print(df['label'].value_counts())
        if is_undersample:
            from imblearn.under_sampling import RandomUnderSampler
            x = df.drop(columns='label')
            y = df['label']
            over = RandomUnderSampler(random_state=43)
            X_oversampled, y_oversampled = over.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
            X_oversampled['label'] = y_oversampled
            df = X_oversampled
        elif is_oversample:
            from imblearn.over_sampling import SMOTE
            x = df.drop(columns='label')
            y = df['label']
            over = RandomOverSampler(random_state=43)
            X_oversampled, y_oversampled = over.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
            X_oversampled['label'] = y_oversampled
            df = X_oversampled
            # from imblearn.under_sampling import RandomUnderSampler
            # x = df.drop(columns='label')
            # y = df['label']
            # over = RandomUnderSampler(random_state=43, sampling_strategy=0.5)
            # X_oversampled, y_oversampled = over.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
            # X_oversampled['label'] = y_oversampled
            # df = X_oversampled
        #df['gender'] = df['gender'].map(lambda x:0 if str(x) == 'FEMALE' else 1)
        #print(df['label'].value_counts())
        self.tiles_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.tiles_cls_ids[i] = np.where(df['label'] == i)[0]
        
        colums_list = df.columns.tolist()
        for idx, row in df.iterrows():
            y = row['path']
            self.wsi_paths.append(os.path.join(self.path, str(y)))
            # data_ = list(row[colums_list.index('MKI67'):colums_list.index('TFRC') + 1])
            # data_.extend(row[colums_list.index('age')])
            data_ = row[colums_list.index('age')]/100 if row[colums_list.index('age')] > 1 else row[colums_list.index('age')]
            #print(data_)
            self.gene_data.append(data_)
            self.labels.append(row['label'])
            self.classes_for_all_imgs.append(row['label'])
        # print(self.feature)
    
    @staticmethod
    def df_prep(df, label_dict, label_col):
        if label_col != 'label':
            df['label'] = df[label_col].copy()
        df['label'] = df['label'].apply(lambda x: label_dict[x])
        return df
    
    def __getitem__(self, item) -> tuple:
        feature_h5path = self.wsi_paths[item]
        features_ = self.gene_data[item]
        data = self.labels[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            hdf5_file.close()
        if self.train_sample:
            np.random.shuffle(features)
            features = features[:int(0.8*len(features))]
        features = torch.as_tensor(features)
        #print(features_)
        features_ = torch.as_tensor(features_)
        if self.is_mean:
            features = torch.mean(features, dim=0)
        elif self.is_max:
            features, _ = torch.max(features, dim=0)
        
        return features,features_, data
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def getlabel(self, ids):
        return self.classes_for_all_imgs[ids]


class FeatureDataset_class(Dataset):
    def __init__(self, feature_path=None, data_path=None,label='TCGA Subtype', label_dict=None,is_mean=False,
                  is_max=False,train_sample=False,is_train = False,is_undersample=False,is_oversample=False):
        super(FeatureDataset_class, self).__init__()
        self.path = feature_path
        self.data_path = data_path
        self.label = label
        self.train_sample = train_sample
        self.label_dict = label_dict
        self.num_classes = len(label_dict)
        self.is_mean = is_mean
        self.is_train = is_train
        self.is_oversample = is_oversample
        self.is_undersample = is_undersample
        self.is_max = is_max
        self.feature = []
        self.classes_for_all_imgs = []
        df = pd.read_csv(data_path, sep=',', header=0)
        df = self.df_prep(df, self.label_dict, label)
        print(df['label'].value_counts())
        if is_undersample :
            from imblearn.under_sampling import RandomUnderSampler
            x = df.drop(columns='label')
            y = df['label']
            over = RandomUnderSampler(random_state=43)
            X_oversampled, y_oversampled = over.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
            X_oversampled['label'] = y_oversampled
            df = X_oversampled
        elif is_oversample:
            from imblearn.over_sampling import RandomOverSampler
            x = df.drop(columns='label')
            y = df['label']
            over = RandomOverSampler(random_state=43)
            X_oversampled, y_oversampled = over.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
            X_oversampled['label'] = y_oversampled
            df = X_oversampled
            # from imblearn.under_sampling import RandomUnderSampler
            # x = df.drop(columns='label')
            # y = df['label']
            # over = RandomUnderSampler(random_state=43, sampling_strategy=0.5)
            # X_oversampled, y_oversampled = over.fit_resample(x, y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
            # X_oversampled['label'] = y_oversampled
            # df = X_oversampled
        print(df['label'].value_counts())
        self.tiles_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.tiles_cls_ids[i] = np.where(df['label'] == i)[0]
        colums_list = df.columns.tolist()
        for idx, row in df.iterrows():
            y = row['path']
            self.feature.append((os.path.join(self.path, str(y)), row['label']))
            self.classes_for_all_imgs.append(row['label'])
            
        # print(self.feature)

    @staticmethod
    def df_prep(df, label_dict, label_col):
        if label_col != 'label':
            df['label'] = df[label_col].copy()
        df['label'] = df['label'].apply(lambda x: label_dict[x])
        return df
    
    def __getitem__(self, item) -> tuple:
        feature_h5path, data = self.feature[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            hdf5_file.close()
        if self.train_sample:
            np.random.shuffle(features)
            features = features[:int(0.75*len(features))]
        features = torch.from_numpy(features)
        if self.is_mean:
            features = torch.mean(features, dim=0)
        elif self.is_max:
            features, _ = torch.max(features, dim=0)
      
        return features, data

    def get_labels(self):
        return self.classes_for_all_imgs
    
    def __len__(self) -> int:
        return len(self.feature)

    def getlabel(self, ids):
        return self.classes_for_all_imgs[ids]


