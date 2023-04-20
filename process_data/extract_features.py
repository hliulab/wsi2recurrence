import argparse
import os
import time
from collections import OrderedDict

import h5py
import openslide
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from utils.file_utils import save_hdf5
from utils.utils import collate_features

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'




def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=128, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,contrastive=False,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    #err slide TCGA-E2-A158-11A-02-TS2.97f08d10-8897-4a8b-8c35-7eaaaf0a3049.h5 全白
    #TCGA-E9-A1NF-11A-04-TSD.10c326fa-637d-4cb9-b839-b182c0c240a0.h5
    x, y = dataset[0]
    kwargs = {'num_workers': 12, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    extract_time = 0
    to_cpu_time = 0
    load_time = 0
    io_time = 0

    s_time = time.time()
    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        load_time += time.time() - s_time
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            s_time = time.time()
            features = model(batch)
            extract_time += time.time() - s_time

            s_time = time.time()
            features = features.cpu().numpy()
            to_cpu_time += time.time() - s_time

            s_time = time.time()
            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            io_time += time.time() - s_time
            mode = 'a'

        s_time = time.time()

    print(f"提取特征用时：{extract_time}\n"
          f"数据从GPU转到CPU用时：{to_cpu_time}\n"
          f"数据加载用时：{load_time}\n"
          f"写入h5文件耗时：{io_time}")

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='../tile_results') #patch地址
parser.add_argument('--data_slide_dir', type=str, default="../slides/TCGA-BRCA")   #svs文件所在地
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default='../dataset_csv/sample_data.csv')   #csv文件所在地
parser.add_argument('--feat_dir', type=str, default='../FEATURES')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=256)
parser.add_argument('--tcga_flag', default=False, action='store_true',help='tcga wsi')
parser.add_argument('--model_type', type=str, default="adco_tcga")  # 用以保存特征
parser.add_argument('--data_type', type=str, default="tcga_brca")  # 用以读取h5数据
parser.add_argument('--model_path', type=str, default=r"../MODELS_SAVE/adco_tcga.pth.tar")
args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def load_model(model_name, weight_path):
    check_point = torch.load(weight_path)
    state_dict = check_point["state_dict"]
    # print(state_dict.keys())
    
    if model_name.startswith("mocov2") or model_name.startswith("ADCO_tcga") or model_name.startswith("adco"):
        model = resnet50_baseline()
        new_sd = OrderedDict()
        for k in list(state_dict.keys()):
            # 只要encoder_q
            if k.startswith('module.encoder_q'):
                new_sd[k[len("module.encoder_q."):]] = state_dict[k]
        missing_key = model.load_state_dict(new_sd, strict=False)
        print(missing_key)
        assert set(missing_key.unexpected_keys) == {"fc.0.weight", "fc.0.bias", "fc.2.weight", "fc.2.bias"}
        return model
    elif model_name.startswith("resnet"):
        model = resnet50_baseline(True)
        return model

if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path  # 所有h5文件的路径
    if csv_path is None:
        raise NotImplementedError
    
    
    bags_dataset = Dataset_All_Bags(csv_path)
    # parser.add_argument('--model_type', type=str, default="ADCO_tcga_new") # 用以保存特征
    # /CLAM/ADCO_MODEL
    type_path = args.model_type
    print(f"使用{type_path} 模型提取特征")

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files', type_path), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files', type_path), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir,'pt_files', type_path))

    model = load_model(args.model_type, args.model_path)
    print(model)

    model = model.to(device)

    model.eval()  # 调整为预测模式
    total = len(bags_dataset)

    # 目录名到slide名的映射
    files = os.listdir(args.data_slide_dir)
    txt_file = None
    for f in files:
        if f.split(".")[-1] == "csv":
            txt_file = f
            break
    slide_to_dir = dict()
    if args.tcga_flag:
        assert txt_file is not None
        kidney_info = pd.read_csv(os.path.join(args.data_slide_dir, txt_file))
        slide_to_dir = dict(zip(kidney_info["filename"].values, kidney_info["id"].values))
    err_slide = []
    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx][:-3]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        if not os.path.isfile(h5_file_path):
            print(f"不存在的h5文件{h5_file_path}")
            continue

        slide_file_path = os.path.join(args.data_slide_dir, slide_to_dir[slide_id + ".svs"], slide_id + args.slide_ext) \
        if args.tcga_flag else slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

        print(slide_id)
        if slide_id in err_slide:
            print(slide_id, '出问题')
            continue

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:

            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', type_path, bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            model=model, batch_size=args.batch_size, verbose=1, print_every=5,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        wsi.close()
        features = torch.randn(1)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', type_path, bag_base + '.pt'))
        # if features.shape[0] < 100:
        #     print('extract_feature number < 100 ,please check it {}'.format(bag_base))
