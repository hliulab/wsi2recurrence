import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import stats
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.feature_dataset import FeatureDataset
from models.att_model import CLAM_SB_Reg_NN_Pool, CLAM_MB_Reg, CLAM_SB_Reg
from utils.config import seed_torch
from utils.getlogger import get_logger
from utils.utils import convlog10

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--feature_path', default=r'../FEATURES', type=str,
					help='path to feature')
parser.add_argument('--exp', type=str, default='None')
parser.add_argument('--save_path', type=str, default='../RESULTS')
parser.add_argument('--csv_path', type=str, default='dataset_csv/gene')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--model_type', type=str, default='clam_mb')
parser.add_argument('--label', type=str, default='all')
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--save_epoch', type=int, default=20)
parser.add_argument("--random_seed", type=int, default=43, help="random seed")
parser.add_argument('--log_output_file', type=str, default="out.log")
parser.add_argument('--tensorboard_log', type=str, default="tensorboard_log")
args = parser.parse_args()

def save_model(model, optimizer, scheduler, epoch, best_loss, model_path, logger):
	checkpoint = {
		'model': model.state_dict(),  # *模型参数
		'optimizer': optimizer.state_dict(),  # *优化器参数
		'scheduler': scheduler.state_dict(),  # *scheduler
		'epoch': epoch,
		'best_score': best_loss
	}
	torch.save(checkpoint, model_path)
	logger.info('epoch: {} save models succeed......'.format(epoch))

def initmodel(model_type,label):
	n_classes = 21
	if label != 'all':
		n_classes = 1
	if model_type == 'clam_mb':
		model = CLAM_MB_Reg(n_classes=n_classes,dropout=True,mlp=args.mlp)
	elif model_type == 'clam_sb':
		model = CLAM_SB_Reg(n_classes=n_classes,dropout=True)
	else:
		model = CLAM_SB_Reg_NN_Pool(n_classes=n_classes,dropout=True)
		
	return model

def main():
	# 文件夹
	columns_list = ['MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'MMP11', 'CTSV', 'ESR1', 'PGR', 'BCL2', 'SCUBE2',
	                'GRB7',
	                'ERBB2', 'GSTM1', 'CD68', 'BAG1', 'ACTB', 'GAPDH', 'RPLP0', 'GUSB', 'TFRC']
	weight_list = np.ones(21)
	#weight_list[14] = 10
	#print(columns_list[14])
	
	savepath = os.path.join(args.save_path, args.exp)
	os.makedirs(savepath, exist_ok=True)
	
	logger = get_logger(os.path.join(savepath, args.log_output_file))
	for k in args.__dict__:
		logger.info(k + ": " + str(args.__dict__[k]))
	
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	
	for k in range(3, args.K + 1):
		# if k == 1 or k ==2:
		# 	continue
		k_path = os.path.join(savepath, str(k))
		print(k_path)
		os.makedirs(k_path, exist_ok=True)
		tb_writer = SummaryWriter(log_dir=os.path.join(k_path, args.tensorboard_log))
		
		loss_weight = torch.from_numpy(weight_list).float().to(device)
		#model = CLAM_SB_Reg_NN_Pool(n_classes=21)#,size_dict = {"small": [2048, 512, 256], "big": [1024, 512, 384]}
		model = initmodel(args.model_type, args.label)
		model.to(device)
		# if torch.cuda.device_count() > 1:
		# 	model = nn.DataParallel(model)
		
		loss_fc = nn.MSELoss(reduction='none')#reduction='sum'
		
		#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
		
		scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		
		train_set = FeatureDataset(feature_path=args.feature_path
								   , data_path=os.path.join(args.csv_path, 'train_dataset_{}.csv'.format(k))
								   , is_normalized=True,label=args.label)
		valid_set = FeatureDataset(feature_path=args.feature_path,
								   data_path=os.path.join(args.csv_path, 'val_dataset_{}.csv'.format(k)),
								   is_normalized=True,label=args.label)
		
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
		val_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
		
		test_set = FeatureDataset(feature_path=args.feature_path
		                         , data_path=os.path.join(args.csv_path, 'test_dataset_{}.csv'.format(k))
		                         , is_normalized=False, label=args.label)
		
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
		
		best_score = 0
		start_epoch = 0
		if os.path.exists(args.resume):
			checkpoint = torch.load(args.resume)
			model.load_state_dict(checkpoint['model'])
			start_epoch = checkpoint['epoch']
			optimizer.load_state_dict(checkpoint['optimizer'])
			scheduler.load_state_dict(checkpoint['scheduler'])
			best_score = checkpoint['best_score']
		task_losses_train = []
		task_losses_val = []
		for epoch in range(start_epoch, args.epochs):
			# 训练
			model.train()
			
			running_train_loss = 0.0
			train_label = []
			train_predicted = []
			# start_time = time.time()
			for data in tqdm(train_loader, desc="train...",ncols=80):
				
				# for data in enumerate(train_loader, 0):
				wsi_feature, label_data = data
				# end_time = time.time()
				# print("耗时: {.2f}秒".format(end_time - start_time))
				optimizer.zero_grad()  # zero the parameter gradients
				wsi_feature = torch.squeeze(wsi_feature)
				
				wsi_feature = wsi_feature.to(device)
				predicted_outputs,_ = model(wsi_feature)  # predict output from the model
				predicted_outputs = predicted_outputs.to(device)
				label_data = label_data.to(device)
				# protein_data = protein_data.squeeze()
				task_loss = loss_fc(predicted_outputs, label_data)  # calculate loss for the predicted output
				weighted_task_loss = torch.mul(loss_weight, task_loss)
				
				train_loss = torch.mean(weighted_task_loss)
				task_loss = np.asarray(task_loss.detach().data.cpu())
				#print(train_loss)
				train_loss.backward()  # back propagate the loss
				optimizer.step()
				
				running_train_loss += train_loss.item()  # track the loss value
				train_label.extend(label_data.cpu().detach().numpy())
				train_predicted.extend(predicted_outputs.cpu().detach().numpy())
				task_losses_train.append(task_loss)
			# Calculate training loss value
			train_loss_value = running_train_loss / len(train_loader)
			
			scheduler.step()
			
			running_valid_loss = 0.0
			val_label = []
			val_predicted = []
			with torch.no_grad():
				model.eval()
				for data in tqdm(val_loader, desc="val...",ncols=80):
					wsi_feature, label_data = data
					wsi_feature = torch.squeeze(wsi_feature)
					wsi_feature = wsi_feature.to(device)
					predicted_outputs,_ = model(wsi_feature)
					
					label_data = label_data.to(device)
					predicted_outputs = predicted_outputs.to(device)
					
					task_loss = loss_fc(predicted_outputs, label_data)
					weighted_task_loss = torch.mul(loss_weight, task_loss)
					val_loss = torch.mean(weighted_task_loss)
					task_loss = np.asarray(task_loss.detach().data.cpu())
					running_valid_loss += val_loss.item()
					
					val_label.extend(label_data.cpu().detach().numpy())
					val_predicted.extend(predicted_outputs.cpu().detach().numpy())
					task_losses_val.append(task_loss)
			# Calculate validation loss value
			val_loss_value = running_valid_loss / len(val_loader)
			
			tb_writer.add_scalars('task_loss_epoch',
			                      dict(zip(list(map(lambda x: x + '_train', columns_list)),
			                               np.asarray(task_losses_train).squeeze().mean(axis=0))),
			                      epoch)
			tb_writer.add_scalars('task_loss_epoch',
			                      dict(zip(list(map(lambda x: x + '_val', columns_list)),
			                               np.asarray(task_losses_val).squeeze().mean(axis=0))),
			                      epoch)
			tb_writer.add_scalars('mse_loss' , {'train': train_loss_value, 'valid': val_loss_value}, epoch)
			
			columns_list = columns_list if args.label == 'all' else [args.label]
			corr_list = np.zeros(len(columns_list))
			corr_list1 = np.zeros(len(columns_list))
			for index in range(len(columns_list)):
				# print(np.array(train_predicted).shape)
				if len(columns_list) == 1:
					corr_train, p_train = stats.pearsonr(convlog10(np.squeeze(np.array(train_predicted))),
					                                     convlog10(np.squeeze(np.array(train_label))))
					corr_val, p_val = stats.pearsonr(convlog10(np.squeeze(np.array(val_predicted))),
					                                 convlog10(np.squeeze(np.array(val_label))))
				else:
					corr_train, p_train = stats.pearsonr(convlog10(np.array(train_predicted)[:, index]),
														 convlog10(np.array(train_label)[:, index]))
					corr_val, p_val = stats.pearsonr(convlog10(np.array(val_predicted)[:, index]),
													 convlog10(np.array(val_label)[:, index]))
				corr_list[index] = corr_train
				corr_list1[index] = corr_val
				tb_writer.add_scalars(columns_list[index] , {'corr_train': corr_train, 'corr_val': corr_val},
									  epoch)
				tb_writer.add_scalars(columns_list[index] +'_p', {'p_train': p_train, 'p_val': p_val}, epoch)
				logger.info('{} corr_train = {},p={}'.format(columns_list[index],corr_train,p_train))
				logger.info('{} corr_val = {},p={}'.format(columns_list[index],corr_val,p_val))
			logger.info("all gene expression average corr is {:.6f}! max corr is {:.6f}!min corr is {:.6f}!".format(corr_list1.mean(),corr_list1.max(),corr_list1.min()))
			if best_score >= val_loss_value or epoch == 0:
				best_score = val_loss_value
				save_model(model, optimizer, scheduler, epoch, val_loss_value,
						   os.path.join(k_path, 'best_checkpoint.pth'), logger)
				logger.info('save best models  succeed......\n')
			
			if epoch % args.save_epoch == 0:
				save_model(model, optimizer, scheduler, epoch, best_score,
						   os.path.join(k_path, 'checkpoint_' + str(epoch) + '.pth'), logger)
			
			logger.info(
				'Epoch:[{}/{}]\t train_mse_loss={:.6f}\t  val_mse_loss={:.6f}\t '.format(epoch, args.epochs, train_loss_value,val_loss_value))
		test(logger,test_loader,loss_fc,os.path.join(k_path, 'best_checkpoint.pth'),device,k_path,columns_list)
		tb_writer.close()

@torch.no_grad()
def test(logger, test_loader, loss_fc, path_checkpoint, device, dir,columns_list):
	if os.path.exists(path_checkpoint):
		checkpoint = torch.load(path_checkpoint)
		model = initmodel(args.model_type, args.label)
		model.to(device)
		model.load_state_dict(checkpoint['model'])
		
		# if torch.cuda.device_count() > 1:
		# 	model = nn.DataParallel(model)
		model.eval()
		running_valid_loss = 0.0
		val_label = []
		val_predicted = []
		with torch.no_grad():
			model.eval()
			for data in tqdm(test_loader, desc="test..."):
				wsi_feature, label_data = data
				wsi_feature = torch.squeeze(wsi_feature)
				wsi_feature = wsi_feature.to(device)
				predicted_outputs,_ = model(wsi_feature)
				# print(wsi_feature.shape)
				label_data = label_data.to(device)
				predicted_outputs = predicted_outputs.to(device)
				
				val_loss = loss_fc(predicted_outputs, label_data)
				val_loss = torch.mean(val_loss)
				running_valid_loss += val_loss.item()
				
				val_label.extend(label_data.cpu().detach().numpy())
				val_predicted.extend(predicted_outputs.cpu().detach().numpy())
		# Calculate validation loss value
		val_loss_value = running_valid_loss / len(test_loader)
		
		# tb_writer.add_scalars('mse_loss', {'train': train_loss_value, 'valid': val_loss_value}, epoch)
		# tb_writer.add_scalars('smoothl1_loss',
		# 					  {'train': running_train_sL1_loss, 'valid': running_valid_sL1_loss}, epoch)
		# tb_writer.add_scalars('huber_loss',
		# 					  {'train': running_train_huber_loss, 'valid': running_valid_huber_loss}, epoch)
		df = pd.DataFrame()
		columns_list = columns_list if args.label == 'all' else [args.label]
		corr_list = np.zeros(len(columns_list))
		for index in range(len(columns_list)):
			if len(columns_list) == 1:
				df[columns_list[index]] = np.array(val_label).squeeze()
				df[columns_list[index] + '_predict'] = np.array(val_predicted).squeeze()
				corr_val, p_val = stats.pearsonr(convlog10(np.squeeze(np.array(val_predicted))),
				                                 convlog10(np.squeeze(np.array(val_label))))
			else:
				df[columns_list[index]] = np.array(val_label).squeeze()[:, index]
				df[columns_list[index] + '_predict'] = np.array(val_predicted).squeeze()[:, index]
				corr_val, p_val = stats.pearsonr(convlog10(np.array(val_predicted)[:, index]),
				                                 convlog10(np.array(val_label)[:, index]))
			logger.info("gene name={} r={} ,p={}".format(columns_list[index], corr_val, p_val))
			corr_list[index] = corr_val
		logger.info("all gene expression average corr is {:.6f}! max corr is {:.6f}!min corr is {:.6f}!".format(
			corr_list.mean(), corr_list.max(), corr_list.min()))
		logger.info('test_mse_loss={:.6f}\t  '.format(val_loss_value))
		df.to_csv(os.path.join(dir, 'att_test_pred.csv'))
	else:
		logger.info('no train file! please train')

if __name__ == '__main__':
	# # 设置随机种子，为了模型效果可复现
	seed_torch(args.random_seed)
	main()