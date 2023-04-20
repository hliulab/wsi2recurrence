import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import stats
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from datasets.feature_dataset import  FeatureDataset_class
from models.att_model import CLAM_SB_Reg_NN_Pool, CLAM_MB_Reg, CLAM_SB_Class
from utils import utils
from utils.config import seed_torch
from utils.getlogger import get_logger
from utils.utils import  make_weights_for_balanced_classes_split

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--feature_path', default=r'../FEATURES',
                    type=str,help='path to feature')
parser.add_argument('--exp', type=str, default='None')
parser.add_argument('--save_path', type=str, default='../RESULTS')
parser.add_argument('--csv_path', type=str, default='dataset_csv/tumor')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--l1', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--optimizer_step', type=int, default=256)
parser.add_argument('--loss', type=str, choices=['normal', 'focal'], default='normal')
parser.add_argument('--model_type', type=str, default='clam_sb')
parser.add_argument('--label', type=str, default='tumor')
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--save_epoch', type=int, default=9)
parser.add_argument("--random_seed", type=int, default=1000, help="random seed")
parser.add_argument('--log_output_file', type=str, default="out.log")
parser.add_argument('--tensorboard_log', type=str, default="tensorboard_log")
args = parser.parse_args()

label_dict = {
	0: 0
	, 1: 1
}


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


def initmodel(model_type, label):
	n_classes = len(label_dict)
	
	if model_type == 'clam_mb':
		model = CLAM_MB_Reg(n_classes=n_classes, dropout=True, mlp=args.mlp)
	elif model_type == 'clam_sb':
		model = CLAM_SB_Class(n_tasks=1, n_classes=len(label_dict), dropout=True,mlp=0)
	else:
		model = CLAM_SB_Reg_NN_Pool(n_classes=n_classes, dropout=True)
	
	return model


def main():
	# 文件夹
	
	savepath = os.path.join(args.save_path, args.exp)
	os.makedirs(savepath, exist_ok=True)
	
	logger = get_logger(os.path.join(savepath, args.log_output_file))
	for k in args.__dict__:
		logger.info(k + ": " + str(args.__dict__[k]))
	
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	test_df = pd.DataFrame()
	for k in range(1, args.K + 1):
		k_path = os.path.join(savepath, str(k))
		os.makedirs(k_path, exist_ok=True)
		tb_writer = SummaryWriter(log_dir=os.path.join(k_path, args.tensorboard_log))
		
		# model = CLAM_SB_Reg_NN_Pool(n_classes=21)#,size_dict = {"small": [2048, 512, 256], "big": [1024, 512, 384]}
		model = initmodel(args.model_type, args.label)
		model.to(device)
		# if torch.cuda.device_count() > 1:
		# 	model = nn.DataParallel(model)
		print(device)
		
		train_set = FeatureDataset_class(feature_path=args.feature_path
		                                 , data_path=os.path.join(args.csv_path, 'train_dataset_{}.csv'.format(k))
		                                 , label=args.label, label_dict=label_dict,is_undersample=True)
		valid_set = FeatureDataset_class(feature_path=args.feature_path,
		                                 data_path=os.path.join(args.csv_path, 'val_dataset_{}.csv'.format(k)),
		                                 label=args.label, label_dict=label_dict)
		test_set = FeatureDataset_class(feature_path=args.feature_path,
		                                 data_path=os.path.join(args.csv_path, 'test_dataset_{}.csv'.format(k)),
		                                 label=args.label, label_dict=label_dict)
		
		N = float(len(train_set))
		weight_per_class = [N / len(train_set.tiles_cls_ids[c]) for c in range(len(train_set.tiles_cls_ids))]
		weight_per_class = torch.as_tensor(weight_per_class).to(device)
		print(weight_per_class)
		weights = make_weights_for_balanced_classes_split(train_set)
		#print(weights)
		#train_loader = DataLoader(train_set, batch_size=args.batch_size,sampler=WeightedRandomSampler(weights, len(weights)))
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
		# train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
		val_loader = DataLoader(valid_set, batch_size=args.batch_size)
		test_loader = DataLoader(test_set, batch_size=args.batch_size)

		# loss_fc = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1000.0]).to(device))  #weight=torch.FloatTensor([1.0, 4.805]).to(device) reduction='sum'
		
		if args.loss == 'focal':
			loss_fc = focal_loss(alpha=[0.5, 0.5], num_classes=len(label_dict))
		else:
			loss_fc = nn.CrossEntropyLoss(reduction='none',weight=weight_per_class.to(device))
		
		# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-2)
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
		
		scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		
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
			train_sum_num = 0
			val_sum_num = 0
			train_label = []
			train_predicted = []
			# start_time = time.time()
			for i, data in enumerate(tqdm(train_loader, desc="train...",ncols=80)):
				# for data in enumerate(train_loader, 0):
				wsi_feature, label_data = data
				# end_time = time.time()
				# print("耗时: {.2f}秒".format(end_time - start_time))
				
				wsi_feature = torch.squeeze(wsi_feature)
				wsi_feature = wsi_feature.to(device)
				predicted_outputs, _ = model(wsi_feature)  # predict output from the model
				predicted_outputs = predicted_outputs.to(device)
				label_data = label_data.to(device)
				# protein_data = protein_data.squeeze()
				task_loss = loss_fc(predicted_outputs, label_data)  # calculate loss for the predicted output
				
				regularization_loss = 0
				for param in model.parameters():
					regularization_loss += torch.sum(abs(param))
				task_loss = task_loss + args.l1 * regularization_loss
				prob_softmax = F.softmax(predicted_outputs, dim=1)
				pred = torch.max(predicted_outputs, dim=1)[1]
				iter_acc = torch.eq(pred, label_data.to(device)).sum()
				train_sum_num += iter_acc
				train_loss = torch.mean(task_loss)
				task_loss = np.asarray(task_loss.detach().data.cpu())
				# print(train_loss)
				train_loss.backward()  # back propagate the loss
				if i % args.optimizer_step == 0 or i == len(train_loader) - 1:
					optimizer.step()
					optimizer.zero_grad()  # zero the parameter gradients
				running_train_loss += train_loss.item()  # track the loss value
				train_label.extend(label_data.cpu().detach().numpy())
				train_predicted.extend(prob_softmax.cpu().detach().numpy())
				task_losses_train.append(task_loss)
			# Calculate training loss value
			train_loss_value = running_train_loss / len(train_loader)
			acc_train = train_sum_num / len(train_loader)
			print(np.array(train_label).shape)
			print(np.array(train_predicted).shape)
			auc_train = utils.cal_roc_auc_score(np.array(train_label), np.array(train_predicted), len(label_dict))
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
					predicted_outputs, _ = model(wsi_feature)
					
					label_data = label_data.to(device)
					
					task_loss = loss_fc(predicted_outputs.to(device), label_data)
					prob_softmax = F.softmax(predicted_outputs, dim=1)
					pred = torch.max(predicted_outputs, dim=1)[1]
					iter_acc = torch.eq(pred, label_data.to(device)).sum()
					val_sum_num += iter_acc
					
					val_loss = torch.mean(task_loss)
					task_loss = np.asarray(task_loss.detach().data.cpu())
					running_valid_loss += val_loss.item()
					
					val_label.extend(label_data.cpu().detach().numpy())
					val_predicted.extend(prob_softmax.cpu().detach().numpy())
					task_losses_val.append(task_loss)
			# Calculate validation loss value
			val_loss_value = running_valid_loss / len(val_loader)
			acc_val = val_sum_num / len(val_loader)
			auc_val = utils.cal_roc_auc_score(np.array(val_label), np.array(val_predicted), len(label_dict))
			tb_writer.add_scalars('loss', {'train': train_loss_value, 'valid': val_loss_value}, epoch)
			tb_writer.add_scalars('acc', {'train': acc_train, 'valid': acc_val}, epoch)
			tb_writer.add_scalars('auc', {'train': auc_train, 'valid': auc_val}, epoch)
			
			if best_score >= val_loss_value or epoch == 0:
				best_score = val_loss_value
				save_model(model, optimizer, scheduler, epoch, val_loss_value,
				           os.path.join(k_path, 'best_checkpoint.pth'), logger)
				logger.info('save best models  succeed......\n')
			
			if epoch % args.save_epoch == 0:
				save_model(model, optimizer, scheduler, epoch, best_score,
				           os.path.join(k_path, 'checkpoint_' + str(epoch) + '.pth'), logger)
			
			logger.info(
				'Epoch:[{}/{}]\t train_loss={:.6f}\t  val_loss={:.6f}\t train_acc={} \t val_acc={} \t train_auc={} \t val_auc={} \t'
					.format(epoch, args.epochs, train_loss_value, val_loss_value, acc_train, acc_val, auc_train,
				            auc_val))
		test_acc,test_auc,test_f1 = test(logger, test_loader, loss_fc, os.path.join(k_path, 'best_checkpoint.pth'), device, k_path)
		series = pd.Series({'acc': test_acc, 'auc': test_auc, 'f1': test_f1}, name=k)
		test_df = test_df.append(series)
		logger.info('K fold {}/{} : end training!'.format(k, args.K))
	logger.info(test_df)


@torch.no_grad()
def test(logger, test_loader, loss_func, path_checkpoint, device, dir):
	n_classes = len(label_dict)
	if os.path.exists(path_checkpoint):
		checkpoint = torch.load(path_checkpoint)
		model = initmodel(args.model_type, args.label)
		model.to(device)
		model.load_state_dict(checkpoint['model'])
		
		# if torch.cuda.device_count() > 1:
		# 	model = nn.DataParallel(model)
		model.eval()
		# 用于存储预测正确的样本个数
		sum_num = torch.zeros(1).to(device)
		mean_loss = torch.zeros(1).to(device)
		# 统计验证集样本总数目
		num_samples = len(test_loader.dataset)
		test_probAll = []
		test_labelAll = []
		prob_labelAll = []
		# 打印验证进度
		data_loader = tqdm(test_loader, desc="Test...",ncols=80)
		for iteration, data in enumerate(data_loader):
			wsi_feature, labels = data
			wsi_feature = torch.squeeze(wsi_feature)
			wsi_feature = wsi_feature.to(device)
			pred, _ = model(wsi_feature)
			
			loss = loss_func(pred, labels.to(device))
			
			prob_softmax = F.softmax(pred, dim=1)
			pred = torch.max(pred, dim=1)[1]
			tmp = torch.eq(pred, labels.to(device))
			sum_num += tmp.sum()
			mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)
			
			test_probAll.extend(prob_softmax.cpu().detach().numpy())
			prob_labelAll.extend(pred.cpu().detach().numpy())
			test_labelAll.extend(labels.cpu().detach().numpy())
		test_probAll = np.array(test_probAll)
		test_labelAll = np.array(test_labelAll)
		prob_labelAll = np.array(prob_labelAll)
		
		print(test_probAll)
		print(test_labelAll)
		print(prob_labelAll)
		# 计算预测正确的比例
		test_acc = sum_num.item() / num_samples
		test_auc = utils.cal_roc_auc_score(test_labelAll, test_probAll, n_classes)
		test_f1 = f1_score(test_labelAll, prob_labelAll, average='macro')
		report = classification_report(test_labelAll, prob_labelAll)
		# plot the confusion matrix
		cm = confusion_matrix(test_labelAll, prob_labelAll)
		logger.info('test acc: {:.4f} \n'.format(test_acc))
		logger.info('test auc: {:.4f} \n'.format(test_auc))
		logger.info('test f1: {:.4f} \n'.format(test_f1))
		logger.info('test classification_report: {} \n'.format(report))
		logger.info('test confusion_matrix: {} \n'.format(cm))
		
		# print(cm)
		utils.plot_auc(label_dict, test_labelAll, test_probAll, dir)
		df = pd.DataFrame(data={'label': test_labelAll, 'prob': test_probAll[:, 1]})
		print(df)
		df.to_csv(os.path.join(dir, 'test_pred.csv'))
		return test_acc, test_auc, test_f1
	else:
		logger.info('no train file! please train')
		return None,None,None

if __name__ == '__main__':
	# # 设置随机种子，为了模型效果可复现
	seed_torch(args.random_seed)
	main()