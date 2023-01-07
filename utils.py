import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_parse():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--input_dim', type=int, default=100, help='input feature dimension of LSTM')	
	argparser.add_argument('--timestep', type=int, default=24,help="the window length of data")
	argparser.add_argument('--num_for_predict', type=int, default=1, help='how many steps we would like to predict for the future')
	argparser.add_argument('--val_ratio', type=float, default=0.2)
	argparser.add_argument('--test_ratio', type=float, default=0.2)

	# LSTM模型 训练时的参数设置
	argparser.add_argument('--hidden_size', type=int, default=64,  help='hidden neurons of LSTM layer')
	argparser.add_argument('--num_layers', type=int, default=2, help='number of layers of LSTM')
	argparser.add_argument('--epochs', type=int, default=100)
	argparser.add_argument('--batch_size', type=int, default=32)
	argparser.add_argument('--lr', type=float, default=0.001)
	argparser.add_argument('--dropout', type=float, default=0.01)
	argparser.add_argument('--seed', type=int, default=2023)
	argparser.add_argument('--gpu', action='store_true', default=True, help='Use CUDA for training') 
	return argparser.parse_args()


def read_txt_file(args):
		f = open("datasets/{}.txt".format(args.data_name),"r")
		data = []
		for line in f:
			rs = line.replace("\n","").split(",")
			data.append(rs)
		return np.array(data)


def load_data(args):
    
	if args.data_name == "cellular_traffic":
		data = pd.read_csv("datasets/cellular_traffic.csv")
		data = np.array(data)[:, 1:]

	elif args.data_name in ["exchange_rate", "electricity", "solar-energy", "traffic"]:
		data = read_txt_file(args)
		
		print("data shape", data.shape)
	else:
		print("please find out the dataset!")

	val_len, test_len = int(len(data) * args.val_ratio), int(len(data) * args.test_ratio)
	mmn = MinMaxScaler()
	mmn.fit(data[: -test_len])
	normalized_data = mmn.transform(data)

	X, Y = [], []
	for i in range(len(normalized_data) - args.timestep):
		X.append(normalized_data[i: i + args.timestep])
		Y.append(normalized_data[i + args.timestep])
	X, Y = np.array(X), np.array(Y)

	train_x,val_x, test_x = X[:-(test_len+val_len)], X[-(test_len+val_len):-test_len], X[-test_len:]
	train_y, val_y,test_y = Y[:-(test_len+val_len)], Y[-(test_len+val_len):-test_len], Y[-test_len:]
	return train_x, train_y,val_x,val_y, test_x, test_y, mmn
	


def cal_pccs(x, y, n):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc


def evaluate(y, yp):
    '''
    :param y: the target value
    :param yp: the prediction value
    :return:
    '''
    y,yp = np.reshape(y, newshape=(-1,1)),np.reshape(yp,newshape=(-1,1))
    rmse = np.sqrt(mean_squared_error(y, yp))
    mae = mean_absolute_error(y, yp)
    corr = cal_pccs(y, yp, len(y))
    return rmse, mae, corr

