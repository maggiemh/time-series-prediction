import numpy as np 
import torch
from utils import get_parse, load_data,evaluate
from models import LSTM, GRU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
import torch.utils.data as Data
import pickle

torch.manual_seed(2023)
np.random.seed(2023)

def _to_numpy(A):
    return A.cpu().detach().numpy()

def numpy_to_tvar(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    return x.cuda()


if __name__=="__main__":
	args = get_parse()
	args.device = device
	args.model_name = ["LSTM","GRU"][0]   
	args.data_name = ["cellular_traffic","exchange_rate","electricity","solar-energy","traffic"][1]
	print("all the paremeters are: ", args)

	train_x, train_y,val_x,val_y, test_x, test_y, mmn = load_data(args)
	args.input_dim = train_x.shape[-1]
	print(train_x.shape,test_x.shape, train_y.shape, test_y.shape)
 
	train_x, train_y,val_x,val_y, test_x, test_y = numpy_to_tvar(train_x), numpy_to_tvar(train_y), numpy_to_tvar(val_x), numpy_to_tvar(val_y), numpy_to_tvar(test_x), numpy_to_tvar(test_y)
	train_dataset, val_dataset, test_dataset = Data.TensorDataset(train_x, train_y), Data.TensorDataset(val_x, val_y), Data.TensorDataset(test_x, test_y)
	train_loader,val_loader, test_loader = Data.DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True), Data.DataLoader(dataset=val_dataset,batch_size=args.batch_size,shuffle=False), Data.DataLoader(dataset=test_dataset,batch_size=args.batch_size, shuffle=False)

	if args.model_name == "LSTM":	
		model = LSTM(args).cuda()
	
	elif args.model_name == "GRU":
		model = GRU(args).cuda()
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.98), eps=1e-9)
	criterion = nn.MSELoss().to(device)  

	total_train_loss,total_val_loss = [], []
	min_loss_value = 1
	for epoch in range(1,args.epochs + 1):
		
		model.train()
		train_loss = 0
		for idx, (x, y) in enumerate(train_loader):
			optimizer.zero_grad()
			model.zero_grad()
			pred = model(x)
			loss = criterion(pred.float(), y.float())
			train_loss += loss.item()

			loss.backward()
			optimizer.step()
		total_train_loss.append(train_loss / len(train_loader))
		#print(f'\n | Global Training Round: {epoch} |\n', " training loss is ", total_loss[-1])

		if epoch % 10 == 0:
			model.eval()			
			val_loss = 0
			for idx, (x, y) in enumerate(val_loader):
				pred = model(x)
				loss = criterion(pred.float(), y.float())
				val_loss += loss.item()
			total_val_loss.append(val_loss / len(val_loader))
			print("training epoch", epoch, "training loss", total_train_loss[-1], "val loss", total_val_loss[-1])
			if total_val_loss[-1] < min_loss_value:
				torch.save(model, "best_model.pt")
				min_loss_value = total_val_loss[-1]
   
	print("begin to evaluate on test data!")
	loaded_model = torch.load("best_model.pt")
	prediction, truth = [], []
	total_test_loss = []
	for idx, (x, y) in enumerate(test_loader):
		pred = loaded_model(x)
		prediction.append(_to_numpy(pred))
		truth.append(_to_numpy(y))
	prediction, truth = np.concatenate(prediction), np.concatenate(truth)
	res_prediction, res_truth = mmn.inverse_transform(prediction), mmn.inverse_transform(truth)
	rmse, mae, corr = evaluate(res_prediction, res_truth)
	print("model: {}  data: {}  rmse: {} mae: {}  corr: {}".format(args.model_name, args.data_name, rmse, mae, corr))



