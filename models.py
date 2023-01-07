
from torch import nn


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()    
        self.grulayer = nn.GRU(
            input_size=args.input_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=False)
        self.fclayer = nn.Linear(args.hidden_size, args.input_dim)


    def forward(self, x):
        x = x.permute((1, 0, 2))   
        out, _ = self.grulayer(x)     
        out = out[-1]  # last prediction
        out = self.fclayer(out)
        return out
    
    

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()    
        self.lstmlayer = nn.LSTM(
            input_size=args.input_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=False)
        self.fclayer = nn.Linear(args.hidden_size, args.input_dim)


    def forward(self, x):
        x = x.permute((1, 0, 2))   
        out, _ = self.lstmlayer(x)     
        out = out[-1]  # last prediction
        out = self.fclayer(out)
        return out