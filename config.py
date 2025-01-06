import torch


# class Config:
#     def __init__(self):
#         super(Config, self).__init__()
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.batch_size = 30
#         self.epoch_size = 2
#         self.learning_rate = 5e-4
#         self.weight_decay = 1e-2
#         self.lookback = 10
import torch
class Config:
    def __init__(self):
        self.lookback = 20 
        self.n_steps = 1    
        self.batch_size = 27
        self.epoch_size = 6
        self.learning_rate = 0.01
        self.weight_decay = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





