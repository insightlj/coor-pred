import torch
import os
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

BATCH_SIZE = 1
EPOCH_NUM = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# # MSE(L2Loss)
run_name = "l2_II"
error_file_name = 'errorlog_l2_II.txt'


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, label):
        loss = (pred - label) ** 2
        loss = (loss.sum(dim=1) ** 0.5).mean()
        return loss

loss_fn = My_loss()

# # MAE(L1Loss)
# run_name = "l1_I"
# error_file_name = 'errorlog_l1_I.txt'
# loss_fn = torch.nn.L1Loss()
