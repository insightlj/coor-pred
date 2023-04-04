import torch
from torch import nn
from config import device
from model.ResNet import MyResNet

def load_model_and_params_transfer(model_pt_name, to_transfer_model):

    model = torch.load(model_pt_name)
    model_params = nn.Module.state_dict(model)
    nn.Module.load_state_dict(to_transfer_model, model_params)

    return

if __name__ == "__main__":
    l1_pt_name = "/home/rotation3/coor-pred/model/checkpoint/l1_I/epoch14.pt"
    init_l1 = MyResNet()

    load_model_and_params_transfer(l1_pt_name, init_l1)
