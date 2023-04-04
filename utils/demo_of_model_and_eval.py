import torch
from main import test_dataset
from torch.utils.data import DataLoader
from config import BATCH_SIZE, device

def demo_pred_label(demo_size, net_pt_name):
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    net_pt = torch.load(net_pt_name, map_location=device)

    i = 0
    ls_pred = []
    ls_label = []
    for data in test_dataloader:
        i += 1
        if i > demo_size:
            break
        embed, atten, coor_label, L = data
        embed = embed.to(device)
        atten = atten.to(device)
        coor_label = coor_label.reshape(-1,3)
        coor_label = coor_label.to(device)
        L = L.to(device)
        pred = net_pt(embed, atten)
        pred, coor_label = pred.reshape((L,L,3)), coor_label.reshape((L,L,3))

        ls_pred.append(pred)
        ls_label.append(coor_label)

    if demo_size == 1:
        return (ls_pred[0], ls_label[0])
    else:
        return ls_pred, ls_label


if __name__ == "__main__":
    DEMO_SIZE = 2
    net_pt_name = "/home/rotation3/coor-pred/model/checkpoint/l1_I/epoch14.pt"

    ls_pred, ls_label = demo_pred_label(demo_size = DEMO_SIZE, net_pt_name = net_pt_name)
    # print(ls_label.__len__(), ls_pred.__len__())
    print(ls_pred[1].shape, ls_label[1].shape)

    from config import loss_fn as l
    p = ls_pred[1].reshape(-1,3)
    q = ls_label[1].reshape(-1,3)
    loss = l(p, q)
    print(loss)
