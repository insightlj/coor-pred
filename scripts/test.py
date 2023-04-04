import torch
from config import device

def test(test_dataloader, model, writer=None):
    total_test_step = 1
    total_test_loss = 0

    test_data_size = len(test_dataloader)

    model.eval()
    from config import loss_fn as l

    with torch.no_grad():
        for data in test_dataloader:
            embed, atten, coor_label, L = data

            embed = embed.to(device)
            atten = atten.to(device)
            coor_label = coor_label.to(device)

            coor_label = coor_label.reshape(-1, 3)
            pred = model(embed, atten).reshape(-1, 3)
            loss = l(pred, coor_label)

            total_test_loss = total_test_loss + loss.item()

            if writer:
                writer.add_scalar("avg_loss", total_test_loss / total_test_step, total_test_step)

            total_test_step += 1

    avg_test_loss = total_test_loss / test_data_size

    return avg_test_loss


if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter
    from config import device
    from main import test_dataloader

    import os

    pt_dir = "model/checkpoint/0214_8_classes_1158/"
    logs_folder_name = "0214_8_classes_1158"
    pt_list = os.listdir(pt_dir)
    num_pt = len(pt_list)

    logs_name_sum = "logs/test_" + logs_folder_name + "/" + "summary"
    writer_sum = SummaryWriter(logs_name_sum)

    for ID in range(num_pt):
        print("开始验证模型{}……".format(ID))

        pt_name = "epoch" + str(ID) + ".pt"
        net = torch.load(pt_dir + pt_name)
        net.to(device)

        logs_dir = "logs/test_" + logs_folder_name + "/" + "epoch" + str(ID)
        writer = SummaryWriter(logs_dir)

        # test
        avg_test_loss = test(test_dataloader, net, writer=writer)

        logs_name = "logs/test_" + logs_folder_name + "/" + "epoch" + str(ID)
        writer_test = SummaryWriter(logs_name)
        writer_sum.add_scalar("avg_test_loss", avg_test_loss, int(ID))
