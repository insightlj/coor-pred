from config import BATCH_SIZE, loss_fn as l, run_name, EPOCH_NUM, error_file_name

import torch
import traceback
import os

from data.MyData import MyData
from torch.utils.data import DataLoader
from model.ResNet import MyResNet


data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
train_file = "/home/rotation3/example/train_list.txt"
test_file = "/home/rotation3/example/valid_list.txt"

train_dataset = MyData(data_path, xyz_path, filename=train_file, train_mode=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = MyData(data_path, xyz_path, filename=test_file, train_mode=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MyResNet()

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from scripts.train import train
    from scripts.test import test
    from utils.init_parameters import weight_init

    try:
        logs_folder_name = run_name
        epoch_num = EPOCH_NUM

        logs_name_sum_train = "logs/" + logs_folder_name + "/" + "summary/train"
        logs_name_sum_test = "logs/" + logs_folder_name + "/" + "summary/test"
        writer_sum_train = SummaryWriter(logs_name_sum_train)
        writer_sum_test = SummaryWriter(logs_name_sum_test)

        model.apply(weight_init)

        # l2_epoch14_pt_name = "/home/rotation3/coor-pred/model/checkpoint/l1_I/epoch14.pt"
        # from utils.model_params_transfer import load_model_and_params_transfer
        # load_model_and_params_transfer(l2_epoch14_pt_name, model)

        for epoch in range(epoch_num):
        # for epoch in range(epoch_num, epoch_num_2):
            logs_name = "logs/" + logs_folder_name + "/" + "train/" + "epoch" + str(epoch)
            writer_train = SummaryWriter(logs_name)

            model_dir_name = logs_folder_name
            dir = "model/checkpoint/" + model_dir_name + "/"
            filename = "epoch" + str(epoch) + ".pt"

            if not os.path.exists(dir):
                os.mkdir(dir)
            torch.save(model, dir + filename)

            # train
            avg_train_loss = train(train_dataloader, model, l, writer_train, epoch, learning_rate=5e-4)
            writer_sum_train.add_scalar("avg_train_loss", avg_train_loss, epoch)

            # test  
            logs_name = "logs/" + logs_folder_name + "/" + "test/" + "epoch" + str(epoch)
            writer_test = SummaryWriter(logs_name)
            avg_test_loss = test(test_dataloader, model, writer_test)
            writer_sum_test.add_scalar("avg_test_loss", avg_test_loss, epoch)

        status = 0

    except Exception as e:
        print("出错：{}".format(e))
        status = e
        traceback.print_exc(file=open(error_file_name, 'a'))
        print(traceback.format_exc())

    from utils.send_email import send_email
    if not status:
        content = "运行成功"
        send_email(content="运行成功")
    else:
        send_email(content="运行失败\n" + str(status))
