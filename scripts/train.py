from torch import optim
from config import device


def train(train_dataloader, model, loss_fn, writer, epoch_ID, learning_rate=5e-4):

    total_train_step = 1
    total_loss = 0

    if epoch_ID == 0:
        if total_train_step < 1000:
            learning_rate = 5e-4
        elif 1000 <= total_train_step < 2000:
            learning_rate = 1e-4
        # elif total_train_step >= 2000:
        #     learning_rate = 2e-5

    else:
        learning_rate = 2e-5

    model.to(device)
    model.train()

    for data in train_dataloader:
        embed, atten, coor_label, L = data

        coor_label = coor_label.to(device)
        embed = embed.to(device)
        atten = atten.to(device)

        pred = model(embed, atten).reshape(-1, 3)
        coor_label = coor_label.reshape(-1, 3)

        loss = loss_fn(pred, coor_label)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        l = loss.item()
        total_loss = total_loss + l

        avg_train_loss = total_loss / total_train_step

        if total_train_step == 1 or total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("avg_train_loss", avg_train_loss, total_train_step)

    return avg_train_loss