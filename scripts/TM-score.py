from utils.demo_of_model_and_eval import demo_pred_label


def cal_tm(pred, label, sample_size=10):
    """
    pred [L,L,3]
    label [L,L,3]
    sample_size 从L套坐标系中选取sample_size套进行tm_score的计算
    """
    from tmtools import tm_align
    import random

    L = pred.shape[0]
    seq = "A" * int(L)
    total_tm = 0
    for _ in range(sample_size):
        index = random.randint(0,L-1)
        pred_index, label_index = pred[index, :,:], label[index, :,:]
        pred_index, label_index = pred_index.cpu().detach().numpy(), label_index.cpu().detach().numpy()
        tm = tm_align(pred_index, label_index, seq, seq)
        total_tm += tm.tm_norm_chain1

    avg_tm = total_tm / sample_size
    return avg_tm

def cal_tm_traverse(pred, label):
    """分别对L套坐标系进行tm_score的计算并计算平均值
    pred [L,L,3]
    label [L,L,3]
    """
    from tmtools import tm_align

    L = pred.shape[0]
    sample_size = L
    seq = "A" * int(L)
    total_tm = 0
    for index in range(sample_size):
        pred_index, label_index = pred[index, :,:], label[index, :,:]
        pred_index, label_index = pred_index.cpu().detach().numpy(), label_index.cpu().detach().numpy()
        tm = tm_align(pred_index, label_index, seq, seq)
        total_tm += tm.tm_norm_chain1

    avg_tm = total_tm / sample_size
    return avg_tm

if __name__ == "__ main__":
    ### PARAMS ###
    eval_size = 20
    sample_size = 10
    model_pt_name = "/home/rotation3/coor-pred/model/checkpoint/l2_I/epoch29.pt"

    pred_ls, label_ls = demo_pred_label(eval_size, model_pt_name)
    demo_tm = 0
    for i in range(eval_size):
        pred = pred_ls[i]
        label = label_ls[i]
        sample_size = pred.shape[0]
        tm = cal_tm(pred, label, sample_size=sample_size)
        print("sample{}: tm-score:{}".format(i+1, tm))
        demo_tm += tm
    print("avg_tm_score:{}".format(demo_tm / eval_size))

if __name__ == "__main__":
    ### PARAMS ###
    eval_size = 20
    model_pt_name = "/home/rotation3/coor-pred/model/checkpoint/l2_I/epoch29.pt"

    pred_ls, label_ls = demo_pred_label(eval_size, model_pt_name)
    demo_tm = 0
    for i in range(eval_size):
        pred = pred_ls[i]
        label = label_ls[i]
        sample_size = pred.shape[0]
        tm = cal_tm(pred, label)
        print("sample{}: tm-score:{}".format(i+1, tm))
        demo_tm += tm
    print("avg_tm_score:{}".format(demo_tm / eval_size))