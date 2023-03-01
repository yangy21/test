import torch
from torch import nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch.optim as optim
from Resnet50 import Resnet50, Resnet101
import argparse
import datetime
from Utils_Classify import *


os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="use gpu")

#  需要读取的路径
parser.add_argument("--dataset_dir", type=str, default=r'E:\SanGu\bar_detect\datasets\Cam_Area_FeiDao_NewStad221210\rand_roate_v5', help="data name")
parser.add_argument('--Train_Infor_Path', type=str, default='./File/Cam_AugFeiDao_N28708_D230203.xls')
parser.add_argument('--Val_Infor_Path', type=str, default='./File/Cam_AugFeiDao_N28708_D230203.xls')

#  需要写入的路径
parser.add_argument("--model_save_path", type=str, default='./Resnet_trained', help="save trained model")

parser.add_argument('--Cls_to_label_dic', type=dict, default={'ok':0, 'linjie_ok':0,
                                                                'quekou':1,'s_que':1,
                                                                'weijianguang':2,
                                                               'guashang':3,
                                                              'liewen':4,
                                                              'shuangdao':5
                                                                }, help='name label')
parser.add_argument('--title_list', type=list,
                    default=['ok', 'quekou',
                    'weijianguang', 'guashang', 'liewen', 'shuangdao'],
                    help='choose join training')

parser.add_argument("--In_Width", type=int, default=416, help="input image Width")
parser.add_argument("--In_Height", type=int, default=416, help="input image Height")

#  训练参数
parser.add_argument("--lr_rate", type=int, default=0.0001, help="learning_rate")
parser.add_argument("--batchsize", type=int, default=4, help="batchsize")
parser.add_argument("--epoches", type=int, default=80, help="train epoches")

parser.add_argument("--sleep_freq", type=int, default=10, help="多少epoch sleep一次")
parser.add_argument("--sleep_time", type=int, default=300, help="sleep 时间")
opt = parser.parse_args()

Class_Nums = len(list(set(opt.Cls_to_label_dic.values())))
title_name_list = list(opt.title_list)
print('类别数:', Class_Nums)

train_imgs, train_masks, train_labels = Read_Names(exl_path=opt.Train_Infor_Path,
                                                   mode='train')

val_imgs, val_masks, val_labels = Read_Names(exl_path=opt.Val_Infor_Path,
                                             mode='val')

# train_imgs = train_imgs[0:10]
# train_masks = train_masks[0:10]
# train_labels = train_labels[0:10]
#
# val_imgs = val_imgs[0:10]
# val_masks = val_masks[0:10]
# val_labels = val_labels[0:10]

# 防止最后一个batch仅1张图像，无法进行BN
if len(train_imgs) % 2 != 0:
    train_imgs = train_imgs[:-1]
    train_masks = train_masks[:-1]
    train_label = train_labels[:-1]

print('实际训练集数量:', len(train_imgs), '实际测试集数量:',len(val_imgs))

# color_aug = torchvision.transforms.ColorJitter(
#     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

Augs = torchvision.transforms.Compose([
    #torchvision.transforms.RandomHorizontalFlip(0.2),  # 水平翻转
    #torchvision.transforms.RandomVerticalFlip(0.2),  # 竖直翻转
    #color_aug,
    torchvision.transforms.ToTensor(),
])


train_set = ReadDataset(dataset_dir=opt.dataset_dir,
                        img_list=train_imgs,
                        lab_list=train_labels,
                        in_wid=opt.In_Width,
                        in_hig=opt.In_Height,
                        trans=Augs,
                        mode='train')


val_set = ReadDataset(dataset_dir=opt.dataset_dir,
                        img_list=val_imgs,
                        lab_list=val_labels,
                        in_wid=opt.In_Width,
                        in_hig=opt.In_Height,
                        trans=Augs,
                        mode='val')

train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True)
val_loader = DataLoader(val_set, batch_size=opt.batchsize, shuffle=True)


def main_train(weight_loss=False):

    if weight_loss:  # 分配权重loss
        weights_ls = []
        train_total = len(train_labels)
        for i in range(len(sorted(set(opt.Cls_to_label_dic.values())))):
            weight = (train_total - train_labels.count(i)) / train_total
            weight = round(weight, 2)
            print(f'类别{i}', f'数量:{train_labels.count(i)}', f'loss权重{weight}')
            weights_ls.append(weight)
        weights_arr = np.array(weights_ls)

        loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_arr).float(),
                                            size_average=True).to(device)

    else:
        loss_function = nn.CrossEntropyLoss()

    model = Resnet50(num_classes=Class_Nums).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_rate)
    print('批次数量:', len(train_loader))

    results_file = f"{opt.model_save_path}" + '\\' + "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    epo_tra_loss = []
    epo_tra_acc = []
    epo_val_loss = []
    epo_val_acc = []
    print('---开始训练---')
    for epoch in range(opt.epoches):
        # ------训练过程------
        m_train_loss, m_train_acc, run_time = train_one_epoch(model=model,
                                                              data_loader=train_loader,
                                                              loss_f=loss_function,
                                                              optimizer=optimizer,
                                                              device=device)
        epo_tra_loss.append(m_train_loss)
        epo_tra_acc.append(m_train_acc)

        # ------验证过程-----
        m_val_loss, m_val_acc, val_confusion_mat = evaluate(model=model,
                                                            data_loader=val_loader,
                                                            loss_f=loss_function,
                                                            class_num=Class_Nums,
                                                            device=device)
        epo_val_loss.append(m_val_loss)
        epo_val_acc.append(m_val_acc)

        # ------训练集，验证集指标------
        train_info = f"Epoch {epoch + 1}/{opt.epoches}:" + f" - {run_time}s" + \
                     f" -loss: {m_train_loss}" + f" -acc: {m_train_acc}"

        val_info = f" -val_loss: {m_val_loss}" + f" -val_acc: {m_val_acc}"

        print(train_info + val_info)

        make_dir(path=opt.model_save_path)
        torch.save(model.state_dict(), f'{opt.model_save_path}/Resnet_{epoch + 1}.pth')
        plot_confu_matrix(matrix=val_confusion_mat, labels=title_name_list, save_dir=opt.model_save_path, epoch=epoch + 1)

        plot_curve(epo_tra_loss, epo_val_loss, 'loss', save_dir=opt.model_save_path)
        plot_curve(epo_tra_acc, epo_val_acc, 'accuracy', save_dir=opt.model_save_path)

        with open(results_file, 'a') as f:
            f.write(train_info + val_info + '\n')
            f.close()

        if opt.sleep_freq is not None:
            if (epoch + 1) % opt.sleep_freq == 0:
                sleep(opt.sleep_time)

if __name__ == '__main__':
    main_train()
















