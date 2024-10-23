import argparse
from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import model
from detection_layers.modules import MultiBoxLoss
from dataset_CelebDF import CelebDF
from lib.util import load_config, update_learning_rate, my_collate
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, help='The path to the train config.', default='./configs/caddm_train.cfg')
    parser.add_argument('--test_cfg', type=str, help='The path to the test config.', default='./configs/caddm_test.cfg')
    #parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.',
    #                    default="/home/gaohui/FF++_CADDM/CADDM_ckpt/epoch_96.pkl")     #记得改！！！
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.',
                        default="/home/gaohui/CADDM-master/checkpoints/resnet34.pkl")
    
    args = parser.parse_args()
    return args


def load_checkpoint(ckpt, net, opt, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module."+k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    opt.load_state_dict(checkpoint['opt_state'])
    base_epoch = int(checkpoint['epoch']) + 1
    return net, opt, base_epoch


def load_checkpoint_for_test(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module." + k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net


def test():
    args = args_func()
    # load conifigs
    cfg = load_config(args.train_cfg)
    test_cfg = load_config(args.test_cfg)
    base_epoch = 0

    # init model.
    net = model.get(backbone=cfg['model']['backbone'])
    if torch.cuda.is_available():
        print("gpu is available")

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=[3])  # 这个地方要注意，可能要改！！！

    #测试好像不需要loss，所以这里暂时没写
    #优化器也没写

    if args.ckpt:
        net = load_checkpoint_for_test(args.ckpt, net, device)

    # get testing data
    test_dataset = CelebDF(cfg)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['train']['batch_size'],
                             shuffle=True, num_workers=10,  # 这个地方可能报错
                             collate_fn=my_collate  # CADDM原本的test_loader是没有这行的。
                             )

    print("dataset is ready, now begin training")

    all_preds = []
    all_labels = []
    total_correct = 0.0
    total_samples = 0.0

    net.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc='Testing', unit='batch'):
            labels, video_name = batch_labels
            labels = labels.long().to(device)
            batch_data = batch_data.to(device)
            outputs = net(batch_data)
            # auc计算
            all_preds.extend(outputs[:, 1].detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            # acc计算
            total_correct += sum(outputs.max(-1).indices == labels).item()
            total_samples += labels.shape[0]
        print(f'总测试样本数量:{total_samples}')
        print(f'测试正确样本数量:{total_correct}')
        eval_acc = total_correct / total_samples
        eval_auc = roc_auc_score(all_labels, all_preds)

        print(f'测试AUC{eval_auc},ACC{eval_acc}。')


if __name__ == "__main__":
    test()










