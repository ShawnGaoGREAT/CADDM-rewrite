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
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, help='The path to the train config.', default='./configs/caddm_train.cfg')
    parser.add_argument('--test_cfg', type=str, help='The path to the test config.', default='./configs/caddm_test.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.', default=None)
    args = parser.parse_args()
    return args

def save_checkpoint(net, opt, save_path, epoch_num):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    opt_state_dict = {}
    opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    opt_state_dict['state'] = OrderedDict()
    for k, v in opt.state_dict()['state'].items():
        opt_state_dict['state'][k] = {}
        opt_state_dict['state'][k]['step'] = v['step']
        if 'exp_avg' in v:
            opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
        if 'exp_avg_sq' in v:
            opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}.pkl')


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

def gh_train():
    args = args_func()
    # load conifigs
    cfg = load_config(args.train_cfg)
    test_cfg = load_config(args.test_cfg)
    base_epoch = 0

    # init model.
    net = model.get(backbone=cfg['model']['backbone'])
    if torch.cuda.is_available():
        print("gpu is available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)

    # loss init
    det_criterion = MultiBoxLoss(
        cfg['det_loss']['num_classes'],
        cfg['det_loss']['overlap_thresh'],
        cfg['det_loss']['prior_for_matching'],
        cfg['det_loss']['bkg_label'],
        cfg['det_loss']['neg_mining'],
        cfg['det_loss']['neg_pos'],
        cfg['det_loss']['neg_overlap'],
        cfg['det_loss']['encode_target'],
        cfg['det_loss']['use_gpu']
    )
    criterion = nn.CrossEntropyLoss()

    # optimizer init.
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

    # load checkpoint if given

    if args.ckpt:
        net, optimizer, base_epoch = load_checkpoint(args.ckpt, net, optimizer, device)

    # get training data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    train_dataset = DeepfakeDataset('train', cfg)
    test_dataset = DeepfakeDataset('test', cfg)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=14,#这个地方可能报错
                              collate_fn=my_collate
                              )
    test_loader = DataLoader(test_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=14,      #这个地方可能报错
                              collate_fn=my_collate             #CADDM原本的test_loader是没有这行的。
                              )
    print("dataset is ready, now begin training")

    all_preds = []
    all_labels = []
    total_correct = 0.0
    total_samples = 0.0
    #训练部分
    
    loss_sum = 0.0
    best_loss = 100000.0
    jishuqi = 0 #用来求平均loss的除数，train样本除以batch_size的结果
    for epoch in tqdm(range(base_epoch, cfg['train']['epoch_num']), desc='Overall Training Progress', unit='epoch'):
        net.train()
        train_loader_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave =False)
        for index, (batch_data, batch_labels) in enumerate(train_loader_pbar):
            
            #学习率更新
            lr = update_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            batch_data = batch_data.to(device)
            #拿到各种label
            labels, location_labels, confidence_labels = batch_labels
            labels = labels.long().to(device)
            location_labels = location_labels.to(device)
            confidence_labels = confidence_labels.long().to(device)
            #优化器更新
            optimizer.zero_grad()
            #前向传播
            locations, confidence, outputs = net(batch_data)
            #计算loss
            loss_end_cls = criterion(outputs, labels)
            loss_l, loss_c = det_criterion(
                (locations, confidence),
                confidence_labels, location_labels
            )
            det_loss = 0.1 * (loss_l + loss_c)
            loss = det_loss + loss_end_cls
            loss_sum += loss.item()
            jishuqi+=1
            #准确率
            #acc = sum(outputs.max(-1).indices == labels).item() / labels.shape[0]
            total_correct += sum(outputs.max(-1).indices == labels).item()
            total_samples += labels.shape[0]
            #反向传播
            loss.backward()
            #梯度下降
            torch.nn.utils.clip_grad_value_(net.parameters(), 2)
            optimizer.step()
            #AUC计算
            softmax_scores = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.extend(softmax_scores)
            all_labels.extend(labels.detach().cpu().numpy())
        epoch_train_auc = roc_auc_score(all_labels, all_preds)
        epoch_train_acc = total_correct / total_samples
        all_preds = []
        all_labels = []
        total_correct = 0.0
        total_samples = 0.0
        #测试
        net.eval()
        test_loader_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1} Testing", leave=False)
        for batch_data, batch_labels in test_loader_pbar:
            labels, video_name = batch_labels
            labels = labels.long().to(device)
            batch_data = batch_data.to(device)


            outputs = net(batch_data)
            #outputs = outputs[:, 1]
            #auc计算
            all_preds.extend(outputs[:,1].detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())#最后这两行再确定下，然后上面没用tolist，研究下行不行，
            #acc计算
            total_correct += sum(outputs.max(-1).indices == labels).item()
            total_samples += labels.shape[0]
        print(f'epoch:{epoch} 总测试样本数量:{total_samples}')
        print(f'epoch:{epoch}测试正确样本数量:{total_correct}')
        epoch_eval_acc = total_correct / total_samples
        epoch_eval_auc = roc_auc_score(all_labels, all_preds)
        all_preds = []
        all_labels = []
        total_correct = 0.0
        total_samples = 0.0
        print(f'第{epoch}个epoch：训练平均Loss：{loss_sum/jishuqi}，AUC：{epoch_train_auc},ACC：{epoch_train_acc}；测试AUC{epoch_eval_auc},ACC{epoch_eval_acc}。')

        if (epoch+1) % 50 == 0 :
            save_checkpoint(net, optimizer,
                            cfg['model']['save_path'],
                            epoch)
            
        
        loss_sum = 0.0
        jishuqi = 0



if __name__ == '__main__':
    gh_train()




