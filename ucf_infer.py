import torch
import numpy as np
from dataset_loader import UCF_crime
from options import parse_args
import pdb
import utils
import os
from models import WSAD
from tqdm import tqdm
from dataset_loader import data
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import prettytable

def get_predict(test_loader, net):
    load_iter = iter(test_loader)
    frame_predict = []
    print(f"Dataset size: {len(test_loader.dataset)}")
    print(f"Batch size: {test_loader.batch_size}")
    print(f"Number of batches: {len(test_loader)}")
    norm = 0
    for i in range(len(test_loader.dataset)//10):
        _data, _label,name = next(load_iter)
        
        _data = _data.cuda()
        _label = _label.cuda()
        print(_data.shape,name)
        if "Normal" in name[0]:
            norm+=1
        try:
            res = net(_data)   
        except:
            
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOO",i,norm)
            # continue
            break
        print("******",res.shape)
        a_predict = res.cpu().numpy().mean(0)   

        fpre_ = np.repeat(a_predict, 16)
        frame_predict.append(fpre_)
        # print(_data.shape)
        # print(fpre_)
        # if "Normal" in name[0]:
        #     print(min(fpre_), max(fpre_))

    frame_predict = np.concatenate(frame_predict, axis=0)
    # print(frame_predict)
    return frame_predict

# def get_sub_metrics(frame_predict, frame_gt):
#     anomaly_mask = np.load('frame_label/xd_anomaly_mask.npy')
#     sub_predict = frame_predict[anomaly_mask]
#     sub_gt = frame_gt[anomaly_mask]
    
#     fpr,tpr,_ = roc_curve(sub_gt, sub_predict)
#     auc_sub = auc(fpr, tpr)

#     precision, recall, th = precision_recall_curve(sub_gt, sub_predict)
#     ap_sub = auc(recall, precision)
#     return auc_sub, ap_sub

def get_metrics(frame_predict, frame_gt):
    metrics = {}
    
    fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)
    
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)

    # auc_sub, ap_sub = get_sub_metrics(frame_predict, frame_gt)
    # metrics['AUC_sub'] = auc_sub
    # metrics['AP_sub'] = ap_sub

    return metrics

def test(net, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        frame_gt = np.load("frame_label/ucf_gt.npy")
        print(frame_gt)
        frame_predict = get_predict(test_loader, net)
        print(frame_gt.shape, frame_predict.shape)
        frame_gt = frame_gt[:len(frame_predict)]
        print(frame_gt.shape, frame_predict.shape)
        metrics = get_metrics(frame_predict, frame_gt)
        
        test_info['step'].append(step)
        for score_name, score in metrics.items():
            metrics[score_name] = score * 100
            test_info[score_name].append(metrics[score_name])

        return metrics


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    worker_init_fn = None
    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    net = WSAD(args.len_feature, flag = "Test", args = args)
    net = net.cuda()
    test_loader = data.DataLoader(
        UCF_crime(root_dir = args.root_dir, mode = 'Test', num_segments = args.num_segments, len_feature = args.len_feature),
            batch_size = 10,
            shuffle = False, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn)
    
    test_info = {'step': [], 'AUC': [],  'AP': []}

    res = test(net, test_loader, test_info, 1, model_file = args.model_path)

    pt = prettytable.PrettyTable()
    pt.field_names = ['AUC', 'AP']
    for k, v in res.items():
        res[k] = round(v, 2)
    pt.add_row([res['AUC'], res['AP']])
    print(pt)