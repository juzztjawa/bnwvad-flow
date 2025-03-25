import torch
from options import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.cuda.amp import autocast
import warnings
warnings.filterwarnings("ignore")


from tqdm import tqdm
from dataset_loader import data
from sklearn.metrics import roc_curve,auc,precision_recall_curve


# def get_predicts(test_loader, net):
#     frame_predict = []
#     frame_gt = []  # Ground truth list
    
#     test_iter = iter(test_loader)
#     num_batches = len(test_loader) // 2  # Reduce test loader size to avoid memory issues
    
#     # Select batches in an interleaved manner to include both normal and anomaly videos
#     normal_batches = []
#     anomaly_batches = []
    
#     for _ in range(len(test_loader)):
#         try:
#             _data, _label, name = next(test_iter)
#             if "Normal" in name[0]:
#                 normal_batches.append((_data, _label, name))
#             else:
#                 anomaly_batches.append((_data, _label, name))
#         except StopIteration:
#             break 
    
#     selected_batches = normal_batches[:num_batches//2] + anomaly_batches[:num_batches//2]
    
#     for _data, _label, name in selected_batches:
#         _data = _data.cuda()
#         _label = _label.cuda()
        
#         with torch.no_grad():
#             res = net(_data)
#             a_predict = res.cpu().numpy().mean(0)
#             fpre_ = np.repeat(a_predict, 16)  # Keep repeat value as 16
#             frame_predict.extend(fpre_)  # Ensure flat 1D structure
        
#         label = 0 if "Normal" in name[0] else 1
#         frame_gt.extend([label] * fpre_.shape[0])  # Ensure flat 1D structure
        
#         # Free GPU memory
#         del _data, _label, res
#         torch.cuda.empty_cache()
    
#     frame_predict = np.array(frame_predict)  # Convert to 1D array
#     frame_gt = np.array(frame_gt)  # Convert to 1D array
    
#     return frame_predict, frame_gt

# def get_metrics(frame_predict, frame_gt):
#     metrics = {}
#     fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
#     metrics['AUC'] = auc(fpr, tpr)
    
#     precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
#     metrics['AP'] = auc(recall, precision)
    
#     return metrics

# def test(net, test_loader, test_info, step, model_file=None):
#     with torch.no_grad():
#         net.eval()
#         net.flag = "Test"
#         if model_file is not None:
#             net.load_state_dict(torch.load(model_file))
        
#         frame_predicts, frame_gt = get_predicts(test_loader, net)
#         metrics = get_metrics(frame_predicts, frame_gt)
        
#         test_info['step'].append(step)
#         for score_name, score in metrics.items():
#             metrics[score_name] = score * 100
#             test_info[score_name].append(metrics[score_name])
        
#         return metrics

def get_predict(test_loader, net):
    load_iter = iter(test_loader)
    frame_predict = []
    norm = 0
    for i in range(len(test_loader.dataset)//10):
        _data, _label,name = next(load_iter)
        
        _data = _data.cuda()
        _label = _label.cuda()
        # print(_data.shape,name)
        if "Normal" in name[0]:
            norm+=1
        try:
            res = net(_data)   
        except:
            
            # print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOO",i,norm)
            # continue
            break
        a_predict = res.cpu().numpy().mean(0)   

        fpre_ = np.repeat(a_predict, 16)
        frame_predict.append(fpre_)
        # if "Normal" in name[0]:
        #     print(min(fpre_), max(fpre_))

    frame_predict = np.concatenate(frame_predict, axis=0)
    # print(frame_predict)
    return frame_predict

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
        # print(frame_gt)
        frame_predict = get_predict(test_loader, net)
        # print(frame_gt.shape, frame_predict.shape)
        frame_gt = frame_gt[:len(frame_predict)]
        # print(frame_gt.shape, frame_predict.shape)
        metrics = get_metrics(frame_predict, frame_gt)
        
        test_info['step'].append(step)
        for score_name, score in metrics.items():
            metrics[score_name] = score * 100
            test_info[score_name].append(metrics[score_name])

        return metrics
