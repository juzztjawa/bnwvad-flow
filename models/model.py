import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from .normal_head import NormalHead
from .translayer import Transformer

class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

# class WSAD(Module):
#     def __init__(self, input_size, flag, args):
#         super().__init__()
#         self.flag = flag
#         self.args = args
        
#         self.ratio_sample = args.ratio_sample
#         self.ratio_batch = args.ratio_batch
        
#         self.ratios = args.ratios
#         self.kernel_sizes = args.kernel_sizes

#         self.normal_head = NormalHead(in_channel=512, ratios=args.ratios, kernel_sizes=args.kernel_sizes)
#         self.embedding = Temporal(input_size,512)
#         self.selfatt = Transformer(512, 2, 4, 128, 512, dropout = 0)
#         self.step = 0
    
#     def get_normal_scores(self, x, ncrops=None):
#         new_x  = x.permute(0, 2, 1)
        
#         outputs = self.normal_head(new_x)
#         normal_scores = outputs[-1]
#         xhs = outputs[:-1]
        
#         if ncrops:
#             b = normal_scores.shape[0] // ncrops
#             normal_scores = normal_scores.view(b, ncrops, -1).mean(1)
        
#         return xhs, normal_scores
    
#     def get_mahalanobis_distance(self, feats, anchor, var, ncrops = None):
#         distance = torch.sqrt(torch.sum((feats - anchor[None, :, None]) ** 2 / var[None, :, None], dim=1))
#         if ncrops:
#             bs = distance.shape[0] // ncrops
#             # b x t
#             distance = distance.view(bs, ncrops, -1).mean(1)
#         return distance
    
#     def pos_neg_select(self, feats, distance, ncrops):
#         batch_select_ratio = self.ratio_batch
#         sample_select_ratio = self.ratio_sample
#         bs, c, t = feats.shape
#         select_num_sample = int(t * sample_select_ratio)
#         select_num_batch = int(bs // 2 * t * batch_select_ratio)
#         feats = feats.view(bs, ncrops, c, t).mean(1) # b x c x t
#         nor_distance = distance[:bs // 2] # b x t
#         nor_feats = feats[:bs // 2].permute(0, 2, 1) # b x t x c
#         abn_distance = distance[bs // 2:] # b x t
#         abn_feats = feats[bs // 2:].permute(0, 2, 1) # b x t x c
#         abn_distance_flatten = abn_distance.reshape(-1)
#         abn_feats_flatten = abn_feats.reshape(-1, c)
        
#         mask_select_abnormal_sample = torch.zeros_like(abn_distance, dtype=torch.bool)
#         topk_abnormal_sample = torch.topk(abn_distance, select_num_sample, dim=-1)[1]
#         mask_select_abnormal_sample.scatter_(1, topk_abnormal_sample, True)
        
#         mask_select_abnormal_batch = torch.zeros_like(abn_distance_flatten, dtype=torch.bool)
#         topk_abnormal_batch = torch.topk(abn_distance_flatten, select_num_batch, dim=-1)[1]
#         mask_select_abnormal_batch.scatter_(0, topk_abnormal_batch, True)
        
#         mask_select_abnormal = mask_select_abnormal_batch | mask_select_abnormal_sample.reshape(-1)
#         select_abn_feats = abn_feats_flatten[mask_select_abnormal]
        
#         num_select_abnormal = torch.sum(mask_select_abnormal)
        
#         k_nor = int(num_select_abnormal / (bs // 2)) + 1
#         topk_normal_sample = torch.topk(nor_distance, k_nor, dim=-1)[1]
#         select_nor_feats = torch.gather(nor_feats, 1, topk_normal_sample[..., None].expand(-1, -1, c))
#         select_nor_feats = select_nor_feats.permute(1, 0, 2).reshape(-1, c)
#         select_nor_feats = select_nor_feats[:num_select_abnormal]
        
#         return select_nor_feats, select_abn_feats

#     def forward(self, x):
#         if len(x.size()) == 4:
#             b, n, t, d = x.size()
#             x = x.reshape(b * n, t, d)
#         else:
#             b, t, d = x.size()
#             n = 1
#         x = self.embedding(x)
#         x = self.selfatt(x)
        
#         normal_feats, normal_scores = self.get_normal_scores(x, n)
        
#         anchors = [bn.running_mean for bn in self.normal_head.bns]
#         variances = [bn.running_var for bn in self.normal_head.bns]

#         distances = [self.get_mahalanobis_distance(normal_feat, anchor, var, ncrops=n) for normal_feat, anchor, var in zip(normal_feats, anchors, variances)]

#         if self.flag == "Train":
            
#             select_normals = []
#             select_abnormals = []
#             for feat, distance in zip(normal_feats, distances):
#                 select_feat_normal, select_feat_abnormal = self.pos_neg_select(feat, distance, n)
#                 select_normals.append(select_feat_normal[..., None])
#                 select_abnormals.append(select_feat_abnormal[..., None])

#             bn_resutls = dict(
#                 anchors = anchors,
#                 variances = variances,
#                 select_normals = select_normals,
#                 select_abnormals = select_abnormals, 
#             )

#             return {
#                     'pre_normal_scores': normal_scores[0:b // 2],
#                     'bn_results': bn_resutls,
#                 }
#         else:

#             distance_sum = sum(distances)
#             # print(distance_sum,normal_scores, distance_sum * normal_scores)
#             return distance_sum * normal_scores


class WSAD(nn.Module):
    def __init__(self, input_size, flag, args):
        super().__init__()
        self.flag = flag
        self.args = args
        
        self.ratio_sample = args.ratio_sample
        self.ratio_batch = args.ratio_batch
        
        self.ratios = args.ratios
        self.kernel_sizes = args.kernel_sizes

        # Split input into RGB and flow (assuming input is concatenated)
        self.rgb_input_size = input_size // 2
        self.flow_input_size = input_size // 2

        # Separate RGB and flow processing layers
        self.embedding_rgb = Temporal(self.rgb_input_size, 512)
        self.embedding_flow = Temporal(self.flow_input_size, 512)
        
        self.selfatt_rgb = Transformer(512, 2, 4, 128, 512, dropout=0)
        self.selfatt_flow = Transformer(512, 2, 4, 128, 512, dropout=0)
        
        # Single Normal Head for concatenated features (512 + 512 = 1024 input)
        self.normal_head = NormalHead(in_channel=1024, ratios=args.ratios, kernel_sizes=args.kernel_sizes)
        
        self.step = 0
    
    def get_normal_scores(self, x, ncrops=None):
        new_x = x.permute(0, 2, 1)  # Convert to (batch, channels, time)
        outputs = self.normal_head(new_x)
        normal_scores = outputs[-1]
        xhs = outputs[:-1]
        
        if ncrops:
            b = normal_scores.shape[0] // ncrops
            normal_scores = normal_scores.view(b, ncrops, -1).mean(1)
        
        return xhs, normal_scores
    
    def get_mahalanobis_distance(self, feats, anchor, var, ncrops=None):
        distance = torch.sqrt(torch.sum((feats - anchor[None, :, None])**2 / var[None, :, None], dim=1))
        if ncrops:
            bs = distance.shape[0] // ncrops
            distance = distance.view(bs, ncrops, -1).mean(1)
        return distance
    
    def pos_neg_select(self, feats, distance, ncrops):
        batch_select_ratio = self.ratio_batch
        sample_select_ratio = self.ratio_sample
        bs, c, t = feats.shape
        select_num_sample = int(t * sample_select_ratio)
        select_num_batch = int(bs // 2 * t * batch_select_ratio)
        
        # Average across crops (if any)
        feats = feats.view(bs, ncrops, c, t).mean(1) if ncrops else feats
        nor_distance = distance[:bs//2]  # Normal samples
        abn_distance = distance[bs//2:]  # Abnormal samples
        
        nor_feats = feats[:bs//2].permute(0, 2, 1)  # (b, t, c)
        abn_feats = feats[bs//2:].permute(0, 2, 1)  # (b, t, c)
        
        # Select abnormal samples and batches
        abn_distance_flatten = abn_distance.reshape(-1)
        abn_feats_flatten = abn_feats.reshape(-1, c)
        
        mask_abn_sample = torch.zeros_like(abn_distance, dtype=torch.bool)
        topk_abn_sample = torch.topk(abn_distance, select_num_sample, dim=-1).indices
        mask_abn_sample.scatter_(1, topk_abn_sample, True)
        
        mask_abn_batch = torch.zeros_like(abn_distance_flatten, dtype=torch.bool)
        topk_abn_batch = torch.topk(abn_distance_flatten, select_num_batch).indices
        mask_abn_batch.scatter_(0, topk_abn_batch, True)
        
        mask_abn = mask_abn_batch | mask_abn_sample.view(-1)
        select_abn = abn_feats_flatten[mask_abn]
        
        # Select normal samples
        num_abn = mask_abn.sum().item()
        k_nor = int(num_abn / (bs//2)) + 1
        topk_nor = torch.topk(nor_distance, k_nor, dim=-1).indices
        select_nor = torch.gather(nor_feats, 1, topk_nor[..., None].expand(-1, -1, c))
        select_nor = select_nor.permute(1, 0, 2).reshape(-1, c)[:num_abn]
        
        return select_nor, select_abn
    
    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.view(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        
        # Split into RGB and flow features
        rgb = x[:, :, :self.rgb_input_size]
        flow = x[:, :, self.rgb_input_size:]
        
        # Process each modality separately
        rgb = self.embedding_rgb(rgb)
        rgb = self.selfatt_rgb(rgb)
        
        flow = self.embedding_flow(flow)
        flow = self.selfatt_flow(flow)
        
        # Concatenate RGB and flow features
        concat = torch.cat([rgb, flow], dim=2)  # dim=2 is the feature dimension
        
        # Process through shared Normal Head
        normal_feats, normal_scores = self.get_normal_scores(concat, ncrops=n)
        
        # Extract anchors and variances from the single Normal Head
        anchors = [bn.running_mean for bn in self.normal_head.bns]
        variances = [bn.running_var for bn in self.normal_head.bns]
        
        # Compute Mahalanobis distances for each layer
        distances = [
            self.get_mahalanobis_distance(feat, anchor, var, ncrops=n)
            for feat, anchor, var in zip(normal_feats, anchors, variances)
        ]
        
        if self.flag == "Train":
            select_normals = []
            select_abnormals = []
            for feat, dist in zip(normal_feats, distances):
                nor, abn = self.pos_neg_select(feat, dist, n)
                select_normals.append(nor[..., None])
                select_abnormals.append(abn[..., None])
            
            bn_results = {
                'anchors': anchors,
                'variances': variances,
                'select_normals': select_normals,
                'select_abnormals': select_abnormals,
            }
            
            return {
                'pre_normal_scores': normal_scores[:b//2],  # Only normal samples for training loss
                'bn_results': bn_results,
            }
        else:
            # Sum distances across layers and multiply by scores
            total_distance = sum(distances)
            return total_distance * normal_scores
        

####################SEPARATE FLOW FOR RGB AND FLOW###########################################


# class WSAD(Module):
#     def __init__(self, input_size, flag, args):
#         super().__init__()
#         self.flag = flag
#         self.args = args
        
#         self.ratio_sample = args.ratio_sample
#         self.ratio_batch = args.ratio_batch
        
#         self.ratios = args.ratios
#         self.kernel_sizes = args.kernel_sizes

#         # Assuming input_size is the combined size of RGB and flow features
#         self.rgb_input_size = input_size // 2
#         self.flow_input_size = input_size // 2

#         self.normal_head_rgb = NormalHead(in_channel=512, ratios=args.ratios, kernel_sizes=args.kernel_sizes)
#         self.normal_head_flow = NormalHead(in_channel=512, ratios=args.ratios, kernel_sizes=args.kernel_sizes)
        
#         self.embedding_rgb = Temporal(self.rgb_input_size, 512)
#         self.embedding_flow = Temporal(self.flow_input_size, 512)
        
#         self.selfatt_rgb = Transformer(512, 2, 4, 128, 512, dropout=0)
#         self.selfatt_flow = Transformer(512, 2, 4, 128, 512, dropout=0)
        
#         # New parameters for controlling the contribution of RGB and flow
#         self.rgb_weight = args.rgb_weight
#         self.flow_weight = args.flow_weight
        
#         self.step = 0
    
#     def get_normal_scores(self, x, normal_head, ncrops=None):
#         new_x = x.permute(0, 2, 1)
#         outputs = normal_head(new_x)
#         normal_scores = outputs[-1]
#         xhs = outputs[:-1]
        
#         if ncrops:
#             b = normal_scores.shape[0] // ncrops
#             normal_scores = normal_scores.view(b, ncrops, -1).mean(1)
        
#         return xhs, normal_scores
    
#     def get_mahalanobis_distance(self, feats, anchor, var, ncrops=None):
#         distance = torch.sqrt(torch.sum((feats - anchor[None, :, None]) ** 2 / var[None, :, None], dim=1))
#         if ncrops:
#             bs = distance.shape[0] // ncrops
#             distance = distance.view(bs, ncrops, -1).mean(1)
#         return distance
    
#     def pos_neg_select(self, feats, distance, ncrops):
#         batch_select_ratio = self.ratio_batch
#         sample_select_ratio = self.ratio_sample
#         bs, c, t = feats.shape
#         select_num_sample = int(t * sample_select_ratio)
#         select_num_batch = int(bs // 2 * t * batch_select_ratio)
#         feats = feats.view(bs, ncrops, c, t).mean(1)  # b x c x t
#         nor_distance = distance[:bs // 2]  # b x t
#         nor_feats = feats[:bs // 2].permute(0, 2, 1)  # b x t x c
#         abn_distance = distance[bs // 2:]  # b x t
#         abn_feats = feats[bs // 2:].permute(0, 2, 1)  # b x t x c
#         abn_distance_flatten = abn_distance.reshape(-1)
#         abn_feats_flatten = abn_feats.reshape(-1, c)
        
#         mask_select_abnormal_sample = torch.zeros_like(abn_distance, dtype=torch.bool)
#         topk_abnormal_sample = torch.topk(abn_distance, select_num_sample, dim=-1)[1]
#         mask_select_abnormal_sample.scatter_(1, topk_abnormal_sample, True)
        
#         mask_select_abnormal_batch = torch.zeros_like(abn_distance_flatten, dtype=torch.bool)
#         topk_abnormal_batch = torch.topk(abn_distance_flatten, select_num_batch, dim=-1)[1]
#         mask_select_abnormal_batch.scatter_(0, topk_abnormal_batch, True)
        
#         mask_select_abnormal = mask_select_abnormal_batch | mask_select_abnormal_sample.reshape(-1)
#         select_abn_feats = abn_feats_flatten[mask_select_abnormal]
        
#         num_select_abnormal = torch.sum(mask_select_abnormal)
        
#         k_nor = int(num_select_abnormal / (bs // 2)) + 1
#         topk_normal_sample = torch.topk(nor_distance, k_nor, dim=-1)[1]
#         select_nor_feats = torch.gather(nor_feats, 1, topk_normal_sample[..., None].expand(-1, -1, c))
#         select_nor_feats = select_nor_feats.permute(1, 0, 2).reshape(-1, c)
#         select_nor_feats = select_nor_feats[:num_select_abnormal]
        
#         return select_nor_feats, select_abn_feats

#     def forward(self, x):
#         if len(x.size()) == 4:
#             b, n, t, d = x.size()
#             x = x.reshape(b * n, t, d)
#         else:
#             b, t, d = x.size()
#             n = 1
        
#         # Split the features into RGB and flow
#         rgb_features = x[:, :, :self.rgb_input_size]
#         flow_features = x[:, :, self.rgb_input_size:]
        
#         # Process RGB features
#         rgb_features = self.embedding_rgb(rgb_features)
#         rgb_features = self.selfatt_rgb(rgb_features)
#         rgb_normal_feats, rgb_normal_scores = self.get_normal_scores(rgb_features, self.normal_head_rgb, n)
        
#         # Process flow features
#         flow_features = self.embedding_flow(flow_features)
#         flow_features = self.selfatt_flow(flow_features)
#         flow_normal_feats, flow_normal_scores = self.get_normal_scores(flow_features, self.normal_head_flow, n)
        
#         # Combine normal scores with weights
#         combined_normal_scores = (self.rgb_weight * rgb_normal_scores + self.flow_weight * flow_normal_scores)
        
#         anchors_rgb = [bn.running_mean for bn in self.normal_head_rgb.bns]
#         variances_rgb = [bn.running_var for bn in self.normal_head_rgb.bns]
        
#         anchors_flow = [bn.running_mean for bn in self.normal_head_flow.bns]
#         variances_flow = [bn.running_var for bn in self.normal_head_flow.bns]
        
#         rgb_distances = [
#             self.get_mahalanobis_distance(normal_feat, anchor, var, ncrops=n)
#             for normal_feat, anchor, var in zip(rgb_normal_feats, anchors_rgb, variances_rgb)
#         ]
        
#         flow_distances = [
#             self.get_mahalanobis_distance(normal_feat, anchor, var, ncrops=n)
#             for normal_feat, anchor, var in zip(flow_normal_feats, anchors_flow, variances_flow)
#         ]
        
#         if self.flag == "Train":
#             rgb_select_normals = []
#             rgb_select_abnormals = []
#             for feat, distance in zip(rgb_normal_feats, rgb_distances):
#                 select_feat_normal, select_feat_abnormal = self.pos_neg_select(feat, distance, n)
#                 rgb_select_normals.append(select_feat_normal[..., None])
#                 rgb_select_abnormals.append(select_feat_abnormal[..., None])
            
#             flow_select_normals = []
#             flow_select_abnormals = []
#             for feat, distance in zip(flow_normal_feats, flow_distances):
#                 select_feat_normal, select_feat_abnormal = self.pos_neg_select(feat, distance, n)
#                 flow_select_normals.append(select_feat_normal[..., None])
#                 flow_select_abnormals.append(select_feat_abnormal[..., None])
            
#             bn_results_rgb = dict(
#                 anchors=anchors_rgb,
#                 variances=variances_rgb,
#                 select_normals=rgb_select_normals,
#                 select_abnormals=rgb_select_abnormals,
#             )
            
#             bn_results_flow = dict(
#                 anchors=anchors_flow,
#                 variances=variances_flow,
#                 select_normals=flow_select_normals,
#                 select_abnormals=flow_select_abnormals,
#             )
            
#             return {
#                 'pre_normal_scores_rgb': rgb_normal_scores,
#                 'pre_normal_scores_flow': flow_normal_scores,
#                 'bn_results_rgb': bn_results_rgb,
#                 'bn_results_flow': bn_results_flow,
#             }
#         else:
#             combined_distance_sum = sum(rgb_distances) + sum(flow_distances)
#             return combined_distance_sum * combined_normal_scores
