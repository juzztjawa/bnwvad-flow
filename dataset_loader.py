import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class UCF_crime(data.Dataset):  # Use `data.Dataset` instead of `data.DataLoader`
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        
        self.root_dir = root_dir  # Store the root directory
        self.mode = mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        # print("yeahhhhhhhhhhh")
        # Load the split file
        split_path = os.path.join('list', 'UCF_{}_updated.list'.format(self.mode))
        split_file = open(split_path, 'r', encoding="utf-8")
        self.vid_list = []
        
        for line in split_file:
            self.vid_list.append(line.strip().split())
        
        split_file.close()
        
        # Filter videos based on normal/abnormal labels for training
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[810:]  # Normal videos
            elif is_normal is False:
                self.vid_list = self.vid_list[:810]  # Abnormal videos
            else:
                assert is_normal is None, "Please set is_normal=[True/False]"
                print("Please ensure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def get_data(self, index):
        # Extract video information
        vid_info = self.vid_list[index][0]
        name = vid_info.split("/")[-1].split("_x264")[0]
        
        # Prepend root_dir to the video path
        video_path = os.path.join(self.root_dir, vid_info)
        video_feature = np.load(video_path).astype(np.float32)
        # print(self.root_dir,vid_info,"  ",video_feature.shape) 
        # Determine the label (0 for normal, 1 for abnormal)
        if "Normal" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        
        # Process the video feature for training
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int32)
            print(new_feat)
            print(r)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], axis=0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        
        # Return the processed data
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        
        self.feature_path = self.data_path
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        if "_label_A" not in vid_name:
            label=1 
        # print(self.feature_path,vid_name) 
        video_feature = np.load(os.path.join(self.feature_path, vid_name )).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)

            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)

            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    
