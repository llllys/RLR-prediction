import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

from sklearn.model_selection import train_test_split

from utils.common_utils import get_all_files_in_folder, get_intersection_points, move_spin_vector_matrix


def create_integer_mapping(string_list):
    unique_strings, indices = np.unique(string_list, return_inverse=True)
    return indices

class yizhuang_dataset(Dataset):
    def __init__(self, args, mode='train'):

        self.args = args
        self.output_length = args.output_length
        
        # read data
        self.training_data = self.read_pickle_data(args)
        
        # make split
        traj_data = self.training_data['ego_motion_x']
        _, _, ind_train, ind_test = train_test_split(traj_data,
                                                     np.array(list(range(len(traj_data)))), 
                                                     test_size=args.test_split, 
                                                     shuffle=args.split_shuffle)
        if mode == 'train':
            self.ind = ind_train
        elif mode == 'test':
            self.ind = ind_test

        self.training_data['id_int'] = np.array(create_integer_mapping(self.training_data['id']))
        self.training_data.pop('id') if 'id' in self.training_data.keys() else None
        for k, v in self.training_data.items():
            self.training_data[k] = np.array(v)[self.ind]
            print('{} data with shape: {}'.format(k, self.training_data[k].shape))
        
        self.rlr_rate = self.training_data['ego_is_rlr'].mean()
        print('Red light running rate: ', self.rlr_rate)

    
    def read_pickle_data(self, args):
        '''
        data keys:
            ego_motion
            ego_motion_x
            ego_motion_y
            ego_motion_feature (scene)
            ego_rlr_label 
            env_feature
            neighbor_motion
            neighbor_mask
            anchor_point
            angle
            rotate_matrix
            m_inverse
            ego_is_rlr
            ego_is_rlr_step
            ego_ttrlr
            training_weight
            neighbor_heading
            intersection_polygon
            id_int
        '''
        data_root = args.data_root
            
        data_save_path_list = get_all_files_in_folder(data_root)
        all_train_data = []
        for i in range(len(data_save_path_list)):
            print('Reading raw data from file: ', data_save_path_list[i])
            training_data_save_path = os.path.join(data_root, data_save_path_list[i])
            with open(training_data_save_path, 'rb') as f:
                training_data_file = pickle.load(f)
                all_train_data.append(training_data_file)
        
        training_data = {}
        for i in range(len(all_train_data)):
            if i == 0:
                training_data = all_train_data[i].copy()
            else:
                for k in training_data.keys():
                    training_data[k] = np.concatenate([training_data[k], all_train_data[i][k]], axis=0)
        
        # generate stepwise is_rlr        
        delta_t = args.rlr_delta_t
        y_is_rlr_step = np.zeros(training_data['ego_rlr_label'][:, :self.output_length, 1].shape)
        y_is_rlr_step[training_data['ego_rlr_label'][:, :self.output_length, 1] < delta_t] = 1
        training_data['ego_is_rlr'] = y_is_rlr_step.max(axis=1)
        training_data['ego_is_rlr_step'] = y_is_rlr_step
        training_data['ego_ttrlr'] = training_data['ego_rlr_label'][:, :self.output_length, 1]
        training_data['ego_motion_y'] = training_data['ego_motion_y'][:, :self.output_length, 0:2]
        training_data['neighbor_motion'][training_data['neighbor_mask'] == 0] = 0
        
        # neighbor heading
        training_data['neighbor_heading'] = training_data['neighbor_motion'][:, :, :, 3] # [B, N, L]
        
        training_data['ego_motion_x'] = training_data['ego_motion_x'][:, :, 0:-1] 
        training_data['neighbor_motion'] = training_data['neighbor_motion'][:, :, :, 0:-2] 

        # ego speed diff
        speed = training_data['ego_motion_x'][:, :, 2] # [B, L]
        speed_diff = np.zeros((speed.shape[0], speed.shape[1], 1))
        speed_diff[:, 1:, 0] = np.diff(speed, axis=1)
        training_data['ego_motion_x'] = np.concatenate([training_data['ego_motion_x'], speed_diff], axis=2)
        
        # neighbor speed diff
        speed = training_data['neighbor_motion'][:, :, :, 2] # [B, N, L]
        speed_diff = np.zeros((*speed.shape, 1))
        speed_diff[:, :, 1:, 0] = np.diff(speed, axis=2)
        training_data['neighbor_motion'] = np.concatenate([training_data['neighbor_motion'], speed_diff], axis=3)
        
        ## MAP
        map_arr = get_intersection_points()
        map_data = np.expand_dims(map_arr, 0)
        map_data = np.repeat(map_data, training_data['ego_motion_x'].shape[0], axis=0)
        map_data = move_spin_vector_matrix(map_arr, training_data['anchor_point'], training_data['rotate_matrix'])
        training_data['intersection_polygon'] = map_data
                
        # regularize
        feat_to_regularize = ['env_feature']
        for key in feat_to_regularize:
            training_data[key] = training_data[key] - training_data[key].mean(axis=0) / training_data[key].std(axis=0)

        print('Length of data: ', len(training_data['ego_motion_x']))
        print('Data keys: ', training_data.keys())
        
        return training_data

     
    def __getitem__(self, idx):
        
        idx_data = {}
        for k, v in self.training_data.items():
            idx_data[k] = v[idx]
        
        return idx_data
        
        
    def __len__(self):
        return len(self.ind)
