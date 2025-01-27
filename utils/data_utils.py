import numpy as np
import torch

def get_voxels_num(sub):
    if sub ==0:
        num_voxels = 15724
    elif sub == 1:
        num_voxels = 15724
    elif sub == 2:
        num_voxels = 14278
    elif sub == 3:
        num_voxels = 15226
    elif sub == 4:
        num_voxels = 13153
    elif sub == 5:
        num_voxels = 13039
    elif sub == 6:
        num_voxels = 17907
    elif sub == 7:
        num_voxels = 12682
    elif sub == 8:
        num_voxels = 14386

    return num_voxels


def read_data(client, training_type):

    def load_and_normalize_data(fmri_path):
        fmri_data = np.load(fmri_path)
        norm_mean = np.mean(fmri_data, axis=0)
        norm_scale = np.std(fmri_data, axis=0, ddof=1)
        return ((fmri_data - norm_mean) / norm_scale)

    def get_clip_path(training_type, client):
        if training_type == 'vision':
            return '/data/NSD/data/extracted_features/subj{:02d}/nsd_clipvision_noavg_train.npy'.format(client), '/data/NSD/data/extracted_features/subj{:02d}/nsd_clipvision_test.npy'.format(client)
        elif training_type == 'text':
            return '/data/NSD/data/extracted_features/subj{:02d}/nsd_cliptext_noavg_train.npy'.format(client), '/data/NSD/data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(client)

    train_fmri_path = '/data/NSD/data/processed_data/subj{:02d}/nsd_train_fmrinoavg_nsdgeneral_sub{}.npy'.format(client, client)
    test_fmri_path = '/data/NSD/data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(client, client)

    train_clip_path, test_clip_path = get_clip_path(training_type, client)
    clip_data = np.load(train_clip_path)
    clip_data_test = np.load(test_clip_path)

    return (load_and_normalize_data(train_fmri_path), clip_data, 
            load_and_normalize_data(test_fmri_path), clip_data_test)

def read_client_data(client, training_type):
    
    frmi_data, latent_data, frmi_data_test, latent_data_test = read_data(client, training_type)
    frmi_data = torch.Tensor(frmi_data).type(torch.float32)
    latent_data = torch.Tensor(latent_data).type(torch.float32)

    train_data = [(x, y) for x, y in zip(frmi_data, latent_data)]
    
    frmi_data_test = torch.Tensor(frmi_data_test).type(torch.float32)
    latent_data_test = torch.Tensor(latent_data_test).type(torch.float32)

    test_data = [(x, y) for x, y in zip(frmi_data_test, latent_data_test)]

    return train_data, test_data
        

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

