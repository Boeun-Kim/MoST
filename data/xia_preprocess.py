'''
NOTE: Some functions in this file are from https://github.com/soomean/Diverse-Motion-Stylization
'''

import os
import sys
import numpy as np
import yaml
import scipy.ndimage as ndi
import shutil
from os.path import join as pjoin
sys.path.append('../')
from data.animation import BVH, Animation
from data.animation.Quaternions import Quaternions
from data.animation.Pivots import Pivots


def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']


def feet_contact_from_positions(positions, fid_l=(3, 4), fid_r=(7, 8)):
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    velfactor = np.array([0.05, 0.05])
    feet_contact = []
    for fid_index in [fid_l, fid_r]:
        foot_vel = (positions[1:, fid_index] - positions[:-1, fid_index]) ** 2
        foot_vel = np.sum(foot_vel, axis=-1)
        foot_contact = (foot_vel < velfactor).astype(float)
        feet_contact.append(foot_contact)
    feet_contact = np.concatenate(feet_contact, axis=-1)

    feet_contact = np.concatenate((feet_contact[0:1].copy(), feet_contact), axis=0)

    return feet_contact


def preprocess(filename, selected_joints, downsample=2, slice=False, window=64, window_step=32, njoints=21):
    
    anim, names, frametime = BVH.load(filename)
    anim = anim[::downsample] 

    global_xforms = Animation.transforms_global(anim)  
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)

    global_positions = global_positions[:, selected_joints]
    global_rotations = global_rotations[:, selected_joints]

    clip, feet = get_motion_data(global_positions, global_rotations, njoints)

    if not slice:
        return clip, feet
    else:
        clip_windows = []
        feet_windows = []
        
        for j in range(0, len(clip) - window // 8, window_step):
            assert (len(global_positions) >= window // 8)
            
            clip_slice = clip[j:j + window]
            clip_feet = feet[j:j + window]

            if len(clip_slice) < window:
                # left slices
                clip_left = clip_slice[:1].repeat((window - len(clip_slice)) // 2 + (window - len(clip_slice)) % 2, axis=0)
                clip_left[:, :, -4:] = 0.0
                clip_feet_l = clip_feet[:1].repeat((window - len(clip_slice)) // 2 + (window - len(clip_slice)) % 2, axis=0)
                # right slices
                clip_right = clip_slice[-1:].repeat((window - len(clip_slice)) // 2, axis=0)
                clip_right[:, :, -4:] = 0.0
                clip_feet_r = clip_feet[-1:].repeat((window - len(clip_slice)) // 2, axis=0)
                # padding
                clip_slice = np.concatenate([clip_left, clip_slice, clip_right], axis=0)
                clip_feet = np.concatenate([clip_feet_l, clip_feet, clip_feet_r], axis=0)
            if len(clip_slice) != window: raise Exception()
            if len(clip_feet) != window: raise Exception()

            clip_windows.append(clip_slice)
            feet_windows.append(clip_feet)

        
        return clip_windows, feet_windows


def get_motion_data(global_positions, global_rotations, njoints):
    # extract forward direction
    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across = ((global_positions[:, sdr_l] - global_positions[:, sdr_r]) + (global_positions[:, hip_l] - global_positions[:, hip_r]))
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]  # (F, 3)

    # smooth forward direction
    direction_filterwidth = 20
    forward = ndi.gaussian_filter1d(np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    # remove translation & rotation
    root_rotation = Quaternions.between(forward, np.array([[0, 0, 1]]).repeat(len(forward), axis=0))[:, np.newaxis]
    positions = global_positions.copy()
    rotations = global_rotations.copy()

    positions[:, :, 0] = positions[:, :, 0] - positions[:, 0:1, 0]
    positions[:, :, 1] = positions[:, :, 1] - positions[:, 0:1, 1] + positions[0:1, 0:1, 1]
    positions[:, :, 2] = positions[:, :, 2] - positions[:, 0:1, 2]
    positions = root_rotation * positions
    rotations = root_rotation * rotations

    # trajectory info
    root_velocity = root_rotation[:-1] * (global_positions[1:, 0:1] - global_positions[:-1, 0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps
    root_velocity = root_velocity.repeat(njoints, axis=1)
    root_rvelocity = root_rvelocity.repeat(njoints, axis=1)[..., np.newaxis]

    # motion clip info
    positions = positions[:-1]
    rotations = rotations[:-1]
    root_trajectory = np.concatenate([root_velocity, root_rvelocity], axis=-1)
    motion_clip = np.concatenate([positions, rotations, root_trajectory], axis=-1)

    # feet contact info """
    motion_feet = feet_contact_from_positions(positions)

    return motion_clip, motion_feet


def generate_data(filename, selected_joints,  njoints=21, downsample=1):
    clip, feet = preprocess(filename, selected_joints, slice=False, 
                                downsample=downsample, njoints=njoints)                  
    max_frame=200
    clip_expand = np.empty((max_frame, clip.shape[1], clip.shape[2]))
    feet_expand = np.empty((max_frame, feet.shape[1]))
    clip_expand[:] = np.nan
    feet_expand[:] = np.nan
    ori_length  = min(max_frame, clip.shape[0])
    clip_expand[:ori_length] = clip[:ori_length]
    feet_expand[:ori_length] = feet[:ori_length]

    clip_expand = np.transpose(clip_expand, (2, 0, 1))  # (C, F, J)
    
    return clip_expand, feet_expand


def set_init(dic, key, value):
    try:
        dic[key]
    except KeyError:
        dic[key] = value

def generate_dataset(data_dir, out_path, downsample=2, max_frame=200):

    with open('../xia_dataset.yml', "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    content_names = cfg["content_names"]
    style_names = cfg["style_names"]
    njoints = cfg["njoints"]
    selected_joints = cfg["selected_joints"]

    content_full_namedict = [full_name.split('_')[0] for full_name in cfg["content_full_names"]]
    content_test_count = cfg["content_test_count"]
    
    style_name_to_idx = {name: i for i, name in enumerate(style_names)}
    content_name_to_idx = {name: i for i, name in enumerate(content_names)}
    
    bvh_files = get_bvh_files(data_dir)

    train_clip = []
    train_feet = []
    train_label_cnt = []
    train_label_sty = []

    gt_files = []
    test_files = []

    test_cnt = {}

    for i, item in enumerate(bvh_files):
        print('Processing %i of %i (%s)' % (i, len(bvh_files), item))
        filename = item.split('/')[-1]
        style, content_num, _ = filename.split('_')
    
        content = content_full_namedict[int(content_num) - 1]
        content_style = "%s_%s" % (content, style)

        content_idx = content_name_to_idx[content]
        style_idx = style_name_to_idx[style]

        set_init(test_cnt, content_style, 0)

        if test_cnt[content_style] < content_test_count[content]:
            test_cnt[content_style] += 1
            test_files.append(filename)

        else:
            clip, feet = preprocess(item, selected_joints, downsample=downsample, njoints=njoints)       

            clip_expand = np.empty((max_frame, clip.shape[1], clip.shape[2]))
            feet_expand = np.empty((max_frame, feet.shape[1]))
            clip_expand[:] = np.nan
            feet_expand[:] = np.nan
            ori_length  = min(max_frame, clip.shape[0])
            clip_expand[:ori_length] = clip[:ori_length]
            feet_expand[:ori_length] = feet[:ori_length]
            clip = [clip_expand]
            feet = [feet_expand]
            
            train_clip += clip
            train_feet += feet
            train_label_cnt.append(content_idx) 
            train_label_sty.append(style_idx)

            gt_files.append(filename)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    train_clip = np.array(train_clip)
    train_clip = np.transpose(train_clip, (0, 3, 1, 2))
    train_feet = np.array(train_feet)
    np.savez_compressed(out_path+'/train', clips=train_clip, feet=train_feet, cnt=train_label_cnt, sty=train_label_sty)

    test_folder = out_path + "_test"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    for file in test_files:
        shutil.copy(pjoin(data_dir, file), pjoin(test_folder, file))

    gt_folder = out_path + "_gt"
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)
    for file in gt_files:
        shutil.copy(pjoin(data_dir, file), pjoin(gt_folder, file))


def generate_mean_std(dataset_path, out_path):
    X = np.load(dataset_path)['clips']
        
    print('Total shape: ', X.shape)  # (N, C, F, J)
    X = X[:, :-4, :, :]  # (N, 7, F, J)
    Xmean = np.nanmean(X, axis=(0,2), keepdims=True)[0]
    Xmean = np.concatenate([Xmean, np.zeros((4,) + Xmean.shape[1:])])
    Xstd = np.nanstd(X, axis=(0,2), keepdims=True)[0]
    idx = Xstd < 1e-5
    Xstd[idx] = 1
    Xstd = np.concatenate([Xstd, np.ones((4,) + Xstd.shape[1:])])

    print('Mean shape', Xmean.shape)
    print('Std shape: ', Xstd.shape)
    np.savez_compressed(out_path, Xmean=Xmean, Xstd=Xstd)



if __name__ == '__main__':

    generate_dataset('mocap_xia', 'preprocessed_xia', downsample=2, max_frame=200)
    generate_mean_std('preprocessed_xia/train.npz', 'preprocessed_xia/distribution')

    print('done!')
