import torch
import numpy as np
import random
import yaml
import os
from data.xia_preprocess import generate_data
from arguments import parse_args_test
from model.mst import MST

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']

def normalize(x, mean, std):
    x = (x - mean) / std
    return x
    
def denormalize(x, mean, std):
    x = x * std + mean
    return x

if __name__ == '__main__':

    eval_datapath = 'data/preprocessed_xia_test'
    gt_datapath = 'data/preprocessed_xia_gt'
    args = parse_args_test()

    with open('xia_dataset.yml', "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Load model
    model = MST(False, cfg, args)
    model = model.to(device)
    model.load_checkpoint()
    model.eval()

    data_dist = np.load(args.dist_datapath)
    Xmean = data_dist['Xmean']
    Xstd = data_dist['Xstd']

    gt_bvh_files = get_bvh_files(gt_datapath)
    bvh_files = get_bvh_files(eval_datapath)
    content_full_namedict = [full_name.split('_')[0] for full_name in cfg["content_full_names"]]

    euc_dist_sum = [0, 0]
    num_test = [0, 0]

    for i, item in enumerate(bvh_files):
        filename = item.split('/')[-1]
        style, content_num, _ = filename.split('_')
        content = content_full_namedict[int(content_num) - 1]
        print('.')

        for i_ref, item_ref in enumerate(bvh_files):
            filename_ref = item_ref.split('/')[-1]
            style_ref, content_num_ref, _ = filename_ref.split('_')
            content_ref = content_full_namedict[int(content_num_ref) - 1]
            
            cnt_path = eval_datapath+'/'+ filename
            sty_path = eval_datapath+'/'+ filename_ref

            cnt_clip_raw, _ = generate_data(cnt_path, selected_joints=cfg["selected_joints"], njoints=cfg["njoints"], downsample=2)
            sty_clip_raw, _ = generate_data(sty_path, selected_joints=cfg["selected_joints"], njoints=cfg["njoints"], downsample=2)

            cnt_clip = normalize(cnt_clip_raw, Xmean, Xstd)
            sty_clip = normalize(sty_clip_raw, Xmean, Xstd)

            cnt_clip = torch.tensor(cnt_clip, dtype=torch.float).unsqueeze(0).cuda()
            sty_clip = torch.tensor(sty_clip, dtype=torch.float).unsqueeze(0).cuda()

            # Generate temporal mask for the motion sequences & change nan to 0.0
            cnt_m  = cnt_clip[:,1,:,0]
            cnt_length  = sum(~torch.isnan(cnt_m[0])).cpu().numpy()
            cnt_mask = ~torch.isnan(cnt_m).unsqueeze(1).repeat(1, cnt_m.size(1), 1).unsqueeze(1)
            cnt_clip[torch.isnan(cnt_clip)] = 0.0
            sty_m  = sty_clip[:,1,:,0]
            sty_mask = ~torch.isnan(sty_m).unsqueeze(1).repeat(1, sty_m.size(1), 1).unsqueeze(1)
            sty_clip[torch.isnan(sty_clip)] = 0.0

            # Perform style transfer
            gen = model.generator(cnt_clip, sty_clip, cnt_mask, sty_mask)
            # Our model generates global translation & local pose sequence
            gen_body = gen[0, :cfg["joint_dims"], :cnt_length,:].cpu().detach().numpy()
            gen_traj = gen[0,cfg["joint_dims"]:,:cnt_length,:].cpu().detach().numpy()
  

            # Select gt motions
            gt_set = []
            for i_gt, item_gt in enumerate(gt_bvh_files):
                filename_gt = item_gt.split('/')[-1]
                style_gt, content_num_gt, _ = filename_gt.split('_')
                content_gt = content_full_namedict[int(content_num_gt) - 1]
                if content_gt == content and style_gt == style_ref:
                    gt_set.append(filename_gt)

            # Calculate Style Consistency ++ (SC++)
            euc_dist = 0
            for gt_file in gt_set:

                gt_path = gt_datapath+'/'+ gt_file
                gt_clip_raw, _ = generate_data(gt_path, selected_joints=cfg["selected_joints"], njoints=cfg["njoints"], downsample=2)
                gt_clip = normalize(gt_clip_raw, Xmean, Xstd)
                gt_clip = torch.tensor(gt_clip, dtype=torch.float).unsqueeze(0).cuda()
                gt_m  = gt_clip[:,1,:,0]
                gt_length  = sum(~torch.isnan(gt_m[0])).cpu().numpy()
                gt_mask = ~torch.isnan(gt_m).unsqueeze(1).repeat(1, gt_m.size(1), 1).unsqueeze(1)
                gt_clip[torch.isnan(gt_clip)] = 0.0
                
                eval_length = min(gt_length, cnt_length)
                
                gt = gt_clip_raw[:3,:eval_length,:]
                gen_body_denorm = denormalize(gen_body, Xmean[:7], Xstd[:7])
                pred = gen_body_denorm[:3,:eval_length,:]

                euc_dist += np.sum(np.linalg.norm(gt-pred, axis=0), axis=(0,1)) / eval_length

            euc_dist = euc_dist/len(gt_set)

            if (content_ref == content):
                euc_dist_sum[0] += euc_dist
                num_test[0] += 1
            else:
                euc_dist_sum[1] += euc_dist
                num_test[1] += 1
            
    print('SC++_same_cnt: ', euc_dist_sum[0]/num_test[0])
    print('SC++_diff_cnt: ', euc_dist_sum[1]/num_test[1])
    print('SC++_total: ', np.sum(euc_dist_sum)/np.sum(num_test))
