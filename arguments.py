import argparse

def parse_args_train():
    parser = argparse.ArgumentParser(description='Training MoST')

    # train hyperparameters
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--total_iters', type=int, default=300000)

    # model hyperparameters
    parser.add_argument('--num_frame', type=int, default=200)
    parser.add_argument('--dim_emb', type=int, default=48)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_enc_blocks', type=int, default=2)
    parser.add_argument('--num_dec_blocks', type=int, default=3)
    parser.add_argument('--num_disc_blocks', type=int, default=1)

    parser.add_argument('--G_lr', type=float, default=1e-5, help='learning rate for G')
    parser.add_argument('--D_lr', type=float, default=1e-6, help='learning rate for D')
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=1.0) 
    parser.add_argument('--lambda_recon', type=float, default=3.0) 
    parser.add_argument('--lambda_cyc_c', type=float, default=3.0)
    parser.add_argument('--lambda_cyc_s', type=float, default=3.0) 
    parser.add_argument('--lambda_sty_disentangle', type=float, default=1.0)  
    parser.add_argument('--lambda_reg_vel', type=float, default=1.0)  
    parser.add_argument('--lambda_reg_acc', type=float, default=0.1)
    parser.add_argument('--lambda_reg_contact', type=float, default=1.0)

    # paths
    parser.add_argument('--train_datapath', type=str, default='data/preprocessed_xia/train.npz', help='train data path')
    parser.add_argument('--dist_datapath', type=str, default='data/preprocessed_xia/distribution.npz', help='data mean and std path')
    parser.add_argument('--save_path', type=str, default='results')
    args = parser.parse_args()

    return args


def parse_args_test():
    parser = argparse.ArgumentParser(description='Testing MoST')

    # test hyperparameters
    parser.add_argument('--gpus', type=int, default=1)

    # model hyperparameters
    parser.add_argument('--num_frame', type=int, default=200)
    parser.add_argument('--dim_emb', type=int, default=48)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_enc_blocks', type=int, default=2)
    parser.add_argument('--num_dec_blocks', type=int, default=3)
    parser.add_argument('--num_disc_blocks', type=int, default=1)

    # paths
    parser.add_argument('--model_path', type=str, default='pretrained/xia_pretrained.pth')
    parser.add_argument('--dist_datapath', type=str, default='data/preprocessed_xia/distribution.npz', help='data mean and std path')
    parser.add_argument('--demo_datapath', type=str, default='data/preprocessed_xia_test', help='demo data path')
    parser.add_argument('--cnt_clip', type=str, default='angry_13_000', help='input content clip name')
    parser.add_argument('--sty_clip', type=str, default='strutting_16_000', help='input style clip name')
    
    args = parser.parse_args()

    return args