from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
import os

from model.networks import StyleTransformer, Discriminator

class MST(nn.Module):
    def __init__(self, istrain, cfg, args):
        super().__init__()

        self.args = args
        self.generator = StyleTransformer(cfg, num_frame=args.num_frame, dim_emb=args.dim_emb, num_heads=args.num_heads,
                            num_enc_blocks=args.num_enc_blocks, num_dec_blocks=args.num_dec_blocks)
        self.discriminator = Discriminator(cfg, num_frame=args.num_frame, dim_emb=args.dim_emb, num_heads=args.num_heads,
                            num_enc_blocks=args.num_disc_blocks)

        self.istrain=istrain
        if istrain:
            self.optimizer_D = torch.optim.Adam(
                        params=self.discriminator.parameters(),
                        lr=self.args.D_lr, betas=(0.9, 0.99), weight_decay=1e-4)
            self.optimizer_G = torch.optim.Adam(
                        params=self.generator.parameters(),
                        lr=self.args.G_lr, betas=(0.9, 0.99), weight_decay=1e-4)
                        
            self.current_iter = 0

        
    def save_checkpoint(self, latest=False):
        os.makedirs(self.args.save_path, exist_ok=True)
        if latest:
            output_path = self.args.save_path + "/latest.pth"
        else:
            output_path = self.args.save_path + "/%d.pth" % self.current_iter

        print('Saving the model into %s...' % output_path)

        checkpoint = {}
        checkpoint['iter'] = self.current_iter
        checkpoint['generator'] = self.generator.state_dict()
        checkpoint['discriminator'] = self.discriminator.state_dict()
        checkpoint['optimizer_G'] = self.optimizer_G.state_dict()
        checkpoint['optimizer_D'] = self.optimizer_D.state_dict()

        torch.save(checkpoint, output_path)
        

    def load_checkpoint(self):
        load_path = self.args.model_path

        checkpoint = torch.load(load_path, map_location='cuda:0')
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

        if self.istrain:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            self.current_iter = checkpoint['iter']


