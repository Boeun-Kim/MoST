from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from model.transformer import TransformerEncoder, TransformerDecoder, TransformerModulator

num_bodypart = 6

LLeg_idx = [1, 2, 3, 4]
RLeg_idx = [5, 6, 7, 8]
Trunk_idx = [9, 10, 11, 12]
LArm_idx = [13, 14, 15, 16]
RArm_idx = [17, 18, 19, 20]
Root_idx = [0]

bodypart_idx = []
bodypart_idx.append(LLeg_idx)
bodypart_idx.append(RLeg_idx)
bodypart_idx.append(Trunk_idx)
bodypart_idx.append(LArm_idx)
bodypart_idx.append(RArm_idx)
bodypart_idx.append(Root_idx)

class StyleTransformer(nn.Module):
    def __init__(self, cfg, num_frame, dim_emb, num_heads, num_enc_blocks, num_dec_blocks):
        super().__init__()
        
        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_part=num_bodypart, num_frame=num_frame+1, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(num_enc_blocks)]
        )
        self.encoder_IN = nn.ModuleList(
            [TransformerEncoder(num_part=num_bodypart, num_frame=num_frame, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(1)]
        )

        self.PSM = TransformerModulator(dim_emb, num_heads, num_bodypart)

        self.decoder = nn.ModuleList(
            [TransformerDecoder(num_part=num_bodypart, num_frame=num_frame, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(num_dec_blocks)]
        )

        self.learnable_sty_token = nn.Parameter(torch.randn(num_bodypart, dim_emb)).cuda()
        
        self.d = cfg["joint_dims"]  # pos+rot (positions:3, rotations:4, root_trajectory:4)
        self.part_emb = nn.ModuleList([
            nn.Linear(len(bodypart_idx[i])*self.d, dim_emb)
            for i in range(num_bodypart-1)]) 
        self.part_emb.append(nn.Linear(self.d, int(dim_emb/2))) # Root pos+rot
        self.part_emb.append(nn.Linear(4, int(dim_emb/2))) # Root vel+rvel

        self.part_upsample = nn.ModuleList([
            nn.Linear(dim_emb, len(bodypart_idx[i])*self.d)
             for i in range(num_bodypart-1)])
        self.part_upsample.append(nn.Linear(dim_emb, self.d))
        self.part_upsample.append(nn.Linear(dim_emb, 4))

        self.dropout_cnt1 = nn.Dropout(p=0.1)
        self.dropout_sty1 = nn.Dropout(p=0.1)
        self.dropout_cnt2 = nn.Dropout(p=0.1)
        self.dropout_sty2 = nn.Dropout(p=0.1)

        
    def forward(self, cnt, sty, cnt_mask, sty_mask):
        
        # Body-part and global translation embedding
        motion = [cnt, sty]
        motion_embs = []

        for i in range(2):
            # LLeg, RLeg, Trunk, LArm, RArm
            part = []
            for j in range(0, num_bodypart-1):
                part.append(motion[i][:, :self.d, :, bodypart_idx[j]])
            Root_pos = motion[i][:, :self.d, :, Root_idx]
            part.append(Root_pos)
            traj = motion[i][:,self.d:, :, [0]] 
            part.append(traj)
         
            part_emb = []
            for j, p_ in enumerate(part):
                p = rearrange(p_, 'b c f p  -> b f (c p)', )
                part_emb.append(self.part_emb[j](p).unsqueeze(2))
  
            part_emb[-2] = torch.cat((part_emb[-2], part_emb[-1]), axis=-1)
            del part_emb[-1]

            motion_emb = part_emb[0]
            for j in range(1, len(part_emb)):
                motion_emb = torch.cat((motion_emb, part_emb[j]), axis=2)
            
            motion_embs.append(motion_emb)
  
        ######### Process of the content motion #########
        # Add learnabel style token
        m_cnt = motion_embs[0]
        learnable_sty_token = self.learnable_sty_token.unsqueeze(0).unsqueeze(0)
        learnable_sty_token = learnable_sty_token.repeat(m_cnt.shape[0], 1, 1, 1)
        m_cnt_ = torch.cat((learnable_sty_token, m_cnt), axis=1)

        # Generate mask for attention
        m_cnt_ = self.dropout_cnt1(m_cnt_)
        cnt_mask_ = torch.cat((cnt_mask[:,:,0,:].unsqueeze(2), cnt_mask), axis=2)
        cnt_mask_ = torch.cat((cnt_mask_[:,:,:,0].unsqueeze(3), cnt_mask_), axis=3)

        # Encode content motion
        for i, block in enumerate(self.encoder):
            m_cnt_ = block(m_cnt_, cnt_mask_, last_block=False)

            # Last encoder block with IN
        cnt_of_content_motion = m_cnt_[:,1:,:,:]
        for i, block in enumerate(self.encoder_IN):
            cnt_of_content_motion = block(cnt_of_content_motion, cnt_mask, last_block=True)

        # Pool content dynamics feater(Y^C) to generate C^C
        cnt_of_content_motion_ = rearrange(cnt_of_content_motion, 'b f p c -> b (p c) f')
        tm_pooling = nn.AvgPool1d(cnt_of_content_motion_.shape[-1])
        pool_cnt_of_content_motion = tm_pooling(cnt_of_content_motion_)
        pool_cnt_of_content_motion = rearrange(pool_cnt_of_content_motion.squeeze(-1), 'b (p c) -> b p c', p=num_bodypart)
        ###############################################

        ######### Process of the style motion #########
        # Add learnable token
        m_sty = motion_embs[1]
        m_sty_ = torch.cat((learnable_sty_token, m_sty), axis=1)
        m_sty_ = self.dropout_sty1(m_sty_)

        # Generate mask for attention
        sty_mask_ = torch.cat((sty_mask[:,:,0,:].unsqueeze(2), sty_mask), axis=2)
        sty_mask_ = torch.cat((sty_mask_[:,:,:,0].unsqueeze(3), sty_mask_), axis=3)

        # Encode style motion
        for i, block in enumerate(self.encoder):
            m_sty_ = block(m_sty_, sty_mask_, last_block=False)

            # Extract style features before last block
        sty_of_style_motion = m_sty_[:,0,:,:]

            # Last encoder block with IN
        cnt_of_style_motion = m_sty_[:,1:,:,:]
        for i, block in enumerate(self.encoder_IN):
            cnt_of_style_motion = block(cnt_of_style_motion, sty_mask, last_block=True)

        # Pool content dynamics feater(Y^S) to generate C^S
        cnt_of_style_motion = rearrange(cnt_of_style_motion, 'b f p c -> b (p c) f')
        tm_pooling = nn.AvgPool1d(cnt_of_style_motion.shape[-1])
        pool_cnt_of_style_motion = tm_pooling(cnt_of_style_motion)
        pool_cnt_of_style_motion = rearrange(pool_cnt_of_style_motion.squeeze(-1), 'b (p c) -> b p c', p=num_bodypart)

        ###############################################

        ################ Process of PSM ###############
        modulated_sty = self.PSM(pool_cnt_of_content_motion, pool_cnt_of_style_motion, sty_of_style_motion, 0) 
        ###############################################

        # Generator
        m_gen = self.dropout_cnt2(cnt_of_content_motion)
        for i, block in enumerate(self.decoder):
            m_gen = block(m_gen, modulated_sty, cnt_mask=cnt_mask)  # b f p c 

        # Part to joints                
        gen_part = []
        for i in range(m_gen.shape[2]):
            if i == 5:
                num_joint = 1 # root
            else:
                num_joint = 4
            x = self.part_upsample[i](m_gen[:,:,i,:])
            x = rearrange(x, 'b f (j c) -> b f j c', j=num_joint)
            gen_part.append(x)

        x = self.part_upsample[-1](m_gen[:,:,-1,:]).unsqueeze(2)
        x = x.expand(-1,-1, cnt.shape[3],-1)
        gen_part.append(x)

        gen_body = []
        gen_body.append(gen_part[5]) # 0
        gen_body.append(gen_part[0]) # 1,2,3,4
        gen_body.append(gen_part[1]) # 5,6,7,8
        gen_body.append(gen_part[2]) # 9,10,11,12
        gen_body.append(gen_part[3]) # 13,14,15,16
        gen_body.append(gen_part[4]) # 17,18,19,20
        gen_body = torch.cat(gen_body, axis=2)
        
        gen_body = torch.cat((gen_body, gen_part[-1]), axis=-1)

        gen_motion = rearrange(gen_body, 'b f j c -> b c f j', )

        return gen_motion


class Discriminator(nn.Module):
    def __init__(self, cfg, num_frame, dim_emb, num_heads, num_enc_blocks):
        super().__init__()

        self.learnable_token = nn.Parameter(torch.randn(num_bodypart+1, dim_emb))

        self.encoder = nn.ModuleList(
            [TransformerEncoder(num_part=num_bodypart+1, num_frame=num_frame+1, dim_emb=dim_emb, num_heads=num_heads)
            for i in range(num_enc_blocks)]
        )

        self.d = cfg["joint_dims"]  # pos+rot (positions:3, rotations:4, root_trajectory:4)
        self.part_emb = nn.ModuleList([
            nn.Linear(len(bodypart_idx[i])*self.d, dim_emb)
            for i in range(num_bodypart-1)]) 
        self.part_emb.append(nn.Linear(self.d, dim_emb)) # pos+rot+traj
        self.part_emb.append(nn.Linear(4, dim_emb)) #traj

        self.dropout = nn.Dropout(p=0.1)
        num_style_cat = len(cfg["style_names"])
        self.head = nn.Sequential(
            nn.Linear(dim_emb*(num_bodypart+1), dim_emb),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_emb, num_style_cat)
        )
        
    def forward(self, motion, style_label, mask):
        # LLeg, RLeg, Trunk, LArm, RArm
        part = []
        for j in range(0, num_bodypart-1):
            part.append(motion[:, :self.d, :, bodypart_idx[j]])
        Root_pos = motion[:, :self.d, :, Root_idx]
        part.append(Root_pos)
        traj = motion[:,self.d:, :, [0]]  # b,4,200
        part.append(traj)
        
        part_emb = []
        for j, p_ in enumerate(part):
            p = rearrange(p_, 'b d f p  -> b f (d p)', )
            part_emb.append(self.part_emb[j](p).unsqueeze(2))

        motion_emb = part_emb[0]
        for j in range(1, len(part)):
            motion_emb = torch.cat((motion_emb, part_emb[j]), axis=2)

        learnable_token = self.learnable_token.unsqueeze(0).unsqueeze(0)
        learnable_token = learnable_token.repeat(motion_emb.shape[0], 1, 1, 1)
        motion_emb = torch.cat((learnable_token, motion_emb), axis=1)

        # mask extension
        mask = torch.cat((mask[:,:,0,:].unsqueeze(2), mask), axis=2)
        mask = torch.cat((mask[:,:,:,-1].unsqueeze(3), mask), axis=3)
            
        # transformer encoder blocks
        motion_emb = self.dropout(motion_emb)
        for i, block in enumerate(self.encoder):
            motion_emb = block(motion_emb, mask)

        class_token = motion_emb[:,0,:,:]

        class_token = rearrange(class_token, 'b p c  -> b (p c)', )
        out = self.head(class_token) 

        out = out.view(out.shape[0], -1)
        idx = range(style_label.size(0))

        out = out[idx, style_label]
        
        return out