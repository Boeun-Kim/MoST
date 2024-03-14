from einops import rearrange
import yaml
import torch
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange

# Get data information
with open('xia_dataset.yml', "r") as fd:
    cfg = yaml.load(fd, Loader=yaml.Loader)

def adv_D_loss(args, model, cnt_clip, sty_clip, sty_label, cnt_mask, sty_mask):
    # adv loss
    sty_clip.requires_grad_()
    real_disc_out = model.discriminator(sty_clip, sty_label, sty_mask)
    loss_real = adv_loss(real_disc_out, 1)
    loss_reg = r1_reg(real_disc_out, sty_clip)

    gen = model.generator(cnt_clip, sty_clip, cnt_mask, sty_mask)
    fake_disc_out = model.discriminator(gen, sty_label, cnt_mask)
    loss_fake = adv_loss(fake_disc_out, 0)

    loss = loss_real + loss_fake + args.lambda_reg*loss_reg

    loss_dict = OrderedDict([('D_loss', loss.item()),
                            ('D_real', loss_real.item()),
                            ('D_fake', loss_fake.item()),
                            ('D_reg', loss_reg.item())])
    return loss, loss_dict


def G_loss(args, model, cnt_clip, sty_clip, sty_clip2, sty_label, cnt_contact, cnt_mask, sty_mask, sty_mask2):
    posrot = cfg["joint_dims"]
    valid_token = cnt_mask[:,0,0,:]
    valid_token_sty = sty_mask[:,0,0,:]

    # adv loss
    gen = model.generator(cnt_clip, sty_clip, cnt_mask, sty_mask)
    fake_disc_out = model.discriminator(gen, sty_label, cnt_mask)
    loss_adv = adv_loss(fake_disc_out, 1)

    # reconstruction loss
    gen_recon = model.generator(cnt_clip, cnt_clip, cnt_mask, cnt_mask)
    gen_recon_valid = rearrange(gen_recon, 'b c f j -> b f c j ', )
    cnt_clip_valid = rearrange(cnt_clip, 'b c f j -> b f c j ', )
    loss_recon = torch.mean((gen_recon_valid[valid_token==True] - cnt_clip_valid[valid_token==True]).norm(dim=2))

    # cycle content consistency loss
    gen_cycle = model.generator(gen, cnt_clip, cnt_mask, cnt_mask)
    gen_cycle_valid = rearrange(gen_cycle, 'b c f j -> b f c j ', )
    cnt_clip_valid = rearrange(cnt_clip, 'b c f j -> b f c j ', )
    loss_cycle_c = torch.mean((gen_cycle_valid[valid_token==True] - cnt_clip_valid[valid_token==True]).norm(dim=2))

    # cycle style consistency loss 
    gen_cycle2 = model.generator(sty_clip, gen, sty_mask, cnt_mask)
    gen_cycle2_valid = rearrange(gen_cycle2, 'b c f j -> b f c j ', )
    sty_clip_valid = rearrange(sty_clip, 'b c f j -> b f c j ', )
    loss_cycle_s = torch.mean((gen_cycle2_valid[valid_token_sty==True] - sty_clip_valid[valid_token_sty==True]).norm(dim=2))
    

    ############  physics-based loss #############
    # velocity regularization 
    vel_gen = gen[:,:,1:,:] - gen[:,:,:-1,:]
    pad = torch.zeros(vel_gen.shape[0], vel_gen.shape[1], 1, vel_gen.shape[3]).cuda()
    vel_gen = torch.cat((vel_gen, pad), axis=2)
    vel_gen_valid = rearrange(vel_gen[:,:posrot,:,:], 'b c f j -> b f c j ', )
    reg_vel = torch.mean((vel_gen_valid[valid_token==True]).norm(dim=2))

    # acceleration regularization
    acc_gen = vel_gen[:,:,1:,:] - vel_gen[:,:,:-1,:] 
    acc_gen = torch.cat((acc_gen, pad), axis=2)
    acc_gen_valid = rearrange(acc_gen[:,:posrot, :,:], 'b c f j -> b f c j ', )
    global_acc_valid = rearrange(vel_gen[:,posrot:,:,:], 'b c f j -> b f c j ', )
    reg_acc = torch.mean((acc_gen_valid[valid_token==True]).norm(dim=2)) + torch.mean((global_acc_valid[valid_token==True]).norm(dim=2)) 

    ## foot contact regularization
    gen_cycle_foot = gen_cycle[:,:3,:,(3,4,7,8)]
    gen_cycle_foot_vel = gen_cycle_foot[:,:,1:,:] - gen_cycle_foot[:,:,:-1,:]
    gen_cycle_foot_vel_sq = torch.norm(gen_cycle_foot_vel, dim=1)
    gen_cycle_foot_vel_sq = gen_cycle_foot_vel_sq[cnt_contact[:,1:,:] == 1]
    reg_contact = torch.sum(gen_cycle_foot_vel_sq)/len(gen_cycle_foot_vel_sq)
    ##############################################

    ########### style disentanglement loss #######
    with torch.no_grad():
        gen2  = model.generator(cnt_clip, sty_clip2, cnt_mask, sty_mask2)

    gen_valid = rearrange(gen, 'b c f j -> b f c j ', )
    gen2_valid = rearrange(gen2, 'b c f j -> b f c j ', )
    loss_sty_disentangle = torch.mean((gen_valid[valid_token==True] - gen2_valid[valid_token==True]).norm(dim=2))
    ##############################################

    loss = args.lambda_adv*loss_adv + args.lambda_recon*loss_recon + args.lambda_cyc_c*loss_cycle_c + args.lambda_cyc_s*loss_cycle_s \
    + args.lambda_reg_vel*reg_vel + args.lambda_reg_acc*reg_acc + args.lambda_reg_contact*reg_contact \
    + args.lambda_sty_disentangle*loss_sty_disentangle
    
    loss_dict = OrderedDict([
                            ('G_adv', loss_adv.item()),
                            ('G_recon', loss_recon.item()),
                            ('G_cyc-c', loss_cycle_c.item()),
                            ('G_cyc-s', loss_cycle_s.item()),
                            ('G_reg_vel', reg_vel.item()),
                            ('G_reg_acc', reg_acc.item()),
                            ('G_reg_contact', reg_contact.item()),
                            ('G_loss_sty_disentangle', loss_sty_disentangle.item())
                            ])


    return loss, loss_dict


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    return loss

def r1_reg(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True, 
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    
    return reg
