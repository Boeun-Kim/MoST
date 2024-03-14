import numpy as np
import yaml

from data.animation.InverseKinematics import JacobianInverseKinematics
from data.animation.Quaternions import Quaternions
from data.animation import BVH


class SaveBVH:
    def __init__(self, args):

        preprocess = np.load(args.dist_datapath)
        self.Xmean = preprocess['Xmean']
        self.Xstd = preprocess['Xstd']

    def save_output(self, output, traj, filename='output.bvh'):

        output = denormalize(output, self.Xmean[:7], self.Xstd[:7])
        output = np.transpose(output, (1, 2, 0))

        traj = denormalize(traj, self.Xmean[-4:], self.Xstd[-4:])
        traj = np.transpose(traj, (1, 2, 0))

        # original output
        positions = restore_animation(output[:, :, :3], traj)

        print('Saving animation of %s in bvh...' % filename)
        to_bvh_cmu(positions, filename=filename, frametime=1.0/30.0)


def restore_animation(pos, traj, start=None, end=None):
    """
    :param pos: (F, J, 3)
    :param traj: (F, J, 4)
    :param start: start frame index
    :param end: end frame index
    :return: positions
    """
    if start is None:
        start = 0
    if end is None:
        end = len(pos)

    Rx = traj[start:end, 0, -4]
    Ry = traj[start:end, 0, -3]
    Rz = traj[start:end, 0, -2]
    Rr = traj[start:end, 0, -1]

    rotation = Quaternions.id(1)
    translation = np.array([[0, 0, 0]])

    for fi in range(len(pos)):
        pos[fi, :, :] = rotation * pos[fi]
        pos[fi] = pos[fi] + translation[0]  # NOTE: xyz-translation
        rotation = Quaternions.from_angle_axis(-Rr[fi], np.array([0, 1, 0])) * rotation
        translation = translation + rotation * np.array([Rx[fi], Ry[fi], Rz[fi]])
    global_positions = pos

    return global_positions


def to_bvh_cmu(targets, filename, silent=True, frametime=1.0/60.0):
    """
    from 21 to 31 joints
    """
    rest, names, _ = BVH.load('data/rest_cmu.bvh')
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)

    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across1 = targets[:, hip_l] - targets[:, hip_r]
    across0 = targets[:, sdr_l] - targets[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[...,np.newaxis]

    forward = np.cross(across, np.array([[0,1,0]]))
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

    anim.positions[:,0] = targets[:,0]
    anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]

    mapping = {
        0: 0,
        2: 1, 3: 2, 4: 3, 5: 4,
        7: 5, 8: 6, 9: 7, 10: 8,
        12: 9, 13: 10, 15: 11, 16: 12,
        18: 13, 19: 14, 20: 15, 22: 16,
        25: 17, 26: 18, 27: 19, 29: 20,
    }

    targetmap = {}
    for k in mapping:
        targetmap[k] = targets[:, mapping[k]]

    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=silent)
    ik()

    BVH.save(filename, anim, names, frametime=frametime)


def denormalize(x, mean, std):
        x = x * std + mean
        return x
