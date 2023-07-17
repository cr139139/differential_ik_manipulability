import pybullet as pb
import pybullet_data
import time
from tqdm import tqdm
import numpy as np
import json
pbId = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())


def getJointStates(robot):
    joint_states = pb.getJointStates(robot, range(pb.getNumJoints(robot)))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def getMotorJointStates(robot):
    joint_states = pb.getJointStates(robot, range(pb.getNumJoints(robot)))
    joint_infos = [pb.getJointInfo(robot, i) for i in range(pb.getNumJoints(robot))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_limits = [[i[8], i[9]] for i in joint_infos if i[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques, joint_limits


# pb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
robotId = pb.loadURDF('/robots/iiwa7_base.urdf', [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                      flags=pb.URDF_USE_SELF_COLLISION)
eeId = 7

from itertools import combinations, product

allPairs = combinations(range(-1, 7, 1), 2)
collisionPairs = [i for i in allPairs if abs(i[0] - i[1]) != 1]
orthants = np.array([list(i) for i in product([0, 1], repeat=7)])
orthants = orthants[:, np.newaxis, :].repeat(6, axis=1)

_, _, _, mlim = getMotorJointStates(robotId)
mlim = np.array(mlim)
search_space = np.linspace(mlim[:, 0], mlim[:, 1], num=7).T
search_space = np.array(np.meshgrid(search_space[0], search_space[1], search_space[2], search_space[3],
                                    search_space[4], search_space[5], search_space[6])).T.reshape(-1, 7)

# initPose = [1.0991564409436654, 1.916140952214074, 0.08419705568139653, -2.0943954035214083, -0.3960754155037616,
#             1.3159501804347722, -0.04148854462260377, 0.0]

logger = open("log.txt", "w")

for joint_config in tqdm(search_space):
    for i, v in enumerate(joint_config):
        pb.resetJointState(robotId, i, v)

    # pb.stepSimulation()
    pb.performCollisionDetection()

    distances = []
    for a, b in collisionPairs:
        closest_points = pb.getClosestPoints(
            robotId,
            robotId,
            distance=2,
            linkIndexA=a,
            linkIndexB=b,
            physicsClientId=pbId,
        )
        if len(closest_points) == 0:
            distances.append(2)
        else:
            distances.append(np.min([pt[8] for pt in closest_points]))
    distances = np.array(distances)
    if np.min(distances) <= 0:
        self_collision = 1
    else:
        self_collision = 0

    if not self_collision:
        mpos, mvel, mtorq, _ = getMotorJointStates(robotId)
        result = pb.getLinkState(robotId,
                                 eeId,
                                 computeLinkVelocity=1,
                                 computeForwardKinematics=1)
        zero_vec = [0.0] * len(mpos)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        jac_t, jac_r = pb.calculateJacobian(robotId, eeId, link_trn, mpos, zero_vec, zero_vec)
        J = np.concatenate([np.array(jac_t), np.array(jac_r)], axis=0)

        _, _, _, _, ee_pos, _ = pb.getLinkState(robotId, eeId)

        mpos = np.array(mpos)
        h_grad = (mlim[:, 1] - mlim[:, 0]) ** 2 * (2 * mpos - mlim[:, 0] - mlim[:, 1]) \
                 / (4 * (mpos - mlim[:, 0]) ** 2 * (mlim[:, 1] - mpos) ** 2 + 1e-8)

        p_neg = np.where(np.abs(mpos - mlim[:, 0]) > np.abs(mlim[:, 1] - mpos), 1, 1 / np.sqrt(1 + np.abs(h_grad)))
        p_pos = np.where(np.abs(mpos - mlim[:, 0]) > np.abs(mlim[:, 1] - mpos), 1 / np.sqrt(1 + np.abs(h_grad)), 1)

        J = J[np.newaxis, :, :]
        p_neg = p_neg[np.newaxis, np.newaxis, :].repeat(2 ** 7, axis=0).repeat(6, axis=1)
        p_pos = p_pos[np.newaxis, np.newaxis, :].repeat(2 ** 7, axis=0).repeat(6, axis=1)
        L = np.where(np.sign(J) * np.sign(orthants) < 0, p_neg, p_pos)
        J_aug = L * J

        ss = []
        for i in range(2 ** 7):
            u, s, vh = np.linalg.svd(J_aug[i])
            ss.append(s)
        quality_measure = np.min(ss) / np.max(ss)

        # time.sleep(1)
        data_dict = {'ee_pos': ee_pos, "quality_measure": quality_measure, "self_collision": self_collision}
        logger.write(json.dumps({"data": data_dict}) + '\n')
        logger.flush()
