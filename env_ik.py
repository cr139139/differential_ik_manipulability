import pybullet as pb
import pybullet_data
import time
from tqdm import tqdm
import numpy as np
import cvxpy as cp

pbId = pb.connect(pb.GUI)
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
pb.resetSimulation()
robotId = pb.loadURDF('/robots/iiwa7_mount_sake.urdf', [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                      flags=pb.URDF_USE_SELF_COLLISION)
eeId = 7
start = True
manipulability_1 = []
manipulability_2 = []
for i in range(2000):
    events = pb.getKeyboardEvents()
    key_codes = events.keys()
    if 115 in key_codes:
        start = True

    pb.stepSimulation()

    if start:
        mpos, mvel, mtorq, mlim = getMotorJointStates(robotId)
        result = pb.getLinkState(robotId,
                                 eeId,
                                 computeLinkVelocity=1,
                                 computeForwardKinematics=1)
        zero_vec = [0.0] * len(mpos)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        jac_t, jac_r = pb.calculateJacobian(robotId, eeId, link_vt, mpos, zero_vec, zero_vec)
        J = np.concatenate((np.array(jac_t), np.array(jac_r)), axis=0)
        J = J[:, :7]

        H = np.zeros((7, 6, 7))
        for j in range(7):
            for k in range(7):
                a = np.min([j, k])
                b = np.max([j, k])
                H[j, :3, k] = np.cross(J[3:, a], J[:3, b])
                H[j, 3:, k] = np.cross(J[3:, k], J[3:, j])

        JJ = J @ J.T
        JJ_inv = np.linalg.inv(JJ)
        m = np.sqrt(np.abs(np.linalg.det(JJ)))
        manipulability_1.append(m)
        J_mani = np.zeros(7)
        for i in range(7):
            J_mani[i] = np.sum((J @ H[i, :, :].T) * JJ_inv)
        J_mani = m * J_mani

        link_rot = pb.getEulerFromQuaternion(link_rot)
        fk_q = np.concatenate((np.array(link_trn), np.array(link_rot)))

        Q = np.eye(6)
        R = np.eye(7)
        mpos = np.array(mpos)[:7]
        mlim = np.array(mlim)[:7]

        # x = np.array([[0.5, 0, 0.5, np.pi/2, 0, 0],
        #               [0.5, 0, 0.5, -np.pi/2, 0, 0],
        #               [0.5, 0, 0.5, 0, np.pi/2, 0],
        #               [0.5, 0, 0.5, 0, -np.pi/2, 0],
        #               [0.5, 0, 0.5, 0, 0, np.pi/2],
        #               [0.5, 0, 0.5, 0, 0, -np.pi/2]])
        #
        # b = cp.Variable(6)
        # q_delta = cp.Variable(7)
        # delta = cp.Variable(6)
        # cost = cp.quad_form(q_delta, R) - J_mani @ q_delta + cp.quad_form(delta, Q)# + cp.sum_squares(fk_q + J @ q_delta - b @ x)
        # constraints = [fk_q + J @ q_delta == b @ x + delta,
        #                mlim[:, 0] <= q_delta + mpos,
        #                q_delta + mpos <= mlim[:, 1],
        #                q_delta <= np.min(np.abs(mlim - mpos[:, np.newaxis]), axis=1),
        #                cp.sum(b) == 1,
        #                0 <= b, b <= 1]
        # q_delta.value = np.zeros(7)
        # b.value = np.zeros(6)
        # prob = cp.Problem(cp.Minimize(cost), constraints)
        # prob.solve(solver="ECOS_BB")



        x = [0.5, 0, 0.5, 0, 0, 0] # x[np.argmax(b.value)]
        diff = (fk_q - x)[3:]
        diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
        diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)
        x[3:] = fk_q[3:] - diff

        delta = cp.Variable(6)
        q_delta = cp.Variable(7)

        cost = cp.quad_form(delta, Q) + cp.quad_form(q_delta, R)# - J_mani @ q_delta
        constraints = [fk_q + J @ q_delta == x + delta,
                       mlim[:, 0] <= q_delta + mpos,
                       q_delta + mpos <= mlim[:, 1],
                       q_delta <= np.min(np.abs(mlim - mpos[:, np.newaxis]), axis=1)]
        q_delta.value = np.zeros(7)
        delta.value = np.zeros(6)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        delta_q = q_delta.value
        # print(np.argmax(b.value))

        if delta_q is not None:
            # for i in range(7):
            #     pb.resetJointState(robotId, i, delta_q[i] * 0.1 + mpos[i])
            pb.setJointMotorControlArray(robotId, jointIndices=[i for i in range(7)], controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocities=delta_q * 10,
                                         forces=np.ones(7) * 100)

    time.sleep(1 / 240)

pb.resetSimulation()
robotId = pb.loadURDF('/robots/iiwa7_mount_sake.urdf', [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                      flags=pb.URDF_USE_SELF_COLLISION)

for i in range(2000):
    events = pb.getKeyboardEvents()
    key_codes = events.keys()
    if 115 in key_codes:
        start = True

    pb.stepSimulation()

    if start:
        mpos, mvel, mtorq, mlim = getMotorJointStates(robotId)
        result = pb.getLinkState(robotId,
                                 eeId,
                                 computeLinkVelocity=1,
                                 computeForwardKinematics=1)
        zero_vec = [0.0] * len(mpos)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        jac_t, jac_r = pb.calculateJacobian(robotId, eeId, link_vt, mpos, zero_vec, zero_vec)
        J = np.concatenate((np.array(jac_t), np.array(jac_r)), axis=0)
        J = J[:, :7]

        H = np.zeros((7, 6, 7))
        for j in range(7):
            for k in range(7):
                a = np.min([j, k])
                b = np.max([j, k])
                H[j, :3, k] = np.cross(J[3:, a], J[:3, b])
                H[j, 3:, k] = np.cross(J[3:, k], J[3:, j])

        JJ = J @ J.T
        JJ_inv = np.linalg.inv(JJ)
        m = np.sqrt(np.abs(np.linalg.det(JJ)))
        manipulability_2.append(m)
        J_mani = np.zeros(7)
        for i in range(7):
            J_mani[i] = np.sum((J @ H[i, :, :].T) * JJ_inv)
        J_mani = m * J_mani

        link_rot = pb.getEulerFromQuaternion(link_rot)
        fk_q = np.concatenate((np.array(link_trn), np.array(link_rot)))

        Q = np.eye(6)
        R = np.eye(7)
        mpos = np.array(mpos)[:7]
        mlim = np.array(mlim)[:7]

        x = [0.5, 0, 0.5, 0, 0, 0] # x[np.argmax(b.value)]
        diff = (fk_q - x)[3:]
        diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
        diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)
        x[3:] = fk_q[3:] - diff

        delta = cp.Variable(6)
        q_delta = cp.Variable(7)

        cost = cp.quad_form(delta, Q) + cp.quad_form(q_delta, R) - J_mani @ q_delta
        constraints = [fk_q + J @ q_delta == x + delta,
                       mlim[:, 0] <= q_delta + mpos,
                       q_delta + mpos <= mlim[:, 1],
                       q_delta <= np.min(np.abs(mlim - mpos[:, np.newaxis]), axis=1)]
        q_delta.value = np.zeros(7)
        delta.value = np.zeros(6)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        delta_q = q_delta.value
        # print(np.argmax(b.value))

        if delta_q is not None:
            pb.setJointMotorControlArray(robotId, jointIndices=[i for i in range(7)], controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocities=delta_q * 10,
                                         forces=np.ones(7) * 100)

    time.sleep(1 / 240)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.plot(range(2000), manipulability_1, alpha=0.5, color='b', label='without manipulability term')
ax.plot(range(2000), manipulability_2, alpha=0.5, color='r', label='with manipulability term')
ax.set_xlim(0, 1999)
ax.set_ylim(0, 0.15)
ax.set_xlabel("timesteps")
ax.set_ylabel("manipulability index")
ax.set_title("Manipulability over time")
ax.legend(loc="upper left")
ax.grid()

plt.show()