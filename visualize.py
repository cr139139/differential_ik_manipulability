import json
import numpy as np
import tqdm

file_name = "log.txt"
log = open(file_name, "r")
lines = log.readlines()

ee_pos = []
quality_measure = []
self_collision = []

for i in tqdm.tqdm(range(len(lines))):
    data = json.loads(lines[i])['data']
    if data['self_collision'] == 0:
        ee_pos.append(data['ee_pos'])
        quality_measure.append(data['quality_measure'])
        self_collision.append(data['self_collision'])

ee_pos = np.array(ee_pos)
quality_measure = np.array(quality_measure)
quality_measure = np.log(quality_measure + 1e-8)
# quality_measure = (quality_measure-np.min(quality_measure))\
#                   /(np.max(quality_measure)-np.min(quality_measure))
self_collision = np.array(self_collision)

threshold = 0
quality_measure = quality_measure[ee_pos[:, 1] > -threshold]
self_collision = self_collision[ee_pos[:, 1] > -threshold]
ee_pos = ee_pos[ee_pos[:, 1] > -threshold, :]

# quality_measure = quality_measure[ee_pos[:, 1] < threshold]
# self_collision = self_collision[ee_pos[:, 1] < threshold]
# ee_pos = ee_pos[ee_pos[:, 1] < threshold, :]

# quality_measure = quality_measure[ee_pos[:, 0] < 0]
# self_collision = self_collision[ee_pos[:, 0] < 0]
# ee_pos = ee_pos[ee_pos[:, 0] < 0, :]



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure(figsize=(8, 8))
# plt.scatter(ee_pos[:, 0], ee_pos[:, 2], c=quality_measure, cmap='RdYlBu_r', alpha=1)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-1, 1])
ax.set_ylim3d([-1, 1])
ax.set_zlim3d([-1, 1])
ax.scatter(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2],
           c=quality_measure, cmap='RdYlBu_r', alpha=1)
plt.show()