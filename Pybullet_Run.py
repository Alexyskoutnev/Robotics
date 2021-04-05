import pybullet as p
from time import sleep
import pybullet_data


physicsClient = p.connect(p.GUI)

#p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)
planeId = p.loadURDF('simpleplane.urdf', useFixedBase= 1)
cubeStartPos = [0, 0, 0]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("kortex_description/arms/gen3/7dof/urdf/GEN3_URDF_V12.urdf", cubeStartPos, cubeStartOrientation)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)

useRealTimeSimulation = 0

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)

while 1:
  if (useRealTimeSimulation):
    p.setGravity(0, 0, -10)
    sleep(0.01)  # Time in seconds.
  else:
    p.stepSimulation()