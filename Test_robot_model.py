import pybullet as p
from time import sleep
from time import sleep
p.connect(p.GUI)
p.loadURDF("../three_link_manipulator.urdf")
p.computeViewMatrix([2,2,2], [0,0,0], [0,0,1])
sleep(100)