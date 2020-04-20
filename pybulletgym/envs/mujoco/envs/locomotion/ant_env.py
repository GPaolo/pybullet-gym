from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.ant import Ant
from pybulletgym.envs.roboschool.scenes import MazeScene
import pybullet as p



class AntMuJoCoEnv(WalkerBaseMuJoCoEnv):
    def __init__(self):
        self.robot = Ant()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)

class AntMuJoCoMazeEnv(WalkerBaseMuJoCoEnv):
    def __init__(self):
        self.robot = Ant()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        scene = MazeScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
        return scene

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)

        r = BaseBulletEnv._reset(self)
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                             self.scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                               self.foot_ground_object_names])
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        if self.stateId < 0:
            self.stateId=self._p.saveState()

        for i in self.robot_body.bodies:
            for link in range(-1, 20): #Changes the color of the robot Ant
                p.changeVisualShape(i, link, rgbaColor=[0.2, 0.3, 0.4, 1])

        return r
