from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.ant import Ant
from pybulletgym.envs.roboschool.scenes import MazeScene, MazeHardScene
import pybullet as p
import numpy as np


class AntMuJoCoEnv(WalkerBaseMuJoCoEnv):
    def __init__(self):
        self.robot = Ant()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)

class AntMuJoCoMazeEnv(WalkerBaseMuJoCoEnv):
    """
    This one has a simple maze and the image is centered at the center of the maze
    """
    def __init__(self):
        self.robot = Ant()
        self.robot.alive_bonus = self._alive_bonus # In this function the robot does not die if upside down
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)
        self.init_camera_params()

    def init_camera_params(self):
        """
        Sets camera parameter
        :return:
        """
        self.fixed = True
        self.base_pos = [6, 4, 0]
        self._cam_dist = 15
        self._cam_pitch = -90
        self._render_width = 64
        self._render_height = 64

    def _alive_bonus(self, z, pitch):
        return +1# if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

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

    def render(self, mode, close=False):#, top_bottom=False):
        """
        Reimplement env_bases render function so to have top_bottom view
        :param mode:
        :return:
        """
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])

        # base_pos = [0, 0, 0]
        if not self.fixed:
            if hasattr(self, 'robot'):
                if hasattr(self.robot, 'body_xyz'):
                    self.base_pos = self.robot.body_xyz
        #
        # if top_bottom:
        #     base_pos = [5, 5, 0]
        #     self._cam_dist = 20
        #     self._cam_pitch = -90
        #     self._render_width = 300
        #     self._render_height = 300
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        if rgb_array.dtype is np.uint8:
            rgb_array = 255 - rgb_array
        else:
            rgb_array = 1 - rgb_array
        return rgb_array


# Identical to AntMaze but uses MazeHardScene instead of MazeScene
class AntMuJoCoMazeHardEnv(AntMuJoCoMazeEnv):
    """
    This one has a more complex maze and the image is top-bottom centered on the robot (so you don't see the whole env)
    """

    def init_camera_params(self):
        """
        Sets camera parameter
        :return:
        """
        self.fixed = False
        self.base_pos = [7.5, 5, 0]
        self._cam_dist = 7
        self._cam_pitch = -90
        self._render_width = 64
        self._render_height = 64

    def create_single_player_scene(self, bullet_client):
        scene = MazeHardScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
        return scene