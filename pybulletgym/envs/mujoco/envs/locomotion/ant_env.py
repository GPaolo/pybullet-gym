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

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()[:29]  # This ones removes all the zeros that are attached at the end for no apparent reason
        state[27:] = self.robot.body_xyz[:2] # Use the last two zeros of the state to put the XY position of the robot.

        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            alive,
            progress,
            joints_at_limit_cost,
            feet_collision_cost
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def create_single_player_scene(self, bullet_client):
        scene = MazeScene(bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
        return scene

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)

        r = BaseBulletEnv._reset(self)[:29]
        r[27:] = self.robot.body_xyz[:2]
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
                p.changeVisualShape(i, link, rgbaColor=[1, 0, 0, 1])

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
        if rgb_array.dtype == np.uint8:
            rgb_array = rgb_array/255
        # else:
        #     rgb_array = 1 - rgb_array
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