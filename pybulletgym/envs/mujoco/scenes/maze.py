# Created by Giuseppe Paolo 
# Date: 20/04/2020

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

from .scene_bases import Scene
import pybullet


class MazeScene(Scene):
  multiplayer = False
  mazeLoaded = 0
  maze_name = "maze0.sdf"

  def episode_restart(self, bullet_client):
    self._p = bullet_client
    Scene.episode_restart(self, bullet_client)
    if self.mazeLoaded == 0:
      self.mazeLoaded = 1

      # stadium_pose = cpp_household.Pose()
      # if self.zero_at_running_strip_start_line:
      #	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants

      filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "maze", "maze_plane.sdf")
      self.ground_plane_mjcf=self._p.loadSDF(filename)
      filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "maze", self.maze_name)
      self.walls=self._p.loadSDF(filename)

      for i in self.ground_plane_mjcf:
        self._p.changeDynamics(i,-1,lateralFriction=0.8, restitution=0.5)
        self._p.changeVisualShape(i,-1,rgbaColor=[0,0,0,1.0])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,0)

    #	for j in range(pybullet.getNumJoints(i)):
    #		self._p.changeDynamics(i,j,lateralFriction=0)
    #despite the name (stadium_no_collision), it DID have collision, so don't add duplicate ground

# Identical to the other. Just changes the loaded maze
class MazeHardScene(MazeScene):
  maze_name = 'maze1.sdf'