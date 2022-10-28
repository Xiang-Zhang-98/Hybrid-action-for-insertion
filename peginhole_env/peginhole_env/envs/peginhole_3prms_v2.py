import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time
import os
import transforms3d.euler as trans_euler
import mujoco_py
from gym import spaces

class Peginhole_env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, pegshape="square", render=False):
        self.work_space_xy_limit = 4
        self.work_space_z_limit = 4
        self.work_space_rollpitch_limit = np.pi*5/180.0
        self.work_space_yaw_limit = np.pi*10/180.0
        self.goal = np.array([0, 0, 1])
        self.goal_ori = np.array([0, 0, 0])
        self.render = render
        self.noise_level = 0.2
        self.ori_noise_level = 0.5
        self.use_noisy_state = True
        self.force_noise = True
        self.force_noise_level = 0.2
        self.evaluation = self.render
        self.moving_pos_threshold = 2.5
        self.moving_ori_threshold = 4
        utils.EzPickle.__init__(self)
        if pegshape == "square":
            mujoco_env.MujocoEnv.__init__(self,
                                      os.getcwd() + '/peginhole_env/peginhole_env/envs/fall2020_peginhole_square_ori.xml',
                                      1)
        elif pegshape == "square_tight":
            mujoco_env.MujocoEnv.__init__(self,os.getcwd() +
                                    '/peginhole_env/peginhole_env/envs/fall2020_peginhole_square_ori_very_tight.xml',1)
        elif pegshape == "pentagon":
            mujoco_env.MujocoEnv.__init__(self, os.getcwd() +
                                          '/peginhole_env/peginhole_env/envs/spring_peginhole_pentagon.xml',
                                          1)
        elif pegshape == "pentagon_tight":
            mujoco_env.MujocoEnv.__init__(self, os.getcwd() +
                                          '/peginhole_env/peginhole_env/envs/pentagon_tight.xml',
                                          1)
        elif pegshape == "triangle":
            mujoco_env.MujocoEnv.__init__(self, os.getcwd() +
                                          '/peginhole_env/peginhole_env/envs/triangular.xml',
                                          1)
        elif pegshape == "triangle_tight":
            mujoco_env.MujocoEnv.__init__(self, os.getcwd() +
                                          '/peginhole_env/peginhole_env/envs/triangle_tight.xml',
                                          1)
        else:
            raise ValueError("Unknown pegshape")
        # obs units are cm now !!
        # action units are cm/s now !!
        self.parameter_high = [np.array([0.5,0.5,0.5,2.5]),np.array([0.5,0.5,0.5,2.5]),np.array([5])]
        self.parameter_low = [np.array([-0.5, -0.5, -0.5, 0]),np.array([-0.5, -0.5, -0.5, 0]),np.array([0])]
        num_actions = len(self.parameter_low)
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            spaces.Tuple(  # parameters
                tuple(spaces.Box(low=self.parameter_low[i],
                                 high=self.parameter_high[i], dtype=np.float32)
                      for i in range(num_actions))
            )
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0., high=1., shape=self._get_obs().shape, dtype=np.float32),
            spaces.Discrete(200),  # steps (200 limit is an estimate)
        ))
        if self.render:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = None

    def reset(self):
        if self.viewer is not None:
            self.viewer.render()
        return  self.reset_model(), 0

    def step(self,action):
        if type(action) is np.ndarray:
            return self._get_obs(), 0, 0, dict(reward_dist=0)
        elif action[0] == 0:
            # Translation
            vel_cmd = action[1][0:3]
            threshold = action[1][-1]
            fullaction = np.concatenate((vel_cmd, np.zeros(3)))
            if fullaction[2] > 0 :
                fullaction[2] = 0
        elif action[0] == 1:
            # Rotation
            vel_cmd = action[1][0:3]
            threshold = action[1][-1]
            fullaction = np.concatenate((np.zeros(3), 10*vel_cmd))
        elif action[0] == 2:
            # Insert
            threshold = action[1][-1]
            fullaction = np.array([0, 0, -1, 0, 0, 0])
        action = fullaction
        force_limit = threshold
        # torque_limit = 100
        init_ob = self._get_obs()
        for i in range(100):
            ob = self._get_obs()
            curr_force = ob[12:]
            if np.abs(np.dot(curr_force,action)/np.linalg.norm(action + 1e-6,ord=2)) >force_limit:
                break
            delta_ob = ob - init_ob
            if np.linalg.norm(delta_ob[0:3], ord=2)> self.moving_pos_threshold or np.linalg.norm(delta_ob[3:6],ord=2)> self.moving_ori_threshold/180*np.pi:
                break
            if np.abs(ob[0]) > self.work_space_xy_limit:
                action[0] = -5 * np.sign(ob[0])
            if np.abs(ob[1]) > self.work_space_xy_limit:
                action[1] = -5 * np.sign(ob[1])
            if np.abs(ob[3]) > self.work_space_rollpitch_limit:
                action[3] = -1 * np.sign(ob[3])
            if np.abs(ob[4]) > self.work_space_rollpitch_limit:
                action[4] = -1 * np.sign(ob[4])
            if np.abs(ob[5]) > self.work_space_yaw_limit:
                action[5] = -1 * np.sign(ob[5])
            if ob[2] > self.work_space_z_limit:
                action[2] = -5
            # check done
            if np.linalg.norm(ob[0:3] - self.goal) < 0.3:
                done = False
                action = np.zeros(6)  # if reach to goal, then stay
            else:
                done = False
            self.do_simulation(action/10 , self.frame_skip)
        ob = self._get_obs()
        # evalute reward
        dist = np.linalg.norm(ob[0:3] - self.goal)
        if dist < 0.3:
            done = False
            reward = 1000
        else:
            done = False
            reward = np.power(10, 3 - dist)
        reward = reward
        if self.evaluation and dist < 0.5:
            done = True
        if self.viewer is not None:
            self.viewer.render()
        return (ob, i), reward, done, dict(reward_dist=reward)

    def viewer_setup(self):
        if self.viewer is not None:
            self.viewer.cam.trackbodyid = 0

    def reset_peg(self):
        angle = np.pi/180*5
        yaw = np.random.uniform(low=-np.pi / 180 * 10, high=np.pi / 180 * 10)
        quat = trans_euler.euler2quat(0, angle, yaw)
        self.model.body_quat[1, :] = quat
        # peg is above the hole for 15 mm, located in a 5mm*5mm*5mm cube
        l = np.array([3,3,0.5])
        cube = np.random.uniform(low=-l, high=l)
        mb = cube + self.goal + np.array([0,0,3])

        self.model.body_pos[1, :] = mb/100

    def reset_hole(self):
        angle = np.random.normal(0, self.ori_noise_level / 180 * np.pi)
        quat = trans_euler.euler2quat(0, 0, angle)
        self.model.body_quat[2, :] = quat

        mb = np.zeros(3)
        if self.use_noisy_state:
            mb[0:2] = mb[0:2] + np.random.normal(0,self.noise_level/100,2)
        mb[2] = 0.01
        self.model.body_pos[2, :] = mb

    def reset_model(self):
        self.reset_peg()
        self.reset_hole()

        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        ob_before = self._get_obs()
        return ob_before

    def _get_obs(self):
        xpos = self.sim.data.get_body_xpos("peg")  # cog position

        xquat = self.sim.data.get_body_xquat("peg") #orientation
        xeul = trans_euler.quat2euler(xquat)
        xvelp = self.sim.data.get_body_xvelp("peg")  # velocity

        xvelr = self.sim.data.get_body_xvelr("peg")# rotation velocity
        force = self.sim.data.cfrc_ext[1, :]
        if self.force_noise:
            force = force + np.random.normal(0, self.force_noise_level, 6)
        return np.concatenate([
            xpos * 100,
            xeul,
            xvelp* 100,
            xvelr,
            force[3:],

            force[0:3]
            ])


if __name__ == "__main__":
    env = Peginhole_env()
    env.reset()
