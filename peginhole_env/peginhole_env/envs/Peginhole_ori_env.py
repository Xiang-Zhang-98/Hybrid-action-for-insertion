import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time
import os
import transforms3d.euler as trans_euler
import mujoco_py

class Peginhole_ori_env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, pegshape="triangle"):
        self.work_space_xy_limit = 4
        self.work_space_z_limit = 4
        self.work_space_rollpitch_limit = np.pi*5/180.0
        self.work_space_yaw_limit = np.pi*10/180.0
        self.goal = np.array([0, 0, 1])
        self.goal_ori = np.array([0, 0, 0])
        self.render = False
        self.noise_level = 0.2
        self.ori_noise_level = 0.5
        self.use_noisy_state = True
        self.force_noise = True
        self.force_noise_level = 0.2
        self.evaluation = self.render
        utils.EzPickle.__init__(self)
        # mujoco_env.MujocoEnv.__init__(self,
        #                               os.getcwd()+'/fall2020_peginhole_square.xml', 1)
        if pegshape == "square":
            mujoco_env.MujocoEnv.__init__(self,
                                          os.getcwd() + '/peginhole_env/peginhole_env/envs/fall2020_peginhole_square_ori.xml',
                                          1)
        elif pegshape == "square_tight":
            mujoco_env.MujocoEnv.__init__(self, os.getcwd() +
                                          '/peginhole_env/peginhole_env/envs/fall2020_peginhole_square_ori_very_tight.xml',
                                          1)
        elif pegshape == "pentagon":
            mujoco_env.MujocoEnv.__init__(self, os.getcwd() +
                                          '/peginhole_env/peginhole_env/envs/spring_peginhole_5bian_32.xml',
                                          1)
        elif pegshape == "triangle":
            mujoco_env.MujocoEnv.__init__(self, os.getcwd() +
                                          '/peginhole_env/peginhole_env/envs/triangular.xml',
                                          1)
        else:
            raise ValueError("Unknown pegshape")
        # mujoco_env.MujocoEnv.__init__(self,
        #                               '/home/fanuc/Xiang/insertion/peginhole_env/peginhole_env/envs/fall2020_peginhole_square_ori.xml',
        #                                1)
        # obs units are cm now !!
        # action units are cm/s now !!
        if self.render:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.viewer = None

    def reset(self):
        if self.viewer is not None:
            self.viewer.render()
        return  self.reset_model()

    def step(self, action):
        # checking workspace
        ob = self._get_obs()
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
            action = np.zeros(6) # if reach to goal, then stay
        else:
            done = False

        velcmd = action
        #velcmd[3:6] = np.array([0,0,0])
        # velcmd[0:3] = velcmd[0:3]/100
        for i in range(3):
            self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        # evalute reward
        dist = np.linalg.norm(ob[0:3] - self.goal)
        ori_error = np.linalg.norm((ob[3:6] - self.goal_ori)/np.pi*180)
        ori_reward = np.power(10,1-ori_error/10)
        # print("ori_error")
        # print(ori_error)
        # print("ori_reward")
        # print(ori_reward)
        if dist < 0.3:
            done = False
            reward = 1000
        else:
            done = False
            reward = np.power(10,3-dist)
            # reward = 0
        reward = reward+0*ori_reward
        if self.viewer is not None:
            self.viewer.render()
        # if self.use_noisy_state:
        #     ob[0:6] = ob[0:6] + np.random.normal(0,self.noise_level,6)
        if self.evaluation and dist < 0.5:
            done = True
        return ob, reward, done, dict(reward_dist=reward)  # _dist)#, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        if self.viewer is not None:
            self.viewer.cam.trackbodyid = 0

    def reset_peg(self):
        # angle = np.random.uniform(low=-np.pi / 180 * 30, high=np.pi / 180 * 30)
        # angle = 0 # don't consider orientation error for now
        # angle = np.random.uniform(low=-np.pi / 180 * 3, high=np.pi / 180 * 3)
        # quat = trans_euler.euler2quat(0, 0, angle)
        angle = np.pi / 180 * 5
        yaw = np.random.uniform(low=-np.pi / 180 * 10, high=np.pi / 180 * 10)
        # previous reset before 6/8
        # yaw = np.pi / 180 * 7
        quat = trans_euler.euler2quat(0, angle, yaw)
        self.model.body_quat[1, :] = quat
        # peg is above the hole for 15 mm, located in a 5mm*5mm*5mm cube
        # previous reset before 6/8
        # l = 0.8
        # cube = np.random.uniform(low=-l, high=l, size=3)
        # mb = cube + self.goal + np.array([0,2.5,2.5])
        l = np.array([3, 3, 0.5])
        cube = np.random.uniform(low=-l, high=l)
        mb = cube + self.goal + np.array([0, 0, 3])

        self.model.body_pos[1, :] = mb / 100

    def reset_hole(self):
        # angle = np.random.uniform(low=-np.pi / 180 * 30, high=np.pi / 180 * 30)
        angle = np.random.normal(0, self.ori_noise_level / 180 * np.pi)
        quat = trans_euler.euler2quat(0, 0, angle)
        self.model.body_quat[2, :] = quat

        # mb = self.model.body_pos[2, :]
        # l = 0.1
        # mb[0] = 0 + np.random.uniform(low=-l, high=l)  # np.random.uniform(low=-0.005, high=0.005)
        # mb[1] = 0 + np.random.uniform(low=-l, high=l)
        # mb[2] = 0.01  # +np.random.uniform(low=-5, high=5)
        mb = np.zeros(3)
        if self.use_noisy_state:
            mb[0:2] = mb[0:2] + np.random.normal(0, self.noise_level / 100, 2)
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
        # vertex_pos = self.sim.data.site_xpos[0,:] # tip position
        force = self.sim.data.cfrc_ext[1, :]
        if self.force_noise:
            force = force + np.random.normal(0, self.force_noise_level, 6)
        return np.concatenate([
            xpos * 100,
            xeul,
            xvelp * 100,
            xvelr,
            # 	vertex_pos,
            force[3:],

            force[0:3]
        ])


if __name__ == "__main__":
    env = Peginhole_ori_env()
    env.reset()

    for i in range(200000):
        #env.reset()
        if i % 200 == 0:
            env.reset()
        ob, reward, done,_ = env.step(np.array([0.0, 1, 0,0.0,0.0,0.1]))
        print(ob)
        print(done)
        print(reward)
        # if i < 200:
        #     ob, reward, done,_ = env.step(np.array([0, 0.0, -0.1]))
        #     print(ob)
        #     print(done)
        #     print(reward)
        # else:
        #     ob, reward, done,_ = env.step(np.array([-0.1, 0.1, 0.1]))
        #     print(ob)
        #     print(done)
        #     print(reward)
