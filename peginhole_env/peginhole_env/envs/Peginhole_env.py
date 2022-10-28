import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time
import os
import transforms3d.euler as trans_euler
import mujoco_py

class Peginhole_env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.work_space_xy_limit = 4
        self.work_space_z_limit = 4
        self.goal = np.array([0, 0, 1])
        self.render = True
        self.noise_level = 0.2
        self.use_noisy_state = True
        self.evaluation = self.render
        utils.EzPickle.__init__(self)
        # mujoco_env.MujocoEnv.__init__(self,
        #                               os.getcwd()+'/fall2020_peginhole_square.xml', 1)
        mujoco_env.MujocoEnv.__init__(self,
                                      '/home/zx/UCBerkeley/insertion/peginhole_env/peginhole_env/envs/fall2020_peginhole_square.xml', 1)
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
        if ob[2] > self.work_space_z_limit:
            action[2] = -5
        # check done
        if np.linalg.norm(ob[0:3] - self.goal) < 0.3:
            done = True
            action = np.zeros(3) # if reach to goal, then stay
        else:
            done = False

        velcmd = action/100
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        # evalute reward
        dist = np.linalg.norm(ob[0:3] - self.goal)
        if dist < 0.3:
            done = False
            reward = 1000
        else:
            done = False
            reward = np.power(10,3-dist)
            # reward = 0

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
        angle = 0 # don't consider orientation error for now
        quat = trans_euler.euler2quat(0, 0, angle)
        self.model.body_quat[1, :] = quat
        # peg is above the hole for 15 mm, located in a 5mm*5mm*5mm cube
        l = 0.8
        cube = np.random.uniform(low=-l, high=l, size=3)
        mb = cube + self.goal + np.array([0,0,2.5])
        self.model.body_pos[1, :] = mb/100

    def reset_hole(self):
        # angle = np.random.uniform(low=-np.pi / 180 * 30, high=np.pi / 180 * 30)
        angle = 0
        quat = trans_euler.euler2quat(0, 0, angle)
        self.model.body_quat[2, :] = quat

        # mb = self.model.body_pos[2, :]
        # l = 0.1
        # mb[0] = 0 + np.random.uniform(low=-l, high=l)  # np.random.uniform(low=-0.005, high=0.005)
        # mb[1] = 0 + np.random.uniform(low=-l, high=l)
        # mb[2] = 0.01  # +np.random.uniform(low=-5, high=5)
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

        # xquat = self.sim.data.get_body_xquat("peg") #orientation
        # xeul = trans_euler.quat2euler(xquat)
        xvelp = self.sim.data.get_body_xvelp("peg")  # velocity

        # xvelr = self.sim.data.get_body_xvelr("peg")# rotation velocity
        # vertex_pos = self.sim.data.site_xpos[0,:] # tip position
        force = self.sim.data.cfrc_ext[1, :]
        # print('number of contacts', self.sim.data.ncon)
        # print("force")
        # print(force)
        # c_array = np.zeros(6, dtype=np.float64)
        # # print('c_array', c_array)
        # mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data,2, c_array)
        # print('c_array', c_array)
        # for i in range(self.sim.data.ncon):
        #     # Note that the contact array has more than `ncon` entries,
        #     # so be careful to only read the valid entries.
        #     contact = self.sim.data.contact[i]
        #     print('contact', i)
        #     print('dist', contact.dist)
        #     print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
        #     print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
        #     # There's more stuff in the data structure
        #     # See the mujoco documentation for more info!
        #     geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2]
        #     print(' Contact force on geom2 body', self.sim.data.cfrc_ext[geom2_body])
        #     print('norm', np.sqrt(np.sum(np.square(self.sim.data.cfrc_ext[geom2_body]))))
        #     # Use internal functions to read out mj_contactForce
        #     c_array = np.zeros(6, dtype=np.float64)
        #     print('c_array', c_array)
        #     mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
        #     print('c_array', c_array)

        return np.concatenate([
            xpos * 100,
            # 	xquat,
            xvelp* 100,
            # 	xvelr,
            # 	vertex_pos,
            force[5:]*0
            ])


if __name__ == "__main__":
    env = Peginhole_env()
    env.reset()

    for i in range(200000):
        #env.reset()
        i = i % 400
        print(i)
        if i % 400 == 0:
            env.reset()
        #ob, reward, done,_ = env.step(np.array([0, 0.0, -0.1]))
        # print(ob)
        # print(done)
        # print(reward)
        if i < 200:
            ob, reward, done,_ = env.step(np.array([0, 0.0, -100]))
            # print(ob)
            # print(done)
            # print(reward)
        else:
            #ob, reward, done, _ = env.step(np.array([0, 0.0, -0.1]))
            ob, reward, done,_ = env.step(np.array([-0.1, 0.1, 0.0]))
            # print(ob)
            # print(done)
            # print(reward)
