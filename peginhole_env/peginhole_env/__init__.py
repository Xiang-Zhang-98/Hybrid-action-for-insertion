from gym.envs.registration import register 


register(id='peginhole_ori-v0',entry_point='peginhole_env.envs.Peginhole_ori_env:Peginhole_ori_env')
register(id='peginhole_3prms-v0',entry_point='peginhole_env.envs.peginhole_3prms:Peginhole_env')
register(id='peginhole_3prms-v2',entry_point='peginhole_env.envs.peginhole_3prms_v2:Peginhole_env')
register(id='peginhole_dis_mps-v0',entry_point='peginhole_env.envs.peginhole_discrete_mps:Peginhole_env')