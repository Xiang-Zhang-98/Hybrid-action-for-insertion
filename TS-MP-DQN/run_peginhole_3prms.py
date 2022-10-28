import os
import click
import time
import gym
# import gym_platform
# from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper

import numpy as np
import peginhole_env

from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper


def pad_action(act, act_param):
    # params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    # params[act][:] = act_param
    # return (act, params)
    return [act, act_param]


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    return np.array(returns)


@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
@click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 0.00001
@click.option('--scale-actions', default=False, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=False, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[128,128]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=1000, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/peginhole", help='Output directory.', type=str)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=False, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="proposed_square_peg", help="Prefix of output files", type=str)
@click.option('--with_warm_start', default=False, type=bool)
@click.option('--td3_noise', default=0.2, type=float)
@click.option('--td3_noise_decay', default=True, type=bool)
@click.option('--td3_noise_decay_episode', default=10000, type=int)
@click.option('--pegshape', default="square", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, with_warm_start, td3_noise,
        td3_noise_decay, td3_noise_decay_episode,pegshape):

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)
    assert not (save_frames and visualise)
    if visualise:
        assert render_freq > 0
    if save_frames:
        assert render_freq > 0
        vidir = os.path.join(save_dir, "frames")
        os.makedirs(vidir, exist_ok=True)

    env = gym.make('peginhole_3prms-v2', pegshape=pegshape, render=False)
    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)

    dir = os.path.join(save_dir,title)
    env.seed(seed)
    np.random.seed(seed)

    print(env.observation_space)

    from agents.TSMPDQN_3prms_v2 import PDQNAgent_Threshold
    from agents.pdqn_split import SplitPDQNAgent
    from agents.TSMPDQN_3prms_v2 import MultiPassPDQNAgent_Threshold
    assert not (split and multipass)
    agent_class = PDQNAgent_Threshold
    if split:
        agent_class = SplitPDQNAgent
    elif multipass:
        agent_class = MultiPassPDQNAgent_Threshold

    if with_warm_start:
        epsilon_steps = -1
    agent = agent_class(
                       env.observation_space.spaces[0], env.action_space,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       weighted=weighted,
                       average=average,
                       random_weighted=random_weighted,
                       initial_memory_threshold=initial_memory_threshold,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=zero_index_gradients,
                       policy_noise=td3_noise,
                       noise_decay=td3_noise_decay,
                       noise_decay_episode=td3_noise_decay_episode,
                       seed=seed)

    if with_warm_start:
        agent.load_models('/home/zx/UCBerkeley/insertion/MP-DQN/results/peginhole/mp_learning_3prms_V2_w_movinglimits_Short_H_ori4_TD3_freq1_[128,128]_square1/38000')
        agent.end_episode()
        max_steps = 10
    else:
        max_steps = 10
    print(agent)
    total_reward = 0.
    returns = []
    total_steps = [0]
    start_time = time.time()
    video_index = 0
    terminal_list = []
    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        if visualise and i % render_freq == 0:
            env.render()

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)

        episode_reward = 0.
        episode_steps = 0
        agent.start_episode()
        for j in range(max_steps):
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            episode_steps += steps
            if visualise and i % render_freq == 0:
                env.render()

            if terminal:
                break
        agent.end_episode()
        if reward > 300:
            terminal_list.append(True)
        else:
            terminal_list.append(False)
        if save_frames and i % render_freq == 0:
            video_index = env.unwrapped.save_render_states(vidir, title, video_index)

        returns.append(episode_reward)
        total_steps.append(total_steps[-1]+episode_steps)
        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
            # dir_return = save_dir + '/returns' + str(i) + '.npy'
            dir_return = save_dir + '/returns' + '.npy'
            with open(dir_return, 'wb') as f:
                np.save(f, np.array(returns))
            dir_step = save_dir + '/steps' + '.npy'
            with open(dir_step, 'wb') as f:
                np.save(f, np.array(total_steps))
            dir_step = save_dir + '/terminals' + '.npy'
            with open(dir_step, 'wb') as f:
                np.save(f, np.array(terminal_list))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()
