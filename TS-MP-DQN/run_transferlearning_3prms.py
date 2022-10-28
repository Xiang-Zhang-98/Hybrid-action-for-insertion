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
@click.option('--episodes', default=5000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
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
@click.option('--title', default="Transferlearning_pentagon", help="Prefix of output files", type=str)
@click.option('--with_warm_start', default=True, type=bool)
@click.option('--loadingpath',default="_", type=str)
@click.option('--td3_noise', default=0.0, type=float)
@click.option('--td3_noise_decay', default=False, type=bool)
@click.option('--td3_noise_decay_episode', default=10000, type=int)
@click.option('--fintune_start_episode', default=1000, type=int)
@click.option('--pegshape', default="square", type=str)
@click.option('--baseline', default=False, type=bool)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, with_warm_start,loadingpath, td3_noise
        ,td3_noise_decay ,td3_noise_decay_episode, fintune_start_episode, pegshape, baseline):
    if loadingpath == '_':
        raise ValueError("Unknown policy file")
    else:
        loadingpath= os.getcwd() + '/results/peginhole' + loadingpath
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

    env = gym.make('peginhole_3prms-v2', pegshape=pegshape)
    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)

    dir = os.path.join(save_dir,title)
    env.seed(seed)
    np.random.seed(seed)

    print(env.observation_space)

    from agents.pdqn import PDQNAgent
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent
    from agents.TSMPDQN_3prms_v2 import MultiPassPDQNAgent_Threshold
    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        agent_class = SplitPDQNAgent
    elif multipass:
        # agent_class = MultiPassPDQNAgent
        agent_class = MultiPassPDQNAgent_Threshold

    if with_warm_start:
        epsilon_steps = -1
    if not baseline:
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
                       seed=seed,
                       policy_noise=td3_noise,
                       noise_decay=td3_noise_decay,
                       noise_decay_episode=td3_noise_decay_episode,
                       learn_actor=True)
    else:
        from agents.mpdqn_3prms_baseline import MultiPassPDQNAgent_Baseline
        agent = MultiPassPDQNAgent_Baseline(
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
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=zero_index_gradients,
                       seed=seed,
                       learn_actor=True)
    if with_warm_start:
        agent.load_Q(loadingpath)
        agent.end_episode()
        max_steps = 10
    print(agent)
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0

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
        agent.start_episode()
        for j in range(max_steps):
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            # agent.step(state, (act, all_action_parameters), reward, next_state,
            #            (next_act, next_all_action_parameters), terminal, steps)
            if max_steps*i < fintune_start_episode*10:
                for _ in range(1):
                    agent.transfer_step(state, (act, all_action_parameters), reward, next_state,
                                    (next_act, next_all_action_parameters), terminal, steps)
            else:
                agent.step(state, (act, all_action_parameters), reward, next_state,
                           (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            if visualise and i % render_freq == 0:
                env.render()

            if terminal:
                break
        agent.end_episode()

        if save_frames and i % render_freq == 0:
            video_index = env.unwrapped.save_render_states(vidir, title, video_index)

        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
            dir_return = save_dir + '/returns' + str(i) + '.npy'
            with open(dir_return, 'wb') as f:
                np.save(f, np.array(returns))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

if __name__ == '__main__':
    run()
