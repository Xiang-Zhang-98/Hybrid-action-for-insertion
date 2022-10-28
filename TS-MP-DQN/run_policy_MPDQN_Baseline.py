import os
import click
import time
import gym
# import gym_platform
from gym.wrappers import Monitor
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

def scale_state(state):
    high = np.ones(18)
    low = np.zeros(18)
    state = 2. * (state - low) / (high - low) - 1.
    return state

def unscale_state(state):
    high = np.ones(18)
    low = np.zeros(18)
    state = (high - low) * (state + 1.) / 2. + low
    return state

def evaluate(env, agent, episodes=100):
    returns = []
    timesteps = []
    paths = []
    verbose = False
    terminals =[]
    for _ in range(episodes):
        if verbose:
            print("new path")
        state, _ = env.reset()
        max_steps = 10
        total_reward = 0.
        running_path = dict(
            observations=[],
            primitive_id=[],
            rewards=[],
            params=[],
        )
        for i in range(max_steps):
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            if verbose:
                print(act)
            running_path["observations"].append(unscale_state(state))
            running_path["primitive_id"].append(act)
            running_path["params"].append(all_action_parameters)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            running_path["rewards"].append(reward)
            total_reward += reward
            if terminal:
                break
        print(terminal)
        running_path["observations"] = np.array(running_path["observations"])
        running_path["primitive_id"] = np.array(running_path["primitive_id"])
        running_path["params"] = np.array(running_path["params"])
        running_path["rewards"] = np.array(running_path["rewards"])
        paths.append(running_path)
        returns.append(total_reward)
        terminals.append(terminal)
    return paths, np.array(returns),terminals

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
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=-1, help='Number of episodes over which to linearly anneal epsilon.', type=int)
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
@click.option('--title', default="PDDQN", help="Prefix of output files", type=str)
@click.option('--policy_file', default="_", help="pkl file that saves policy weights", type=str)

def run_policy(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, policy_file):
    env = gym.make('peginhole_3prms-v2', render=True)
    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    env.seed(seed)
    np.random.seed(seed)
    from agents.TSMPDQN_3prms_v2 import PDQNAgent_Threshold
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent
    from agents.mpdqn_3prms_baseline import MultiPassPDQNAgent_Baseline
    assert not (split and multipass)
    agent_class = PDQNAgent_Threshold
    if split:
        agent_class = SplitPDQNAgent
    elif multipass:
        # agent_class = MultiPassPDQNAgent
        agent_class = MultiPassPDQNAgent_Baseline
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
        use_ornstein_noise=use_ornstein_noise,
        replay_memory_size=replay_memory_size,
        epsilon_final=epsilon_final,
        inverting_gradients=inverting_gradients,
        actor_kwargs={'hidden_layers': layers,
                      'action_input_layer': action_input_layer, },
        actor_param_kwargs={'hidden_layers': layers,
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001, },
        zero_index_gradients=zero_index_gradients,
        seed=seed)
    max_steps = 10
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    if policy_file == '_':
        raise ValueError("Unknown policy file")
    else:
        agent.load_models(os.getcwd() + '/results/peginhole' + policy_file)
    agent.end_episode()
    paths,_,terminals = evaluate(env, agent, 100)
    print(np.sum(terminals)/len(terminals))
if __name__ == '__main__':
    run_policy()
