import gym
from rl_agents.agents.deep_q_network.pytorch import DQNAgent
from rl_agents.trainer.evaluation import Evaluation
from gif_saver import save_frames_as_gif
from tal_reward_functions import *
from user_defined import *

ENV = "highway"
AGENT_DETAILS = {"FastLeft": "trained", "Fast": "trained", "Truck": "trained"}

NUM_SIMULATIONS = 10
LIST_N = [2, 5, 10, 20]

config = dict(model=dict(type="MultiLayerPerceptron"),
              optimizer=dict(type="ADAM",
                             lr=1e-4,
                             weight_decay=0,
                             k=5),
              loss_function="l2",
              memory_capacity=5000000,
              batch_size=32,
              gamma=0.99,
              device="cuda:best",
              exploration=dict(method="EpsilonGreedy"),
              target_update=10000,
              maximum_episode_length=1000,
              replay_buffer=dict(type="PrioritizedReplayBuffer",
                                 alpha=0.6,
                                 beta=0.4,
                                 beta_schedule=dict(type="LinearSchedule",
                                                    initial_p=0.4,
                                                    final_p=1.0,
                                                    schedule_timesteps=1000000)),

              double=True)


def load_agents(agents_details):
    """
    :param agents_details: dict of env: run
    :return: dict of env_name : loaded agent
    """
    agents = {}
    for env, run in agents_details.items():
        agent = DQNAgent(gym.make(ENV + "-v0"), config)
        agent.load(f"out/{env}/DQNAgent/{run}/checkpoint-best.tar")
        evaluation = Evaluation(gym.make(ENV + "-v0"), agent, display_env=False, num_episodes=1)
        evaluation.test()
        agents[f"{env}"] = agent
    return agents


def f_1(trajectory, policy, agent_actions, n):
    """
    :param trajectory: list of states
    :param policy: function that takes a state and returns an action (representing user's policy)
    :param agent_actions: list of actions of the trained agent
    :param n: length of trajectory
    :return: list of actions
    """
    if len(trajectory) != n:
        return False
    user_actions = []
    for i, state in enumerate(trajectory):
        if policy(state) != agent_actions[i] and policy(state) != 5:
            return False
        user_actions.append(policy(state))
    return user_actions


def f_2(trajectory, policy, agent_actions, n, l2):
    """
    :param trajectory: list of states
    :param policy: function that takes a state and returns an action (representing user's policy)
    :param agent_actions: list of actions of the trained agent
    :param n: length of trajectory
    :param l2: number of actions that has to be different
    :return: list of actions
    """
    return f_1(trajectory, policy, agent_actions, n) and len(set(agent_actions)) >= l2


def f_5(trajectory, value, agent_actions, n):
    """
    :param trajectory: list of states
    :param value: function that takes a state and returns a value (representing user's value)
    :param agent_actions: list of actions of the trained agent
    :param n: length of trajectory
    :param l2: number of actions that has to be different
    :return: list of actions
    """
    if len(trajectory) != n:
        return -float("inf")
    value_sum = 0
    for state, action in zip(trajectory, agent_actions):
        value_sum += value(state, action)
    return value_sum


def main():
    env = gym.make(f"{ENV}-v0")
    agents = load_agents(AGENT_DETAILS)
    policies = [policy_fast, policy_slow]

    for agent_name, agent in agents.items():
        print(f"Running {agent_name}")
        for i in range(NUM_SIMULATIONS):
            frames = []
            speeds = []
            actions = []
            trajectory = []
            agent_actions = []
            done = False
            obs = env.reset()
            e_length = 0
            while not done:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)

                # save my speed
                speeds.append(env.vehicle.speed)

                # save my action
                actions.append(action)

                frame = env.render(mode="rgb_array")
                frames.append(frame)

                e_length += 1
                trajectory.append({"speed": env.vehicle.speed, "lane": env.vehicle.lane_index})
                agent_actions.append(action)

                for n in LIST_N:
                    for policy in policies:
                        f_1_result = f_1(trajectory[e_length - n:e_length], policy, agent_actions, n)
                        if f_1_result:
                            save_frames_as_gif(frames[e_length - n:e_length], speeds[e_length - n:e_length],
                                               actions[e_length - n:e_length], user=f_1_result,
                                               path=f"explanation/{policy.__name__}/{agent_name}/f_1/n={n}/",
                                               filename=f"{(i, e_length - n, e_length)}.gif")
                            print(f"saved {agent_name} on {policy.__name__} with n={n}")


if __name__ == "__main__":
    main()
