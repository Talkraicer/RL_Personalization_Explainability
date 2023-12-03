import gym
from rl_agents.agents.deep_q_network.pytorch import DQNAgent
import highway_env
from rl_agents.trainer.evaluation import Evaluation
from tal_reward_functions import *

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
              maximum_episode_length=100,
              replay_buffer=dict(type="PrioritizedReplayBuffer",
                                    alpha=0.6,
                                    beta=0.4,
                                    beta_schedule=dict(type="LinearSchedule",
                                                        initial_p=0.4,
                                                        final_p=1.0,
                                                        schedule_timesteps=1000000)),


              double=True)

env = gym.make("Fast-v0")
env.config["high_speed_reward"] = 4
env.config["collision_reward"] = -20
env.config["keep_distance_reward"] = 1
env.config["gas_reward"] = 0

agent = DQNAgent(env,config)  # default configuration -  can be altered

evaluation = Evaluation(env, agent, display_env=False, num_episodes=10000)
evaluation.save_every = 5

evaluation.train()
