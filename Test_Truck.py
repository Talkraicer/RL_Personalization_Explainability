import gym
from rl_agents.agents.deep_q_network.pytorch import DQNAgent
import highway_env
from rl_agents.trainer.evaluation import Evaluation
from gif_saver import save_frames_as_gif
from tal_reward_functions import *
import numpy as np
import matplotlib.pyplot as plt

ENV = "Truck"
RUN = "trained"

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


env = gym.make(f"{ENV}-v0")


agent = DQNAgent(env,config)

agent.load(f"out/{ENV}/DQNAgent/{RUN}/checkpoint-best.tar")

evaluation = Evaluation(env, agent, display_env=False, num_episodes=1)

evaluation.test()

frames = []
speeds = []
for i in range(1,11):
    done = False
    obs = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        # save my speed
        speeds.append(env.vehicle.speed)

        frame = env.render(mode="rgb_array")
        frames.append(frame)

save_frames_as_gif(frames, speeds, f"{ENV}-{RUN}.gif")