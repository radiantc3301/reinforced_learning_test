import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v1'
log_path = os.path.join('Training', 'Logs')
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')

# Create the environment with a render mode
env = gym.make(environment_name, render_mode="human")

model = PPO.load(PPO_Path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)