import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
 
environment_name = 'CartPole-v1'
log_path = os.path.join('Training', 'Logs')
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')   

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.save(PPO_Path)

model.learn(total_timesteps=20000)  

#model = PPO.load(PPO_Path, env=env)