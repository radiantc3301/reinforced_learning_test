import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
 
environment_name = 'CartPole-v0'
env = gym.make("CartPole-v1", render_mode="rgb_array")

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
 
    while not done:
        env.render()
        action = env.action_space.sample()
        # print(env.step(action))
        state, reward, done, info, x = env.step(action)
        score += reward
    print(f'Episode: {episode}, Score: {score}')

env.close()