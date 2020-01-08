import gym
import gym_SnakeGame

from numpy_rl import DQN, FCNPolicy

env = gym.make("CartPole-v0")
model = DQN(env=env, policy=FCNPolicy)

# model.train(1000000, render=False, model_dir="./models/", log_dir="./logs/", trained_model_path='models\\model-253000.weight')
model.test(model_path='models\\model-5000.weight')