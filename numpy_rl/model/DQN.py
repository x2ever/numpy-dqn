import gym
import numpy as np
import os
import sys
import random
import cv2
import copy
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from policy.Policy import Policy
from collections import deque



class DQN:
    def __init__(self, env: gym.Env, policy: Policy, replay_memory_len=1000000, batch_size=256):
        self.env = env
        self.env.reset()
        state_n = self.env.state_size
        action_n = self.env.action_size

        self.Q = policy(input_n=state_n, output_n=action_n)
        self.target_Q = policy(input_n=state_n, output_n=action_n)
        self.memory = deque()
        self.batch_size = batch_size
        self.update_epoch = 1000
        self.replay_memory_len = replay_memory_len
        self.alpha = 0.65
        self.e = 1.0

    def train(self, epoch, ready_epoch=10000, render=False, log_dir=None, model_dir=None, trained_model_path=None):
        ep_rewards = list()
        costs = list()
        if trained_model_path:
            with open(trained_model_path, 'rb') as f:
                layers = pickle.load(f)

            self.Q.layers = layers
            self.target_Q.layers = layers

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(log_dir + "\\log.txt", 'w') as f:
                pass

        for i in range(epoch):
            state = self.env.reset()
            ep_reward = 0
            ep_len = 0
            while True:
                ep_len += 1
                flat_state = np.reshape(state, (-1))
                if i > ready_epoch and self.e <= np.random.random():
                    Q_value = self.Q.predict(flat_state)
                    copy_Q_val = copy.copy(Q_value)
                    copy_Q_val -= np.min(copy_Q_val) - 0.1
                    action = random.choices(population=[0, 1, 2, 3], weights=copy_Q_val)[0]
                else:
                    self.e *= 0.999
                    action = np.random.randint(0, 4)
                next_state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                if render and i > ready_epoch:
                    img = self.env.render()
                    cv2.imshow("Game", img)
                    cv2.waitKey(1)
                
                flat_next_state = np.reshape(next_state, (-1))
                self.memory.append((flat_state, flat_next_state, action, reward, done))
                if len(self.memory) > self.replay_memory_len:
                    self.memory.popleft()

                if done:
                    if render:
                        cv2.destroyAllWindows()
                    break

            if i > ready_epoch:
                state, next_state, action, reward, terminal = self._sample_memory()
                target_Q_value = self.target_Q.predict(next_state)
                Q_value = self.Q.predict(state)

                Y = list()
                for j in range(self.batch_size):
                    if terminal[j]:
                        Q_value[j, action[j]] = (1 - self.alpha) * Q_value[j, action[j]] + self.alpha * reward[j]
                        Y.append(Q_value[j])
                    else:
                        Q_value[j, action[j]] = (1 - self.alpha) * Q_value[j, action[j]] + self.alpha * (reward[j] + 0.99 * np.max(target_Q_value[j]))
                        Y.append(Q_value[j])

                state = np.array(state)
                Y = np.array(Y)
                cost = self.Q.train(state, Y)

                print("[Epsode: {:>8}] Reward: {:>25} Cost: {:>25} Episode Length: {:>4}".format(
                    i, ep_reward, cost, ep_len
                ), end="\r")
                ep_rewards.append(ep_reward)
                costs.append(cost)
            else:
                print("[Epsode: {:>8}] Exploring until {:>6}".format(
                    i, ready_epoch                    
                ))

            if i % self.update_epoch == 0 and i != 0 and i > ready_epoch:
                self.Q.layers = self.target_Q.layers
                if model_dir:
                    with open(model_dir + "\\model-%d.weight" % i, 'wb') as f:
                        pickle.dump(self.Q.layers, f)
                
                if log_dir:
                    with open(log_dir + "\\log.txt", 'a') as f:
                        data = "%d\t%f\t%f\n" % (i, sum(ep_rewards) / len(ep_rewards), sum(costs) / len(costs))
                        f.write(data)



                print(f"\nFrom {i - self.update_epoch} to {i},")
                print(f"\tMean Reward: {sum(ep_rewards) / len(ep_rewards)}")
                print(f"\tMean Cost: {sum(costs) / len(costs)}")
                ep_rewards = list()
                costs = list()

    def test(self, model_path=None, epoch=None):
        if model_path:
            with open(model_path, 'rb') as f:
                layers = pickle.load(f)
            self.Q.layers = layers
        state = self.env.reset()
        while True:
            img = self.env.render()
            cv2.imshow("Game", img)
            cv2.waitKey(1000)
            flat_state = np.reshape(state, (-1))
            Q_value = self.Q.predict(flat_state)
            action = np.argmax(Q_value)
            next_state, reward, done, _ = self.env.step(action)
            flat_next_state = np.reshape(next_state, (-1))
            state = flat_next_state
            if done:
                cv2.destroyAllWindows()
                break


    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.batch_size)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

if __name__ == "__main__":
    import gym_SnakeGame
    from policy.FCNPolicy import FCNPolicy
    env = gym.make("SnakeGame-v0")
    model = DQN(env=env, policy=FCNPolicy)