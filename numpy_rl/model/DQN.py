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
    def __init__(self, env: gym.Env, policy: Policy, replay_memory_len=100000, batch_size=256):
        self.env = env
        state_n = self.env.state_size
        self.action_n = self.env.action_size

        self.Q = policy(input_n=state_n, output_n=self.action_n)
        self.target_Q = policy(input_n=state_n, output_n=self.action_n)
        self.memory = deque()
        self.batch_size = batch_size
        self.update_interval = 1000
        self.replay_memory_len = replay_memory_len

    def train(self, epoch, ready_epoch=300, render=False, log_dir=None, model_dir=None, trained_model_path=None):
        ep_rewards = list()
        costs = list()
        e = 1
        if trained_model_path:
            with open(trained_model_path, 'rb') as f:
                layers = pickle.load(f)

            self.Q.layers = layers
            e = 0.05

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(log_dir + "\\log.txt", 'w') as f:
                pass
        self.target_Q.layers = np.copy(self.Q.layers)

        step = 0
        for i in range(1, epoch + 1):
            state = self.env.reset()
            ep_reward = 0
            ep_len = 0
            cost = 0
            while True:
                ep_len += 1
                step += 1
                flat_state = np.reshape(state, (-1))
                if e > 0.1 and i > ready_epoch:
                    e -= 0.0001

                if i > ready_epoch and e <= np.random.random():
                    Q_value = self.Q.predict([flat_state])
                    action = np.argmax(Q_value)
                else:
                    action = np.random.randint(0, self.action_n)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.copy(next_state)
                ep_reward += reward
                if render and i > ready_epoch:
                    img = self.env.render()
                    cv2.imshow("Game", img)
                    cv2.waitKey(1)
                
                flat_next_state = np.reshape(next_state, (-1))
                self.memory.append((flat_state, flat_next_state, action, reward, done))
                if len(self.memory) > self.replay_memory_len:
                    self.memory.popleft()

                if i > ready_epoch:
                    states, next_states, action, reward, terminal = self._sample_memory()

                    next_states = np.array(next_states)
                    target_Q_value = self.target_Q.predict(next_states)
                    states = np.array(states)
                    Q_value = self.Q.predict(states)

                    Y = list()
                    for j in range(self.batch_size):
                        if terminal[j]:
                            Q_value[j, action[j]] = reward[j]
                            Y.append(Q_value[j])
                        else:
                            Q_value[j, action[j]] = reward[j] + 0.99 * np.max(target_Q_value[j])
                            Y.append(Q_value[j])

                    Y = np.array(Y)
                    cost = self.Q.train(states, Y)
                    costs.append(cost)

                    if step % self.update_interval == 0:
                        self.target_Q.layers = np.copy(self.Q.layers)

                if done:
                    if render:
                        cv2.destroyAllWindows()
                    break
                state = next_state
            
            if i <= ready_epoch:
                continue

            if i % 1 == 0:
                self_e = "%.5f" % e
                print("[Epsode: {:>8}] Reward: {:>5} epsilon: {:>5} Cost: {:>25} Episode Length: {:>4}".format(
                    i, ep_reward, self_e, cost, ep_len
                ), end="\r")
            ep_rewards.append(ep_reward)
            if model_dir and (i % 1000 == 0):
                with open(model_dir + "\\model-%d.weight" % i, 'wb') as f:
                    pickle.dump(self.Q.layers, f)
            
            if log_dir and (i % 50 == 0):
                with open(log_dir + "\\log.txt", 'a') as f:
                    data = "%d\t%f\t%f\n" % (i, sum(ep_rewards) / len(ep_rewards), sum(costs) / len(costs))
                    f.write(data)


            if i % 200 == 0:
                print(f"\nFrom {i - 200} to {i},")
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