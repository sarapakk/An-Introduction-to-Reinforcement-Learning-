Python 3.8.8 (tags/v3.8.8:024d805, Feb 19 2021, 13:18:16) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Policy_network(nn.Module):
    def __init__(self,num_inputs, num_actions, hidden1, learning_rate = 5e-3):
        super(Policy_network, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions


        self.fc1 = nn.Linear(self.num_inputs, hidden1)
        self.fc2 = nn.Linear(hidden1, self.num_actions)
        #self.fc3 = nn.Linear(hidden2, self.num_actions)
        #self.fc4 = nn.Linear(hidden3, self.num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc2(x)
        x = F.softmax(x, dim = 1)
        return x


    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


def update_policy(policy_network, rewards, log_probs, gamma):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + gamma ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    #policy_gradient = []
    policy_loss = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * Gt)

    policy_loss = torch.stack(policy_loss).mean()
    policy_network.optimizer.zero_grad()
    #policy_loss = torch.cat(policy_loss).sum()

    policy_loss.backward()
    policy_network.optimizer.step()
        #del policy.rewards[:]
        #del policy.saved_log_probs[:]
        #policy_gradient.append(-log_prob * Gt)

    #policy_network.optimizer.zero_grad()
    #policy_gradient = torch.stack(policy_gradient).sum()
    #policy_gradient.backward()
    #policy_network.optimizer.step()


def main():
    gamma = 0.99
    env = gym.make('CartPole-v0')
    #env.max_episode_steps = 2000
    policy_net = Policy_network(4, 2, 64)
    #optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
    max_episode_num = 5000
    #max_steps = 200
    numsteps = []
    avg_numsteps = []
    all_rewards = []
    for episode in range(max_episode_num):
        #print(episode)
        done = False
        state = env.reset()
        log_probs = []
        rewards = []
        #if episode % 10 ==0:
        #env.render()
        steps = 0
        while not done:
        #for steps in range(max_steps):
            if episode % 100 ==0:
                env.render()
                #print(episode)

            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            steps +=1
            state = new_state
        if steps > 190:
            print(steps)
        #print(steps)

            #if done:
        update_policy(policy_net, rewards, log_probs, gamma)
        numsteps.append(steps)
        avg_numsteps.append(np.mean(numsteps[-10:]))
        all_rewards.append(np.sum(rewards))
                #if episode % 1 == 0:
                #    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,                                                                                                decimals=3),
                #                                                                                              steps))
                #break

            #state = new_state

 #   plt.plot(numsteps)
 #   plt.plot(avg_numsteps)
 #   plt.xlabel('Episode')
 #   plt.show()


if __name__ == '__main__':
    main()


    
