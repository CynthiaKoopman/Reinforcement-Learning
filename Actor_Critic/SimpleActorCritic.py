#!/usr/bin/env python
# coding: utf-8

# ## Implementation of a simple Actor-Critic algorithm
# ### Using Cartpole environment provided by gym

import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd


# neural network
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        # critic layers
        torch.manual_seed(1)
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        #actor layers
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        torch.manual_seed(1)
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)) # state to tensor
        value = F.relu(self.critic_linear1(state)) # go through first layer
        value = self.critic_linear2(value) # go through last layer
        
        policy_dist = F.relu(self.actor_linear1(state)) # go through first layer
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1) # go through last layer, and turn into probabilities

        return value, policy_dist # return value (given state) & action prob_dist (given state)


def take_steps(actor_critic, env, max_steps, entropy_term): 
    num_outputs = env.action_space.n
    action_space = np.arange(env.action_space.n)
    done = False
    state = env.reset()
    states, rewards, logprobs, values = [], [], [], []
    
    for step in range(max_steps):
        value, policy_dist = actor_critic.forward(state)
        value = value.detach().numpy()[0,0]
        dist = policy_dist.detach().numpy() 

        action = np.random.choice(num_outputs, p=np.squeeze(dist))
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))
        new_state, reward, done, _ = env.step(action)
        
        states.append(state)
        rewards.append(reward)
        values.append(value)
        logprobs.append(log_prob)
        entropy_term += entropy
        state = new_state
        
        if done or step == max_steps-1:
            #save only the last next_value
            next_value, _ = actor_critic.forward(new_state)
            next_value = next_value.detach().numpy()[0,0]
            break # breaks if episode ended or max number of steps is achieved
                  #(can also chose to continue but need to save several next values then)

    
    return states, rewards, logprobs, values, next_value, entropy_term


def get_Qvals(Qvals, rewards, gamma, next_value):
    for t in reversed(range(len(rewards))):
        next_value = rewards[t] + gamma * next_value
        Qvals[t] = next_value
    return Qvals

def test_net(actor_critic, env, count=10):
    rewards = 0.0
    action_space = np.arange(env.action_space.n)
    for _ in range(count): # test for count episodes
        state = env.reset()
        while True:
            _, policy_dist = actor_critic.forward(state)
            dist = policy_dist.detach().numpy() 
            action = np.random.choice(action_space, p=np.squeeze(dist))
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
    return rewards / count # return average reward of the episodes


def a2c(env, hidden_size, learning_rate, gamma, max_steps, total_steps):
    all_rewards = []
    test_returns = []
    
    #initialize model and optimizer
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size) # can call predict(state)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    
    #define others
    action_space = np.arange(env.action_space.n)
    entropy_term = 0 

    for idx in range(total_steps):
        #get some steps
        states, rewards, logprobs, values, next_value, entropy_term = take_steps(actor_critic, env, max_steps, entropy_term)
        
        #save and print progress
        all_rewards.append(sum(rewards))
        if idx % 10 == 0:
            test = test_net(actor_critic, env, 10)
            test_returns.append(test)
            print("step: ", idx," return: ", test)
            
        Qvals = np.zeros_like(values)
        Q = get_Qvals(Qvals, rewards, gamma, next_value)
        
        # convert to tensors
        values = torch.FloatTensor(values)
        Q = torch.FloatTensor(Q)
        logprobs = torch.stack(logprobs)
        
        #calculate Advantage
        A = Q - values
        
        #calculate loss for actor-critic
        actor_loss = (-logprobs * A).mean()
        critic_loss = 0.5 * A.pow(2).mean() # MSE
        ac_loss = (actor_loss + critic_loss) - 0.01 * entropy_term
        
        #update actor-critic parameters
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()  
    
    
    return all_rewards, test_returns

env = gym.make("CartPole-v0")
env.seed(0)
all_rewards, test_returns = a2c(env, hidden_size = 256, learning_rate = 3e-4, gamma = 0.99, max_steps = 200, total_steps = 1000)

smooth_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
smooth_rewards = [elem for elem in smooth_rewards]

plt.plot(all_rewards)
plt.plot(smooth_rewards)
plt.ylabel("reward")
plt.xlabel("steps")
plt.show()

plt.plot(test_returns)
plt.ylabel("reward")
plt.xlabel("steps")
plt.show()

env = gym.make("CartPole-v0")
env.seed(0)
all_rewards, test_returns = a2c(env, hidden_size = 256, learning_rate = 3e-4, gamma = 0.99, max_steps = 50, total_steps = 2000)

smooth_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
smooth_rewards = [elem for elem in smooth_rewards]

plt.plot(all_rewards)
plt.plot(smooth_rewards)
plt.ylabel("reward")
plt.xlabel("steps")
plt.show()

plt.plot(test_returns)
plt.ylabel("reward")
plt.xlabel("steps")
plt.show()

env = gym.make("CartPole-v0")
env.seed(0)
all_rewards, test_returns = a2c(env, hidden_size = 256, learning_rate = 3e-4, gamma = 0.99, max_steps = 150, total_steps = 2000)

smooth_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
smooth_rewards = [elem for elem in smooth_rewards]

plt.plot(all_rewards)
plt.plot(smooth_rewards)
plt.ylabel("reward")
plt.xlabel("steps")
plt.show()

plt.plot(test_returns)
plt.ylabel("reward")
plt.xlabel("steps")
plt.show()
