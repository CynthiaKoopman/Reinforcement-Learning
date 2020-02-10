#!/usr/bin/env python
# coding: utf-8

# ## Implementation of PPO with GAE
# ### *heavily inpsired by OpenAI code
# This code makes use of a Memory to store various values, making GAE calculation easier. Weights are not shared between actor and critic. PPO is performed using several policy update iterations on the same batch, and early stopping using KL. Mini-batches are not used. 
# 
# Also please note that parameters have not been optimized fully.

# In[1]:


import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import matplotlib.pyplot as plt


# In[32]:


#actor crtic module
class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes = (64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        #self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation) #actor

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation) #critic

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# In[33]:


#critic
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


# In[35]:


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


# In[36]:


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


# In[37]:


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# In[38]:


class Memory:
    def __init__(self, obs_dim, act_dim, size, gamma, lambda_gae):
        # save obs, act, adv, rew, rewtg
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rewtg_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # define gamma and lambda
        self.gamma, self.lambda_gae = gamma, lambda_gae
        #define pointers to keep track of beginning and ending of episode (used for GAE)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        
    #make functions to: store, gae and get
    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size     # make sure that there is enough room in memory
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1 # move pointer
        
    def GAE(self, last_val=0):
        # take only steps that are part of that episode
        path_slice = slice(self.path_start_idx, self.ptr) # the indices
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lambda_gae)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.rewtg_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards to-go = discounted rewards?
        
        self.path_start_idx = self.ptr # reset pointer
        
    def get(self): # get all data in memory
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0 # reset pointers
        
        # advantage normalization (usually mean&std of all parallell workers)
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        
        # return data in seperate tensors
        data = dict(obs=self.obs_buf, act=self.act_buf, rewtg=self.rewtg_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        


# In[39]:


def ppo(env_fn, actor_critic=MLPActorCritic, hidden_sizes=(64,64), gamma=0.99, 
        seed=0, steps_per_epoch=200, epochs=2, pi_lr=3e-4, vf_lr=1e-3, lambda_gae=0.97, 
        max_ep_len=200, activation=nn.Tanh, train_v_iters=1, clip_ratio = 0.2, train_pi_iters = 5, target_kl = 0.01):
    
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, hidden_sizes, activation)
    
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    
    # Set up memory
    buf = Memory(obs_dim, act_dim, steps_per_epoch, gamma, lambda_gae)
    
    #--------------------------------------------------------------------------------------------------------------
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        #calculate the ratio
        ratio = torch.exp(logp - logp_old)
        # PPO clip:
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        # for loss take minimum, either clipped surr or unclipped ratio surr
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        
        return loss_pi, pi_info
    
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, rewtg = data['obs'], data['rewtg']
        return ((ac.v(obs) - rewtg)**2).mean() # (values - reward to go) MSE
    

    # Set up function for updating pi and v
    def update():
        data = buf.get()
        
        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            #instead of using minibatches, use KL early stopping
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()
        
        for i in range(train_v_iters):
            # Value function learning
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()
            
        kl, ent = pi_info['kl'], pi_info_old['ent']
        return kl, ent
        
    #--------------------------------------------------------------------------------------------------------------    
    # Prepare for interaction with environment
    o = env.reset()
    ep_len = 0
    ep_rew = 0
    all_ep_rew, avg_rew, kls, ents = [], [], [], []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32)) # get action based on initial obs

            next_o, r, d, _ = env.step(a) # get next obs
            ep_len += 1
            ep_rew += r
            
            # save and log
            buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = next_o
            
            # keep track of where you are in episode/batch
            timeout = ep_len == max_ep_len #end of episode
            terminal = d or timeout #done or end of episode
            epoch_ended = t==steps_per_epoch-1 #end of an epoch when all steps used (when terminal continue)
            
            if terminal or epoch_ended: # only for these cases need to save next value & reset env
                #if epoch_ended and not(terminal):
                    #print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32)) # get next value
                    all_ep_rew.append(ep_rew)
                else:
                    v = 0 # if terminal next value is 0
                    all_ep_rew.append(ep_rew)
                    
                buf.GAE(v) # calculate advantages using these values

                o, ep_len, ep_rew = env.reset(), 0, 0 # reset the env
                
        avg_rew.append(np.mean(all_ep_rew))
        if epoch % 10 == 0:
            print("Avg rewards at epoch ", epoch," : ", np.mean(all_ep_rew))
        # Perform policy update, every time step
        kl, ent = update()
        kls.append(kl)
        ents.append(ent)
    return avg_rew, kls, ents


# In[44]:


result, kls, ents = ppo(lambda : gym.make("LunarLanderContinuous-v2"), actor_critic=MLPActorCritic, hidden_sizes=(400,200), gamma=0.99, 
             seed=0, steps_per_epoch=500, epochs=500, pi_lr=1e-4, vf_lr=1e-3, lambda_gae=0.97, 
             max_ep_len=200, activation=nn.ReLU, train_v_iters = 30, clip_ratio = 0.2, train_pi_iters = 40, target_kl = 0.01)

plt.plot(result)
plt.xlabel("epoch")
plt.ylabel("avg reward")
plt.show()

plt.plot(kls)
plt.xlabel("epoch")
plt.ylabel("KL")
plt.show()

plt.plot(ents)
plt.xlabel("epoch")
plt.ylabel("entropy")
plt.show()


# In[ ]:





# In[ ]:




