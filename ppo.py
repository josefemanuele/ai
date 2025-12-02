import gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import torch
from torch import nn
from network import FeedForwardNN
from torch.distributions import MultivariateNormal

class PPO:
    def __init__(self, env):
        self.env = env
        # for discrete action spaces
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.obs_dim = env.observation_space.n
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0]
            self.obs_dim = env.observation_space.shape[0]
        print(f"Observation space dimension: {self.obs_dim}, Action space dimension: {self.act_dim}")
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        # Set hyperparameters
        self._init_hyperparameters()
        # 0.5 is chosen arbitrarily
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Create covariance matrix for multivariate normal distribution
        self.cov_mat = torch.diag(self.cov_var)
        # Optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 64
        self.max_timesteps_per_episode = 32
        self.n_updates_per_iteration = 5
        self.batch_size = 64
        self.lr = 0.005
        self.clip = 0.2
        self.target_kl = 0.01
        self.gamma = 0.99
        self.lam = 0.97

    def get_action(self, obs):
        # Get mean action from actor network
        mean = self.actor(obs)
        # Create multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample action from the distribution
        action = dist.sample()
        # Get log probability of the action
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rewards):
        batch_rtgs = []
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0
            rtgs = []
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                rtgs.insert(0, discounted_reward)
            batch_rtgs += rtgs
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        # Query critic for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()
        # Calculate log probs of actions taken
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
    
    def scalar_to_one_hot(self, value, size):
        vec = [0] * size
        vec[value] = 1
        vec = torch.tensor(vec, dtype=torch.float)
        return vec
    
    def get_max_action(self, action):
        max = action[0]
        max_index = 0
        for i in range(1, len(action)):
            if action[i] > max:
                max = action[i]
                max_index = i
        return max_index

    def rollout(self):
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtgs = []
        batch_lens = []
        # Number of timesteps collected so far
        t = 0
        while t < self.timesteps_per_batch:
            ep_rewards = []
            obs, _ = self.env.reset()
            obs = self.scalar_to_one_hot(obs, size=self.obs_dim)
            #obs = torch.tensor(obs, dtype=torch.float)
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                action = self.get_max_action(action)
                obs, reward, done, _, _ = self.env.step(action)
                obs = self.scalar_to_one_hot(obs, size=self.obs_dim)
                obs = torch.tensor(obs, dtype=torch.float)
                action = self.scalar_to_one_hot(action, size=self.act_dim)
                action = torch.tensor(action, dtype=torch.float)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rewards.append(reward)
                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)
            if done:
                print(ep_rewards)
            print(f"t: {t} Episode length: {ep_t + 1}, total reward: {sum(ep_rewards)}")
        # Reshape data as tensors
        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.stack(batch_acts)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # Compute rewards-to-go
        batch_rtgs = self.compute_rtgs(batch_rewards)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            # Alg setp 3: collect batch of data
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            # Alg step 5: compute advantages
            A_k = batch_rtgs - V.detach()
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            for _ in range(self.n_updates_per_iteration):
                # Evaluate old actions and values
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                # Actor loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                # Update actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()
                # Critic loss
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                # Update critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            t_so_far += sum(batch_lens)

# env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8))
env = gym.make("CartPole-v1")
ppo = PPO(env)
ppo.learn(total_timesteps=100000)