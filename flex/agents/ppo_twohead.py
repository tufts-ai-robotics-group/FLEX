import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class MultiheadActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(MultiheadActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space 
        self._train = False
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = {
                torch.full((action_dim,), action_std_init * action_std_init).to(device)
            }
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh()
                        )
            self.actor_head = {
                'strength': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Tanh()
                ), 
                'direction': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Tanh()
                )
            }
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh()
                        )
            self.actor_head = {
                'magnitude': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Softmax(dim=-1)
                ), 
                'direction': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Softmax(dim=-1)
                )
            }
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64), 
                        nn.Tanh(), 
                        nn.Linear(64, 64), 
                        nn.Tanh(), 
                        nn.Linear(64, 64) 
                    ) 
        self.critic_head = {
                'magnitude': nn.Sequential(
                    nn.Linear(64, 1), 
                    nn.Tanh()
                ), 
                'direction': nn.Sequential(
                    nn.Linear(64, 1), 
                    nn.Tanh()
                )
            }
    def train(self):
        self._train = True 

    def eval(self):
        self._train = False

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        actor_feat = self.actor(state)
        critic_feat = self.critic(state)

        if self.has_continuous_action_space:
            action_mean = {
                k: self.actor_head[k](actor_feat) for k in self.actor_head.keys()
            }
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = {
                k: MultivariateNormal(action_mean[k], cov_mat) for k in action_mean.keys()
            }
        else:
            action_probs = {
                k: self.actor_head[k](actor_feat) for k in self.actor_head.keys()
            }
            dist = {
                k: Categorical(action_probs[k]) for k in action_probs.keys()
            }
        if self._train:
            action = {
                k: dist[k].sample().detach() for k in dist.keys()
            } 
        else:
            action = {
                k: dist[k].mode for k in dist.keys()
            }
        # action = dist.sample() 
        # action = dist.mode
        action_logprob = {
            k: dist[k].log_prob(v).detach() for k, v in action.items()
        }
        state_val = {
            k: self.critic_head[k](critic_feat).detach() for k in self.critic_head.keys()
        }

        # real_force = action['direction'] / nn.functional.normalize(action['direction']) * action['magnitude']

        return action, action_logprob, state_val
    
    def evaluate(self, state, action):
        
        action_feat = self.actor(state)
        critic_feat = self.critic(state)

        if self.has_continuous_action_space:
            action_mean = {
                k: self.actor_head[k](action_feat) for k in self.actor_head.keys()
            }
            
            action_var = {
                k: self.action_var.expand_as(action_mean) for k in self.actor_head.keys()
            }
            cov_mat = {
                k: torch.diag_embed(action_var[k]).to(device) for k in action_var.keys()
            }
            dist = {
                k: MultivariateNormal(action_mean[k], cov_mat) for k in action_mean.keys()
            }
            
            # For Single Action Environments.
            if self.action_dim == 1:
                for k, v in action.items():
                    v = v.reshape(-1, self.action_dim)
        else:
            action_probs = {
                k: self.actor_head[k](action_feat) for k in self.actor_head.keys()
            }
            dist = {
                k: Categorical(action_probs[k]) for k in action_probs.keys()
            }
        action_logprobs = {
            k: dist[k].log_prob(v) for k, v in action.items()
        }
        dist_entropy = {
            k: v.entropy() for k, v in dist.items()
        }
        state_values = {
            k: self.critic_head[k](critic_feat) for k in self.critic_head.keys()
        }
        
        return action_logprobs, state_values, dist_entropy


class MultiheadPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = MultiheadActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = MultiheadActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = {
            k: torch.squeeze(torch.stack(self.buffer.logprobs[k], dim=0)).detach().to(device) for k in self.buffer.logprobs.keys()
        }
        old_state_values = {
            k: torch.squeeze(torch.stack(self.buffer.state_values[k], dim=0)).detach().to(device) for k in self.buffer.state_values.keys()
        }

        # calculate advantages
        advantages = {
            k: rewards.detach() - old_state_values[k].detach() for k in old_state_values.keys()
        }

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = {
                k: torch.squeeze(v) for k, v in state_values.items()
            }
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = {
                k: torch.exp(logprobs[k] - old_logprobs[k].detach()) for k in logprobs.keys()
            }

            # Finding Surrogate Loss  
            surr1 = {
                k: ratios[k] * advantages[k] for k in ratios.keys()
            }
            surr2 = {
                k: torch.clamp(ratios[k], 1-self.eps_clip, 1+self.eps_clip) * advantages[k] for k in ratios.keys()
            }

            # final loss of clipped objective PPO
            loss = 0
            for k in surr1.keys():
                loss += -torch.min(surr1[k], surr2[k]) + 0.5 * self.MseLoss(state_values[k], rewards) - 0.01 * dist_entropy[k]
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

