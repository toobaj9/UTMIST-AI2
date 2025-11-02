'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=10,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )
    
class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def off_ground_penalty(
    env: WarehouseBrawl,
    boundary: float = 7,
    penalty: float = 1.0
) -> float:
    """Penalize being too close to screen edges."""
    player: Player = env.objects["player"]
    x_pos = player.body.position.x
    y_pos = player.body.position.y
    # Penalty increases as you get closer to edge
    if x_pos < -boundary * 0.9:
        return -penalty * (-boundary*0.9 - x_pos) * env.dt
    if (x_pos > -2 and y_pos > 0 and y_pos < 2.85) :  # Start penalty at 80% of max width
        return -penalty * (x_pos - (-2)) * env.dt
    if x_pos > boundary * 0.9:
        return -penalty * (x_pos - boundary*0.9) * env.dt
    if (x_pos < 2 and y_pos > 0 and y_pos < 2.85) :  # Start penalty at 80% of max width
        return -penalty * (2 - x_pos) * env.dt
    return 0.0


# TODO: This reward function has not been written and is left as an exercise to try and implement
#       yourself. Think about the following before implementing:
#
#       - While having a stock lead is generally good in fighting games,
#         how would this reward influence agent behaviour?
#       - Is this behaviour even desirable?
#       - Is this behaviour more valuable near the beggingin or end of the match,
#         and based on that answer how can you change the reward so it considers time?


def stock_advantage_reward(
    env: WarehouseBrawl,
    success_value: float = 0.5, #TODO
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:

    """
    Computes the reward given for every time step your agent is edge guarding the opponent.

    Args:
        env (WarehouseBrawl): The game environment
        success_value (float): Reward value related to having/gaining a weapon (however you define it)
    Returns:
        float: The computed reward.
    """
    reward = 0.0
    # TODO: Write the function
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    player_stocks = player.stocks
    opponent_stocks = opponent.stocks
    stock_diff = player_stocks - opponent_stocks
    damage_taken = env.objects["player"].damage_taken_this_stock
    damage_dealt = env.objects["opponent"].damage_taken_this_stock
    risk_factor = player.damage_taken_this_stock * 0.001 * env.dt
    if stock_diff > 0 and player_stocks > 0 and opponent_stocks > 0:
        if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
            reward = damage_dealt * 1.5
        elif mode == RewardMode.SYMMETRIC:
            reward = damage_dealt - damage_taken
        elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
            reward = (stock_diff * success_value * env.dt) - (damage_taken * 2.0) - risk_factor
        else:
            raise ValueError(f"Invalid mode: {mode}")
    elif stock_diff < 0 and player_stocks > 0 and opponent_stocks > 0:
        if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
            reward = damage_dealt * 2.0
        elif mode == RewardMode.SYMMETRIC:
            reward = damage_dealt - damage_taken
        elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
            opponent_vulnerability = opponent.damage_taken_this_stock * 0.001 * env.dt
            reward = (-abs(stock_diff) * success_value * env.dt) - (damage_taken * 1.5) + opponent_vulnerability
        else:
            raise ValueError(f"Invalid mode: {mode}")
    return reward

def jump_to_moving_platform(
    env: WarehouseBrawl,
    bonus: float = 1.0,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    reward = 0.0
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    platform = env.objects["platform1"]
    x, y = player.body.position.x, player.body.position.y
    px, py = platform.body.position.x, platform.body.position.y
    ox, oy = opponent.body.position.x, opponent.body.position.y

    platform_safe = not (px - 1.0 <= ox <= px + 1.0 and py - 0.2 <= oy <= py + 0.05)
    player_above_platform = (px - 1.0 <= x <= px + 1.0) and (py - 0.2 <= y <= py + 0.05)


    # Check if player is standing roughly above the platform
    if player_above_platform and platform_safe:
        reward += bonus * env.dt
    opponent_close = abs(ox - x) < 1.5 and abs(oy - y) < 1.5

    if opponent_close and player_above_platform and platform_safe:
        reward += 0.5 * env.dt
    if player_above_platform and not platform_safe and opponent_close:
        reward -= 0.5 * env.dt
    if player_above_platform and not platform_safe and not opponent_close:
        reward -= 0.2 * env.dt
    if player_above_platform and platform_safe and not opponent_close:
        reward -= 0.1 * env.dt
    return reward


def idle_penalty(
    env: WarehouseBrawl,
    penalty_per_second: float = 0.5,
    velocity_threshold: float = 0.05
) -> float:
    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
  
    vx = player.body.velocity.x
    vy = player.body.velocity.y
    
    speed = np.sqrt(vx**2 + vy**2)
    
    is_idling = (speed < velocity_threshold)
    
    reward = 0.0

    if is_idling:
        reward -= penalty_per_second * env.dt
        distance_to_opponent = np.sqrt(
            (player.body.position.x - opponent.body.position.x)**2 + 
            (player.body.position.y - opponent.body.position.y)**2 
        )
        
        # If the player is idling AND the opponent is far away (e.g., more than half the arena)
        WIDTH = 14.9 
        if distance_to_opponent > (WIDTH / 2.5):
            # Apply a heavier penalty for wasting time while separated
            reward -= penalty_per_second * 1.5 * env.dt
            
    return reward

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
#___________________________________________________________
#writing new reward functions
def move_to_opponent_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Computes the reward based on whether the agent is moving toward the opponent.
    The reward is calculated by taking the dot product of the agent's normalized velocity
    with the normalized direction vector toward the opponent.

    Args:
        env (WarehouseBrawl): The game environment

    Returns:
        float: The computed reward
    """
    # Getting agent and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Extracting player velocity and position from environment
    player_position_dif = np.array([player.body.position.x_change, player.body.position.y_change])

    direction_to_opponent = np.array([opponent.body.position.x - player.body.position.x,
                                      opponent.body.position.y - player.body.position.y])

    # Prevent division by zero or extremely small values
    direc_to_opp_norm = np.linalg.norm(direction_to_opponent)
    player_pos_dif_norm = np.linalg.norm(player_position_dif)

    if direc_to_opp_norm < 1e-6 or player_pos_dif_norm < 1e-6:
        return 0.0

    # Compute the dot product of the normalized vectors to figure out how much
    # current movement (aka velocity) is in alignment with the direction they need to go in
    reward = np.dot(player_position_dif / direc_to_opp_norm, direction_to_opponent / direc_to_opp_norm)

    return reward
#_________________________________________________________________
def edge_guard_reward(
    env: WarehouseBrawl,
    mode: RewardMode= RewardMode.SYMMETRIC,
) -> float:

    """
    Computes the reward given for every time step your agent is edge guarding the opponent.

    Args:
        env (WarehouseBrawl): The game environment
        success_value (float): Reward value for the player hitting first
        fail_value (float): Penalty for the opponent hitting first

    Returns:
        float: The computed reward.
    """
    #reward = 0.0
    #when the agent is close to x = -3 to x = -2 then there's reward
    #the agent, while being in x= -3 and x=-2 attacks the opponent
    #-------------------------------------------------
    #getting player and opponent
    player : Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    #SITUATION 1 when the player is on ground 1 and opponent is on ground 2
    #check if opponent is in the zone x=0 and x=-2
    opponent_in_zone_1 = -2 <= opponent.body.position.x <= 0
    #check is player is close to the edge, to guard, between x=-3 and x=-2
    player_in_position_1  = -3 <=player.body.position.x <= -2

    #SITUATION 2 when the player is on ground 2 and opponent is on ground 1
    #check if opponent is in the zone x=0 to x=2
    opponent_in_zone_2 = 0 <= opponent.body.position.x <= 2
    #check si player is close to the edge of ground 2, between x=2 and x=3
    player_in_position_2 = 2 <= player.body.position.x <= 3

    #if these conditions are met, then my agent should be in offensive mode
    if opponent_in_zone_1 and player_in_position_1:
        # Use the damage interaction reward logic for offensive behavior
        damage_taken = player.damage_taken_this_frame
        damage_dealt = opponent.damage_taken_this_frame
        if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
            reward = damage_dealt
        elif mode == RewardMode.SYMMETRIC:
            reward = damage_dealt - damage_taken
        elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
            reward = -damage_taken
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        # Add extra reward for maintaining the defensive position
        position_reward = 0.5
        return (reward / 140) + position_reward
    
    elif opponent_in_zone_2 and player_in_position_2:
        # Use the damage interaction reward logic for offensive behavior
        damage_taken = player.damage_taken_this_frame
        damage_dealt = opponent.damage_taken_this_frame
        if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
            reward = damage_dealt
        elif mode == RewardMode.SYMMETRIC:
            reward = damage_dealt - damage_taken 
        elif mode==RewardMode.ASYMMETRIC_DEFENSIVE:
            reward = -damage_taken
        else:
            raise ValueError(f"Invalid mode: {mode} ")
        #add extra reward for maintaining the defensive position
        position_reward = 0.5
        return (reward/140) + position_reward

    
    return 0.0
#____________________________________________________________
#function that when opponent throws spear, agent should jump
def jump_on_spear_throw(env: WarehouseBrawl, agent: str = 'player') -> float:
    """
    Rewards the agent for jumping when the opponent throws a spear. Gives a reward when:
    1. Opponent has a spear and presses H to throw it
    2. Agent is jumping at that moment

    Args:
        env (WarehouseBrawl): The game environment
        agent (str): Either 'player' or 'opponent' to identify which agent we're rewarding

    Returns:
        float: The computed reward
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    # Only reward if opponent is throwing spear (weapon is Spear and pressing H key)
    if opponent.weapon == "Spear" and opponent.input.key_status['h'].just_pressed:
        # Check if agent is jumping (in air + upward velocity)
        if isinstance(player.state, InAirState) and player.body.velocity.y < 0:
            return 1.5  # Reward for jumping during spear throw
    
    return 0.0  # No reward otherwise
#________________________________________________________________________
#reward function for using hammer when close to the opponent (offense), opponent is 1 unit close to the agent.
def close_to_agent(env: WarehouseBrawl,)-> float:
    '''
    Attacks using a hammer whenever the opponent is 1 unit close to the agent
    '''
    #getting player and opponent
    player : Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    #check if the agent has hammer
    if getattr(player, "weapon", None) != "Hammer":
        return 0.0
    if abs(player.body.position.x - opponent.body.position.x) <= 1:
        #check is attack key was pressed
        if player.input.key_status.get('j', None) and player.input.key_status['j'].just_pressed:
            return 1.0
        if player.input.key_status.get('k', None) and player.input.key_status['k'].just_pressed:
            return 1.0
    return 0.0
#_______________________________________________________________________________________


'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        'target_height_reward': RewTerm(func=base_height_l2, weight=-0.04, params={'target_height': -4, 'obj_name': 'player'}),
        #^^ since -4 would be way above the opponent, so going far from opponent is discouraged so -0.04 weight
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=2.0),
        'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.02), #encourages to stay away from edges and stay in middle
        'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.08), 
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
        'edge_guard_reward': RewTerm(func=edge_guard_reward, weight=1.0),
        'stock_advantage_reward': RewTerm(func=stock_advantage_reward, weight=1.5),
        'off_ground_penalty': RewTerm(func=off_ground_penalty, weight=0.5, params={'boundary': 7, 'penalty': 1.0}),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
        'jump_to_moving_platform': RewTerm(func=jump_to_moving_platform, weight=0.7),
        'idle_penalty': RewTerm(func=idle_penalty, weight=-0.2, params={'penalty_per_second': 0.5, 'velocity_threshold': 0.05}),
        #'move_to_opponent_reward': RewTerm(func=move_to_opponent_reward, weight= 0.5),
        'jump_on_spear_throw': RewTerm(func=jump_on_spear_throw, weight= 1.0),
        'close_to_agent': RewTerm(func=close_to_agent, weight = 1.0)


    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=15))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':

    # Create your agent (using PPO algorithm)
    #my_agent = SB3Agent(sb3_class=PPO)
    my_agent = SB3Agent(sb3_class=PPO, file_path='checkpoints/my_first_training/rl_model_4017600_steps.zip')
    # Use the existing reward manager
    reward_manager = gen_reward_manager() #defined above
    
    # Set up self-play for training
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent))
    )
    
    # Configure where to save your trained models
    save_handler = SaveHandler(
        agent=my_agent,
        save_freq=100_000,  # Save every 100k steps
        max_saved=40,
        save_path='checkpoints',
        run_name='my_first_training',
        mode=SaveHandlerMode.RESUME
    )
 # Set up training opponents
    opponent_specification = {
        'self_play': (8, selfplay_handler),
        'based_agent': (1.5, partial(BasedAgent)),
    }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)
    
    # Start training
    train(my_agent,
      reward_manager,
      save_handler,
      opponent_cfg,
      CameraResolution.LOW,
      train_timesteps=1000000,  # Train for 1M steps
      train_logging=TrainLogging.PLOT
    )
    
    # Create agent
    #my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    #reward_manager = gen_reward_manager()
    # Self-play settings
    #selfplay_handler = SelfPlayRandom(
        #partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    #)

    # Set save settings here:
    #save_handler = SaveHandler(
        #agent=my_agent, # Agent to save
        #save_freq=100_000, # Save frequency
        #max_saved=40, # Maximum number of saved models
        #save_path='checkpoints', # Save path
        #run_name='experiment_9',
        #mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
    #)

    # Set opponent settings here:
    #opponent_specification = {
                    #'self_play': (8, selfplay_handler),
                    #'constant_agent': (0.5, partial(ConstantAgent)),
                    #'based_agent': (1.5, partial(BasedAgent)),
                #}
    #opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    #train(my_agent,
        #reward_manager,
        #save_handler,
        #opponent_cfg,
        #CameraResolution.LOW,
        #train_timesteps=1_000_000_000,
        #train_logging=TrainLogging.PLOT
    #)
