from environment import ActHelper, AirTurnaroundState, Animation, AnimationSprite2D, AttackState, BackDashState, Camera, CameraResolution, Capsule, CapsuleCollider, Cast, CastFrameChangeHolder, CasterPositionChange, CasterVelocityDampXY, CasterVelocitySet, CasterVelocitySetXY, CompactMoveState, DashState, DealtPositionTarget, DodgeState, Facing, GameObject, Ground, GroundState, HurtboxPositionChange, InAirState, KOState, KeyIconPanel, KeyStatus, MalachiteEnv, MatchStats, MoveManager, MoveType, ObsHelper, Particle, Player, PlayerInputHandler, PlayerObjectState, PlayerStats, Power, RenderMode, Result, Signal, SprintingState, Stage, StandingState, StunState, Target, TauntState, TurnaroundState, UIHandler, WalkingState, WarehouseBrawl, hex_to_rgb
# ### Imports

# In[ ]:


import warnings
from typing import TYPE_CHECKING, Any, Generic, \
 SupportsFloat, TypeVar, Type, Optional, List, Dict, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, MISSING
from collections import defaultdict
from functools import partial
from typing import Tuple, Any

from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

import gdown, os, math, random, shutil, json

import numpy as np
import torch
from torch import nn

import gymnasium
from gymnasium import spaces

import pygame
import pygame.gfxdraw
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

import cv2
import skimage.transform as st
import skvideo
import skvideo.io
from IPython.display import Video

from stable_baselines3.common.monitor import Monitor


# ## Agents

# ### Agent Abstract Base Class

# In[ ]:


SelfAgent = TypeVar("SelfAgent", bound="Agent")

class Agent(ABC):

    def __init__(
            self,
            file_path: Optional[str] = None
        ):

        # If no supplied file_path, load from gdown (optional file_path returned)
        if file_path is None:
            file_path = self._gdown()

        self.file_path: Optional[str] = file_path
        self.initialized = False

    def get_env_info(self, env):
        if isinstance(env, Monitor):
            self_env = env.env
        else:
            self_env = env
        self.observation_space = self_env.observation_space
        self.obs_helper = self_env.obs_helper
        self.action_space = self_env.action_space
        self.act_helper = self_env.act_helper
        self.env = env
        self._initialize()
        self.initialized = True

    def get_num_timesteps(self) -> int:
        if hasattr(self, 'model'):
            return self.model.num_timesteps
        else:
            return 0

    def update_num_timesteps(self, num_timesteps: int) -> None:
        if hasattr(self, 'model'):
            self.model.num_timesteps = num_timesteps

    @abstractmethod
    def predict(self, obs) -> spaces.Space:
        pass

    def save(self, file_path: str) -> None:
        return

    def reset(self) -> None:
        return

    def _initialize(self) -> None:
        """

        """
        return

    def _gdown(self) -> Optional[str]:
        """
        Loads the necessary file from Google Drive, returning a file path.
        Or, returns None, if the agent does not require loaded files.

        :return:
        """
        return


# ### Agent Classes

# In[ ]:


class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


# ## StableBaselines3 Integration

# ### Reward Configuration

# In[ ]:


@dataclass
class RewTerm():
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """



# In[ ]:


class RewardManager():
    """Reward terms for the MDP."""

    # (1) Constant running reward
    def __init__(self,
                 reward_functions: Optional[Dict[str, RewTerm]]=None,
                 signal_subscriptions: Optional[Dict[str, Tuple[str, RewTerm]]]=None) -> None:
        self.reward_functions = reward_functions
        self.signal_subscriptions = signal_subscriptions
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0

    def subscribe_signals(self, env) -> None:
        if self.signal_subscriptions is None:
            return
        for _, (name, term_cfg) in self.signal_subscriptions.items():
            getattr(env, name).connect(partial(self._signal_func, term_cfg))

    def _signal_func(self, term_cfg: RewTerm, *args, **kwargs):
        term_partial = partial(term_cfg.func, **term_cfg.params)
        self.collected_signal_rewards += term_partial(*args, **kwargs) * term_cfg.weight


    def process(self, env, dt) -> float:
        # reset computation
        reward_buffer = 0.0
        # iterate over all the reward terms
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                # skip if weight is zero (kind of a micro-optimization)
                if term_cfg.weight == 0.0:
                    continue
                # compute term's value
                value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
                # update total reward
                reward_buffer += value

        reward = reward_buffer + self.collected_signal_rewards
        self.collected_signal_rewards = 0.0

        self.total_reward += reward

        log = env.logger[0]
        log['reward'] = f'{reward_buffer:.3f}'
        log['total_reward'] = f'{self.total_reward:.3f}'
        env.logger[0] = log
        return reward

    def reset(self):
        self.total_reward = 0
        self.collected_signal_rewards


# ### Save, Self-play, and Opponents

# In[ ]:


class SaveHandlerMode(Enum):
    FORCE = 0
    RESUME = 1

class SaveHandler():
    """Handles saving.

    Args:
        agent (Agent): Agent to save.
        save_freq (int): Number of steps between saving.
        max_saved (int): Maximum number of saved models.
        save_dir (str): Directory to save models.
        name_prefix (str): Prefix for saved models.
    """

    # System for saving to internet

    def __init__(
            self,
            agent: Agent,
            save_freq: int=10_000,
            max_saved: int=20,
            run_name: str='experiment_1',
            save_path: str='checkpoints',
            name_prefix: str = "rl_model",
            mode: SaveHandlerMode=SaveHandlerMode.FORCE
        ):
        self.agent = agent
        self.save_freq = save_freq
        self.run_name = run_name
        self.max_saved = max_saved
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.mode = mode

        self.steps_until_save = save_freq
        # Get model paths from exp_path, if it exists
        exp_path = self._experiment_path()
        self.history: List[str] = []
        if self.mode == SaveHandlerMode.FORCE:
            # Clear old dir
            if os.path.exists(exp_path) and len(os.listdir(exp_path)) != 0:
                while True:
                    answer = input(f"Would you like to clear the folder {exp_path} (SaveHandlerMode.FORCE): yes (y) or no (n): ").strip().lower()
                    if answer in ('y', 'n'):
                        break
                    else:
                        print("Invalid input, please enter 'y' or 'n'.")

                if answer == 'n':
                    raise ValueError('Please switch to SaveHandlerMode.FORCE or use a new run_name.')
                print(f'Clearing {exp_path}...')
                if os.path.exists(exp_path):
                    shutil.rmtree(exp_path)
            else:
                print(f'{exp_path} empty or does not exist. Creating...')

            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
        elif self.mode == SaveHandlerMode.RESUME:
            if os.path.exists(exp_path):
                # Get all model paths
                self.history = [os.path.join(exp_path, f) for f in os.listdir(exp_path) if os.path.isfile(os.path.join(exp_path, f))]
                # Filter any non .csv
                self.history = [f for f in self.history if f.endswith('.zip')]
                if len(self.history) != 0:
                    self.history.sort(key=lambda x: int(os.path.basename(x).split('_')[-2].split('.')[0]))
                    if max_saved != -1: self.history = self.history[-max_saved:]
                    print(f'Best model is {self.history[-1]}')
                else:
                    print(f'No models found in {exp_path}.')
                    raise FileNotFoundError
            else:
                print(f'No file found at {exp_path}')


    def update_info(self) -> None:
        self.num_timesteps = self.agent.get_num_timesteps()

    def _experiment_path(self) -> str:
        """
        Helper to get experiment path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, self.run_name)

    def _checkpoint_path(self, extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self._experiment_path(), f"{self.name_prefix}_{self.num_timesteps}_steps.{extension}")

    def save_agent(self) -> None:
        print(f"Saving agent to {self._checkpoint_path()}")
        model_path = self._checkpoint_path('zip')
        self.agent.save(model_path)
        self.history.append(model_path)
        if self.max_saved != -1 and len(self.history) > self.max_saved:
            os.remove(self.history.pop(0))

    def process(self) -> bool:
        self.num_timesteps += 1

        if self.steps_until_save <= 0:
            # Save agent
            self.steps_until_save = self.save_freq
            self.save_agent()
            return True
        self.steps_until_save -= 1

        return False

    def get_random_model_path(self) -> str:
        if len(self.history) == 0:
            return None
        return random.choice(self.history)

    def get_latest_model_path(self) -> str:
        if len(self.history) == 0:
            return None
        return self.history[-1]

class SelfPlayHandler(ABC):
    """Handles self-play."""

    def __init__(self, agent_partial: partial):
        self.agent_partial = agent_partial
    
    def get_model_from_path(self, path) -> Agent:
        if path:
            try:
                opponent = self.agent_partial(file_path=path)
            except FileNotFoundError:
                print(f"Warning: Self-play file {path} not found. Defaulting to constant agent.")
                opponent = ConstantAgent()
        else:
            print("Warning: No self-play model saved. Defaulting to constant agent.")
            opponent = ConstantAgent()
        opponent.get_env_info(self.env)
        return opponent

    @abstractmethod
    def get_opponent(self) -> Agent:
        pass


class SelfPlayLatest(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_latest_model_path()
        return self.get_model_from_path(chosen_path)

class SelfPlayDynamic(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    @NotImplementedError
    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        assert self.save_handler.max_saved == -1, "Save handler must have max_saved=-1 for dynamic self-play (save all past opponents)"
        chosen_path = self.save_handler.get_random_model_path()
        return self.get_model_from_path(chosen_path)

class SelfPlayRandom(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_random_model_path()
        return self.get_model_from_path(chosen_path)

@dataclass
class OpponentsCfg():
    """Configuration for opponents.

    Args:
        swap_steps (int): Number of steps between swapping opponents.
        opponents (dict): Dictionary specifying available opponents and their selection probabilities.
    """
    swap_steps: int = 10_000
    opponents: dict[str, Any] = field(default_factory=lambda: {
                'random_agent': (0.8, partial(RandomAgent)),
                'constant_agent': (0.2, partial(ConstantAgent)),
                #'recurrent_agent': (0.1, partial(RecurrentPPOAgent, file_path='skibidi')),
            })

    def validate_probabilities(self) -> None:
        total_prob = sum(prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values())

        if abs(total_prob - 1.0) > 1e-5:
            print(f"Warning: Probabilities do not sum to 1 (current sum = {total_prob}). Normalizing...")
            self.opponents = {
                key: (value / total_prob if isinstance(value, float) else (value[0] / total_prob, value[1]))
                for key, value in self.opponents.items()
            }

    def process(self) -> None:
        pass

    def on_env_reset(self) -> Agent:

        agent_name = random.choices(
            list(self.opponents.keys()),
            weights=[prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values()]
        )[0]

        # If self-play is selected, return the trained model
        print(f'Selected {agent_name}')
        if agent_name == "self_play":
            selfplay_handler: SelfPlayHandler = self.opponents[agent_name][1]
            return selfplay_handler.get_opponent()
        else:
            # Otherwise, return an instance of the selected agent class
            opponent = self.opponents[agent_name][1]()

        opponent.get_env_info(self.env)
        return opponent


# ### Self-Play Warehouse Brawl

# In[ ]:


class SelfPlayWarehouseBrawl(gymnasium.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 reward_manager: Optional[RewardManager]=None,
                 opponent_cfg: OpponentsCfg=OpponentsCfg(),
                 save_handler: Optional[SaveHandler]=None,
                 render_every: int | None = None,
                 resolution: CameraResolution=CameraResolution.LOW):
        """
        Initializes the environment.

        Args:
            reward_manager (Optional[RewardManager]): Reward manager.
            opponent_cfg (OpponentCfg): Configuration for opponents.
            save_handler (SaveHandler): Configuration for self-play.
            render_every (int | None): Number of steps between a demo render (None if no rendering).
        """
        super().__init__()

        self.reward_manager = reward_manager
        self.save_handler = save_handler
        self.opponent_cfg = opponent_cfg
        self.render_every = render_every
        self.resolution = resolution

        self.games_done = 0


        # Give OpponentCfg references, and normalize probabilities
        self.opponent_cfg.env = self
        self.opponent_cfg.validate_probabilities()

        # Check if using self-play
        for key, value in self.opponent_cfg.opponents.items():
            if isinstance(value[1], SelfPlayHandler):
                assert self.save_handler is not None, "Save handler must be specified for self-play"

                # Give SelfPlayHandler references
                selfplay_handler: SelfPlayHandler = value[1]
                selfplay_handler.save_handler = self.save_handler
                selfplay_handler.env = self       

        self.raw_env = WarehouseBrawl(resolution=resolution, train_mode=True)
        self.action_space = self.raw_env.action_space
        self.act_helper = self.raw_env.act_helper
        self.observation_space = self.raw_env.observation_space
        self.obs_helper = self.raw_env.obs_helper

    def on_training_start(self):
        # Update SaveHandler
        if self.save_handler is not None:
            self.save_handler.update_info()

    def on_training_end(self):
        if self.save_handler is not None:
            self.save_handler.agent.update_num_timesteps(self.save_handler.num_timesteps)
            self.save_handler.save_agent()

    def step(self, action):

        full_action = {
            0: action,
            1: self.opponent_agent.predict(self.opponent_obs),
        }

        observations, rewards, terminated, truncated, info = self.raw_env.step(full_action)

        if self.save_handler is not None:
            self.save_handler.process()

        if self.reward_manager is None:
            reward = rewards[0]
        else:
            reward = self.reward_manager.process(self.raw_env, 1 / 30.0)

        return observations[0], reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset MalachiteEnv
        observations, info = self.raw_env.reset()

        self.reward_manager.reset()

        # Select agent
        new_agent: Agent = self.opponent_cfg.on_env_reset()
        if new_agent is not None:
            self.opponent_agent: Agent = new_agent
        self.opponent_obs = observations[1]


        self.games_done += 1
        #if self.games_done % self.render_every == 0:
            #self.render_out_video()

        return observations[0], info

    def render(self):
        img = self.raw_env.render()
        return img

    def close(self):
        pass


# ## Run Match

# In[ ]:


from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str]=None,
              agent_1_name: Optional[str]=None,
              agent_2_name: Optional[str]=None,
              mode=RenderMode.RGB_ARRAY,
              resolution = CameraResolution.LOW,
              reward_manager: Optional[RewardManager]=None,
              train_mode=False
              ) -> MatchStats:
    # Initialize env
    env = WarehouseBrawl(resolution=resolution, train_mode=train_mode)
    observations, infos = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]

    if reward_manager is not None:
        reward_manager.reset()
        reward_manager.subscribe_signals(env)

    if agent_1_name is None:
        agent_1_name = 'agent_1'
    if agent_2_name is None:
        agent_2_name = 'agent_2'

    env.agent_1_name = agent_1_name
    env.agent_2_name = agent_2_name


    writer = None
    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")
        # Initialize video writer
        writer = skvideo.io.FFmpegWriter(video_path, outputdict={
            '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
            '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
            '-preset': 'fast',  # Faster encoding
            '-crf': '20',  # Quality-based encoding (lower = better quality)
            '-r': '30'  # Frame rate
        })

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)
    # 596, 336

    for _ in tqdm(range(max_timesteps), total=max_timesteps):
        # actions = {agent: agents[agent].predict(None) for agent in range(2)}

        # observations, rewards, terminations, truncations, infos

        full_action = {
            0: agent_1.predict(obs_1),
            1: agent_2.predict(obs_2)
        }

        observations, rewards, terminated, truncated, info = env.step(full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        if reward_manager is not None:
            reward_manager.process(env, 1 / env.fps)

        if video_path is not None:
            img = env.render()
            writer.writeFrame(img)
            del img

        if terminated or truncated:
            break
        #env.show_image(img)

    if video_path is not None:
        writer.close()

    env.close()


    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)
    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=Result.WIN if player_1_stats.lives_left > player_2_stats.lives_left else Result.LOSS
    )

    del env

    return match_stats

# # SUBMISSION: Additional Imports
# Note that all the imports up to this point (for the Malachite Env, WarehouseBrawl, etc...) will be automatically included in the submission, so you need not write them.
# 
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here

# In[ ]:


from stable_baselines3 import PPO, A2C, SAC # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM


# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# 

# In[ ]:


# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!
class SubmittedAgent(Agent):

    def __init__(
            self,
            file_path: Optional[str] = None,
            # example_argument = 0,
    ):
        # Your code here
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            print('hii')
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
        else:
            self.model = PPO.load(self.file_path)
            # self.model = A2C.load(self.file_path)
            # self.model = SAC.load(self.file_path)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1G60ilYtohdmXsYyjBtwdzC1PRBerqpfJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)



# # Training
# 
# Here, you can set the reward functions and train your agent. If you'd like to write a heuristic (if-statement) agent, you can also reference the Example Agents here.

# ## Example Agent Classes
# Reference these to design a gamut of opponents for your model to face off against!

# In[ ]:


# Recall the possible observations
# Set name='player', or name='opponent'
# obs_helper.get_section(obs, f"{name}_pos") # low=[-1, -1], high=[1, 1]
# obs_helper.get_section(obs, f"{name}_facing") # low=[0], high=[1]
# obs_helper.get_section(obs, f"{name}_vel") # low=[-1, -1], high=[1, 1]
# obs_helper.get_section(obs, f"{name}_grounded") # low=[0], high=[1]
# obs_helper.get_section(obs, f"{name}_aerial") # low=[0], high=[1]
# obs_helper.get_section(obs, f"{name}_jumps_left") # low=[0], high=[2]
# obs_helper.get_section(obs, f"{name}_state") # low=[0], high=[12]
# obs_helper.get_section(obs, f"{name}_recoveries_left") # low=[0], high=[1]
# obs_helper.get_section(obs, f"{name}_dodge_timer") # low=[0], high=[1]
# obs_helper.get_section(obs, f"{name}_stun_frames") # low=[0], high=[1]
# obs_helper.get_section(obs, f"{name}_damage") # low=[0], high=[1]
# obs_helper.get_section(obs, f"{name}_stocks") # low=[0], high=[3]
# obs_helper.get_section(obs, f"{name}_move_type") # low=[0], high=[11]


# In[ ]:


# player_state and opponent_state map to this
# state_mapping = {
#             'WalkingState': 0,
#             'StandingState': 1,
#             'TurnaroundState': 2,
#             'AirTurnaroundState': 3,
#             'SprintingState': 4,
#             'StunState': 5,
#             'InAirState': 6,
#             'DodgeState': 7,
#             'AttackState': 8,
#             'DashState': 9,
#             'BackDashState': 10,
#             'KOState': 11,
#             'TauntState': 12,
#         }


# In[ ]:


class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action


# In[ ]:


class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


# In[ ]:


class BasedAgent(Agent):

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


# In[ ]:


class UserInputAgent(Agent):

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

        if keys[pygame.K_q]:
            action = self.act_helper.press_keys(['q'], action)
        if keys[pygame.K_v]:
            action = self.act_helper.press_keys(['v'], action)
        return action


# In[ ]:


class ClockworkAgent(Agent):

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
                (30, []),
                (7, ['d']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (20, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
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


# In[ ]:


from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

class SB3Agent(Agent):

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


# In[ ]:


from sb3_contrib import RecurrentPPO

class RecurrentPPOAgent(Agent):

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


# ## Training Function
# A helper function for training.

# In[ ]:


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TrainLogging(Enum):
    NONE = 0
    TO_FILE = 1
    PLOT = 2

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")

    weights = np.repeat(1.0, 50) / 50
    print(weights, y)
    y = np.convolve(y, weights, "valid")
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")

    # save to file
    plt.savefig(log_folder + title + ".png")

def train(agent: Agent,
          reward_manager: RewardManager,
          save_handler: Optional[SaveHandler]=None,
          opponent_cfg: OpponentsCfg=OpponentsCfg(),
          resolution: CameraResolution=CameraResolution.LOW,
          train_timesteps: int=400_000,
          train_logging: TrainLogging=TrainLogging.PLOT
          ):
    # Create environment
    env = SelfPlayWarehouseBrawl(reward_manager=reward_manager,
                                 opponent_cfg=opponent_cfg,
                                 save_handler=save_handler,
                                 resolution=resolution
                                 )
    reward_manager.subscribe_signals(env.raw_env)
    if train_logging != TrainLogging.NONE:
        # Create log dir
        log_dir = f"{save_handler._experiment_path()}/" if save_handler is not None else "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir)

    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    try:
        agent.get_env_info(base_env)
        base_env.on_training_start()
        agent.learn(env, total_timesteps=train_timesteps, verbose=1)
        base_env.on_training_end()
    except KeyboardInterrupt:
        pass

    env.close()

    if save_handler is not None:
        save_handler.save_agent()

    if train_logging == TrainLogging.PLOT:
        plot_results(log_dir)


# ## Example Reward Functions
# Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).

# In[ ]:


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

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0


# ## Run Training
# 
# Run this cell to run training. Be sure to set your agent under the `my_agent` variable, and modify the training using the `reward_manager`, `selfplay_handler`, `save_handler`, and `opponent_cfg`.

def gen_reward_manager():
        reward_functions = {
            #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
            'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
            'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0),
            #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
            #'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
            'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
            'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
            #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
        }
        signal_subscriptions = {
            'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
            'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
            'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        }
        return RewardManager(reward_functions, signal_subscriptions)

## Run Human vs AI match function
import pygame
from pygame.locals import QUIT

def run_real_time_match(agent_1: UserInputAgent, agent_2: Agent, max_timesteps=30*90, resolution=CameraResolution.LOW):
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))  # Set screen dimensions
    pygame.display.set_caption("AI Squared - Player vs AI Demo")
    clock = pygame.time.Clock()

    # Initialize environment
    env = WarehouseBrawl(resolution=resolution, train_mode=False)
    observations, _ = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]

    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)

    # Run the match loop
    running = True
    timestep = 0
    while running and timestep < max_timesteps:
        # Pygame event to handle real-time user input 
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # User input
        action_1 = agent_1.predict(obs_1)

        # AI input
        action_2 = agent_2.predict(obs_2)

        # Sample action space
        full_action = {0: action_1, 1: action_2}
        observations, rewards, terminated, truncated, info = env.step(full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        # Render the game
        img = env.render()
        screen.blit(pygame.surfarray.make_surface(img), (0, 0))
        pygame.display.flip()

        # Control frame rate (30 fps)
        clock.tick(30)

        # If the match is over (either terminated or truncated), stop the loop
        if terminated or truncated:
            running = False

        timestep += 1

    # Clean up pygame after match
    pygame.quit()

    # Return match stats
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)
    match_stats = MatchStats(
        match_time=timestep / 30.0,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=Result.WIN if player_1_stats.lives_left > player_2_stats.lives_left else Result.LOSS
    )

    # Close environment
    env.close()

    return match_stats

if __name__ == '__main__':
    # Create agent
    # Start here if you want to train from scratch
    my_agent = RecurrentPPOAgent()
    # Start here if you want to train from a specific timestep
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    

    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(RecurrentPPOAgent), # Agent class and its keyword arguments
    )

    # Save settings
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='experiment_6',
        mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
    )

    # Opponent settings
    opponent_specification = {
                    'self_play': (8, selfplay_handler),
                    'constant_agent': (0.5, partial(ConstantAgent)),
                    'based_agent': (1.5, partial(BasedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=1_000_000_000,
        train_logging=TrainLogging.PLOT
    )

