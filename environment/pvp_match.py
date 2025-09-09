import skvideo
import skvideo.io
from IPython.display import Video
from environment import RenderMode
from agent import SB3Agent, CameraResolution, RecurrentPPOAgent, BasedAgent, UserInputAgent, ConstantAgent, run_match, gen_reward_manager

reward_manager = gen_reward_manager()

experiment_dir_1 = "model" #input('Model experiment directory name (e.g. experiment_1): ')
model_name_1 = "rl_model_81688323_steps_cracked" #input('Name of first model (e.g. rl_model_100_steps): ')

my_agent = BasedAgent()
opponent = BasedAgent()

num_matches = 1 #int(input('Number of matches: '))
#opponent=BasedAgent()
match_time = 500
# Run a single real-time match
run_match(my_agent,
    agent_2=opponent,
    video_path=f'test.mp4',
    agent_1_name='Agent 1',
    agent_2_name='Agent 2',
    mode=RenderMode.PYGAME_WINDOW,
    resolution=CameraResolution.LOW,
    reward_manager=reward_manager,
    max_timesteps=30 * match_time,
    train_mode=True
)