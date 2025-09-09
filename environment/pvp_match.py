import skvideo
import skvideo.io
from IPython.display import Video
from agent import SB3Agent, RecurrentPPOAgent, BasedAgent, UserInputAgent, ConstantAgent, run_match, run_real_time_match, CameraResolution,  gen_reward_manager

reward_manager = gen_reward_manager()

experiment_dir_1 = "model" #input('Model experiment directory name (e.g. experiment_1): ')
model_name_1 = "rl_model_81688323_steps_cracked" #input('Name of first model (e.g. rl_model_100_steps): ')

my_agent = UserInputAgent()
opponent = BasedAgent()

num_matches = 1 #int(input('Number of matches: '))
#opponent=BasedAgent()
match_time = 500
# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 270,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)