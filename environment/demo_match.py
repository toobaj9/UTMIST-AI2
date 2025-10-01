import skvideo
import skvideo.io
from IPython.display import Video
from environment import RenderMode
from agent import RecurrentPPOAgent, BasedAgent, run_match, CameraResolution,  gen_reward_manager, SubmittedAgent

def get_model(experiment_dir, name):
    if name == 'based':
        return BasedAgent()
    if name.isdigit():
        name = f'rl_model_{name}_steps'
    return SubmittedAgent(file_path=f'checkpoints/{experiment_dir}/{name}.zip')

reward_manager = gen_reward_manager()

experiment_dir_1 = input('First model experiment directory name (e.g. experiment_1): ')
experiment_dir_2 = input('Second model experiment directory name (e.g. experiment_1): ')
model_name_1 = input('Name of first model (e.g. rl_model_100_steps): ')
model_name_2 = input('Name of second model (e.g. rl_model_100_steps): ')

my_agent = get_model(experiment_dir_1, model_name_1)
opponent = get_model(experiment_dir_2, model_name_2)

num_matches = int(input('Number of matches: '))
filename = input('Filename for video (e.g. vis): ')
#opponent=BasedAgent()
match_time = 90
for i in range(num_matches):
    run_match(my_agent,
            agent_2=opponent,
            video_path=f'{filename}_{i}.mp4',
            agent_1_name='Agent 1',
            agent_2_name='Agent 2',
            mode=RenderMode.PYGAME_WINDOW,
            resolution=CameraResolution.LOW,
            reward_manager=reward_manager,
            max_timesteps=30 * match_time,
            train_mode=True
            )
