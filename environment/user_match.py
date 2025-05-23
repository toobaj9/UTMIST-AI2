import skvideo
import skvideo.io
from IPython.display import Video
from agent import RecurrentPPOAgent, BasedAgent, UserInputAgent, run_match, CameraResolution,  gen_reward_manager

def get_model(experiment_dir, name):
    if name == 'based':
        return BasedAgent()
    if name.isdigit():
        name = f'rl_model_{name}_steps'
    return RecurrentPPOAgent(file_path=f'checkpoints/{experiment_dir}/{name}.zip')

reward_manager = gen_reward_manager()

experiment_dir_1 = input('Model experiment directory name (e.g. experiment_1): ')
model_name_1 = input('Name of first model (e.g. rl_model_100_steps): ')

my_agent = get_model(experiment_dir_1, model_name_1)
opponent = UserInputAgent()

num_matches = int(input('Number of matches: '))
#opponent=BasedAgent()
match_time = 90
for i in range(num_matches):
    run_match(my_agent,
            agent_2=opponent,
            video_path=f'vis_{i}.mp4',
            agent_1_name='Agent 1',
            agent_2_name='Agent 2',
            resolution=CameraResolution.LOW,
            reward_manager=reward_manager,
            max_timesteps=30 * match_time,
            train_mode=True
            )
