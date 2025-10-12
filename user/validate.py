import pytest 
import skvideo
import skvideo.io
from loguru import logger
from IPython.display import Video
from environment.agent import UserInputAgent, ConstantAgent, run_match, CameraResolution,  gen_reward_manager
from user.my_agent import SubmittedAgent


@pytest.mark.timeout(60) 
def test_agent_validation():
    logger.info("Warming up your agent ...")
    my_agent = SubmittedAgent() 
    logger.info("Warming up your opponent's agent ...")
    opponent = ConstantAgent()
    match_time = 90
    reward_manager = gen_reward_manager()
    logger.info("Validation match has started ...")
    run_match(my_agent,
            agent_2=opponent,
            video_path=f'validate.mp4',
            agent_1_name='Agent 1',
            agent_2_name='Agent 2',
            resolution=CameraResolution.LOW,
            reward_manager=reward_manager,
            max_timesteps=30 * match_time,
            train_mode=True
            )
    logger.info("Validation match has completed successfully! Your agent is ready for battle!")

