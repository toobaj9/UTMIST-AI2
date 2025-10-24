import os
import pytest 
import skvideo
import skvideo.io
from loguru import logger
from IPython.display import Video
from environment.agent import UserInputAgent, ConstantAgent, run_match, CameraResolution,  gen_reward_manager
from user.my_agent import SubmittedAgent
try:
    from server.api import create_participant, update_validation_status
except ImportError:
    import sys
    import importlib.util
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    server_api_path = os.path.join(repo_root, 'server', 'api.py')
    spec = importlib.util.spec_from_file_location('server.api', server_api_path)
    api = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(api)
    create_participant = api.create_participant
    update_validation_status = api.update_validation_status

@pytest.mark.timeout(60) 
def test_agent_validation():
    username = os.getenv("USERNAME")
    create_participant(username)
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
    update_validation_status(username, True)
    logger.info("Validation match has completed successfully! Your agent is ready for battle!")

