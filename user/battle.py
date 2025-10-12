import pytest 
from loguru import logger
import importlib.util
import os
import sys

def load_agent_class(file_path):
    """Dynamically load SubmittedAgent class from a given Python file."""
    file_path = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load module spec and import dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Expecting a class named SubmittedAgent in the file
    if not hasattr(module, "SubmittedAgent"):
        raise AttributeError(f"File {file_path} does not define a SubmittedAgent class.")

    return module.SubmittedAgent

@pytest.mark.timeout(300) 
def test_agent_batte():
    # Get paths to the agents
    logger.info(f"Loading agents: ")
    agent_1_path = os.getenv("AGENT1_PATH")
    agent_2_path = os.getenv("AGENT2_PATH")
    assert agent_1_path is not None and agent_2_path is not None, "Could not find path to agents"

    # Dynamically import and instantiate both agents
    Agent1 = load_agent_class(agent_1_path)
    Agent2 = load_agent_class(agent_2_path)

    agent1_instance = Agent1()
    agent2_instance = Agent2()

    logger.info("✅ Both agents successfully instantiated.")
    logger.info(f"{Agent1.__name__} vs {Agent2.__name__}")
