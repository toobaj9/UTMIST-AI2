import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Run RL Tournament battle.")
    parser.add_argument("--agent1", required=True, help="Path to first agent file")
    parser.add_argument("--agent2", required=True, help="Path to second agent file")
    args = parser.parse_args()

    print(f"Loading agents: {args.agent1} vs {args.agent2}")

    # Dynamically import and instantiate both agents
    Agent1 = load_agent_class(args.agent1)
    Agent2 = load_agent_class(args.agent2)

    agent1_instance = Agent1()
    agent2_instance = Agent2()

    print("✅ Both agents successfully instantiated.")
    print(f"{Agent1.__name__} vs {Agent2.__name__}")

    # Example match logic
    # (Replace this with your environment logic or RL arena logic)
    # print("Starting match...")

    # print("Match complete! Reporting results...")


if __name__ == "__main__":
    main()
