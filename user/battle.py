# Battle Workflow Script
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run RL Tournament battle.")
    parser.add_argument("--agent1", required=True, help="First agent path name")
    parser.add_argument("--agent2", required=True, help="Second agent path name")
    args = parser.parse_args()
    print(f"Success! Branches: {args.agent1} vs {args.agent2}")
    # Pulls models from some centralized DB
    # Load their models and get their agent code 
    # Run match with the agent class instances
    # Get match data post match and report it somewhere
if __name__ == "__main__":
    main()
