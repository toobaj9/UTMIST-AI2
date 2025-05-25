"""
run_rl_agent.py

Brief description:
------------------
This module simulates the behavior of Reinforcement Learning (RL) agents competing in a match.
The simulation involves a dummy matrix multiplication workload using PyTorch to mimic agent inference time.
Once completed, the match result is randomly decided and the outcome is written to a PostgreSQL database.

This script is intended to be launched as a subprocess from the main tournament server for match execution.

Modules Used:
-------------
- torch: To simulate computation workload
- psycopg2: For PostgreSQL database interaction
- argparse: To handle command-line arguments
- os: For environment variable access
- random: To simulate probabilistic outcomes
- multiprocessing: (imported but not used in this script)

Function:
---------
- `run_rl_agent(agent_id_1, agent_id_2, match_id, blob_url_1, blob_url_2)`: 
    Simulates match logic, determines a winner, and updates the match result in the database.

Usage:
------
This script is meant to be run from the command line or a subprocess:

    python run_rl_agent.py --agent_id_1 101 --agent_id_2 102 --match_id 1

Example:
--------
    > python run_rl_agent.py --agent_id_1 10 --agent_id_2 11 --match_id 3

    Running matmul 0
    ...
    Agent 10 and 11 completed a computation cycle, result: 0.0123
    Agent 10 won
    Agent 10 and 11 wrote to DB

Note:
-----
- The match result is randomly chosen for simulation purposes.
- The `frames_blob_url` is hardcoded as a placeholder.

Author: Ambrose Ling  
Date: 2025-05-25
"""

import random
import argparse
import time
import os
import psycopg2
import torch
from multiprocessing import Process

# Simulate RL Agent
def run_rl_agent(agent_id_1,agent_id_2,match_id,blob_url_1,blob_url_2):
    try:
        a = torch.randn(20, 20)
        b = torch.randn(20, 20)
        for i in range(100): 
            print(f"Running matmul {i}")
            c = a @ b
        result = c.mean().item()  # Get a scalar value
        print(f"Agent {agent_id_1} and {agent_id_2} completed a computation cycle, result: {result:.4f}")
        db_url = os.getenv("DATABASE_URL")
        print(db_url)
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        # Update query
        update_query = """
            UPDATE matches 
            SET status = %s,
            result = %s,
            frames_blob_url = %s
            WHERE id = %s;
        """
        new_value = "complete"
        row_id = match_id  # The row to modify
        if random.random() > 0.5:   
            result = str(agent_id_1)
            print(f"Agent {agent_id_1} won")
        else:
            result = str(agent_id_2)
            print(f"Agent {agent_id_2} won")
        frames_blob_url = "https://www.google.com"
        cur.execute(update_query, (new_value, result, frames_blob_url, row_id))
        conn.commit()
        print(f"Agent {agent_id_1} and {agent_id_2} wrote to DB")

    except Exception as e:
        print(f"Agent {agent_id_1} and {agent_id_2} encountered an error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_id_1", type=int, required=True)
    parser.add_argument("--agent_id_2", type=int, required=True)
    parser.add_argument("--blob_url_1", type=str, required=False)
    parser.add_argument("--blob_url_2", type=str, required=False)
    parser.add_argument("--match_id", type=int, required=True)
    args = parser.parse_args()
    run_rl_agent(args.agent_id_1,args.agent_id_2,args.match_id,args.blob_url_1,args.blob_url_2)