"""
utils_container.py

Brief description:
------------------
This module manages the creation and lifecycle of Docker containers for head-to-head agent matches.
It uses the Docker SDK for Python to create containers dynamically with specific environment variables
and networking settings, enabling agents to run in isolation and log results to a PostgreSQL database.

Functions:
----------
- create_container(image_name, command, agent_id_1, agent_id_2):
    Creates and runs a Docker container for a match between two agents.
    If a container with the same name already exists, it will be removed and replaced.

Usage:
------
Call `create_container` with the name of the Docker image, the command to run inside the container,
and the two agent IDs that are competing in the match. A container will be created on the custom
Docker network `rl_network`, and will have access to the host's PostgreSQL service via `host.docker.internal`.

Example:
--------
    container_id = create_container(
        image_name="rl_match_runner",
        command="python run_match.py --agent_id_1 1 --agent_id_2 2 --match_id 42",
        agent_id_1=1,
        agent_id_2=2
    )

Notes:
------
- Assumes the Docker network `rl_network` already exists.
- Requires the PostgreSQL server to be accessible via `host.docker.internal` from within the container.
- Automatically handles container name conflicts by removing old containers with the same name.

Author: Ambrose Ling  
Date: 2025-05-25
"""



import docker
import psycopg2

client = docker.from_env()

def create_container(image_name, command, agent_id_1, agent_id_2):
    container_config = {
        'image': image_name,
        'command': command,
        'detach': True,
        'name': f"game_container-{agent_id_1}-{agent_id_2}",
        'environment': {
            'DATABASE_URL': 'postgres://postgres:postgres@host.docker.internal:5432/rl_db',
            'AGENT_ID_1': str(agent_id_1),
            'AGENT_ID_2': str(agent_id_2)
        },
        'network': 'rl_network',
        'networking_config': client.api.create_networking_config({
            'rl_network': client.api.create_endpoint_config()
        }),
        'extra_hosts': {
            'host.docker.internal': 'host-gateway'
        }
    }
    
    try:
        container = client.containers.run(**container_config)
        return container.id
    except docker.errors.APIError as e:
        # Handle potential name conflicts
        if 'Conflict' in str(e):
            # Remove existing container if it exists
            try:
                old_container = client.containers.get(f"game_container-{agent_id_1}-{agent_id_2}")
                old_container.remove(force=True)
                # Retry container creation
                container = client.containers.run(**container_config)
                return container.id
            except:
                raise e
        raise e

