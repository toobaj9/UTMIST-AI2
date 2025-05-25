"""
tournament_server.py

Brief description:
------------------
This Flask server manages a Reinforcement Learning (RL) competition using a double-elimination tournament format.
Teams submit RL agents, which are validated and run inside Docker containers. The tournament executes matches between 
teams and records match outcomes.

This server exposes RESTful API endpoints to manage team registration, agent submissions, match execution, and tournament progression.

Modules Used:
-------------
- Flask: For creating API endpoints
- Docker SDK: For running agents in isolated containers
- SQLAlchemy: ORM for database interactions
- Custom modules: `double_elimination`, `models`, and `utils_container`

Main Endpoints:
---------------
- GET `/`: Health check
- GET `/teams`: Retrieve all registered teams
- POST `/create_team`: Register a new team
- GET `/submissions`: List all submissions for a team
- POST `/submit`: Submit an agent link to be tested and stored
- GET `/matches`: View all recorded matches
- GET `/start_tournament`: Begin the double elimination tournament
- GET `/docker-health`: Verify if Docker is running

Functions:
----------
- `validate_submission_with_tester(agent_link)`: Validates submitted agent via Docker container
- `launch_match(match_id, team_id_1, team_id_2)`: Launches a match between two teams in Docker
- `after_request(response)`: Adds CORS headers
- Flask route handlers: Serve API endpoints described above

Example:
--------
    # Register a team
    POST /create_team?team_name=Alpha&team_id=team123

    # Submit an agent
    POST /submit?agent_id=team123&agent_link=https://storage.example.com/agent.py

    # Start tournament
    GET /start_tournament

Author: Ambrose Ling  
Date: 2025-05-25
"""




from flask import Flask, request, jsonify, render_template
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from double_elimination import Tournament
from models import Submission, Team, Match
from utils_container import create_container
import docker
from datetime import datetime
import psycopg2
import time
import uuid
from __init__ import create_app, db

client = docker.from_env()
app = create_app()

def validate_submission_with_tester(agent_link):
    """Download and test the submitted agent against a tester."""
    try:
        response = request.get(agent_link, timeout=10)
        if response.status_code != 200:
            print("Failed to download the agent.")
            return False
        
        # Check if Docker is installed and accessible
        try:
            client = docker.from_env()
            client.ping()  # Ensure Docker daemon is running
        except docker.errors.DockerException as e:
            print(f"Docker is not running or accessible: {e}")
            return False

        # Run the agent in a sandboxed Docker container (without auto-removal)
        container = client.containers.run(
            image="rl-agent-simulator",
            command=f"python3 simul_agents.py --agent_file {agent_link} --test_mode",
            mem_limit="512m",  # Memory limit to prevent OOM
            cpu_period=100000,
            cpu_quota=50000,  # Limit CPU usage
            detach=True,
            stderr=True,
            stdout=True
        )
        
        exit_code = container.wait(timeout=10)["StatusCode"]  # Enforce timeout
        logs = container.logs().decode("utf-8")

        if exit_code != 0:
            print(f"Agent failed with logs: {logs}")
            return False

        print(f"Container {container.id} finished successfully.")
        return True
    except docker.errors.ContainerError as e:
        print(f"Container execution failed: {e}")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")
    except request.RequestException as e:
        print(f"Error downloading agent: {e}")
    except OSError as e:
        print(f"OS-related error (Docker or file system issue): {e}")
    return False


def launch_match(match_id,team_id_1,team_id_2):
    try:
        container_id = create_container(
            image_name="rl-agent-simulator",
            command="python3 simul_agents.py --agent_id_1 {} --agent_id_2 {} --match_id {}".format(team_id_1,team_id_2,match_id),
            agent_id_1=team_id_1,
            agent_id_2=team_id_2
        )
        print(f"========== We just created container {container_id} for match {team_id_1} vs {team_id_2} ==========")

        return container_id
    except Exception as e:
        print(e)
        return None

@app.after_request
def after_request(response):
    allowed_origins = ["http://192.168.2.33:8080","http://localhost:8080","https://rl-server-gui.lovable.app/"]
    origin = request.headers.get('Origin')
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Server is running"}), 200

@app.route("/teams", methods=["GET"])
def teams():
    return jsonify({"teams": [team.to_dict() for team in Team.query.all()]}), 200

@app.route("/submissions", methods=["GET"])
def submissions():
    team_id = request.args.get('team_id')
    submissions = Submission.query.filter_by(team_id=team_id).all()
    return jsonify({"submissions": [submission.to_dict() for submission in submissions]}), 200

@app.route("/matches", methods=["GET"])
def matches():
    return jsonify({"matches": [match.to_dict() for match in Match.query.all()]}), 200

@app.route("/docker-health", methods=["GET"])
def docker_health():
    try:
        client.ping()
        return jsonify({"status": "Docker is running", "connected": True}), 200
    except Exception as e:
        return jsonify({
            "status": "Docker is not running or not accessible",
            "connected": False,
            "error": str(e)
        }), 503

@app.route("/create_team",methods=["POST"])
def create_team():
    team_name = request.args.get('team_name')
    team_id = request.args.get('team_id')
    if Team.query.filter_by(id=team_id).first():
        return jsonify({"status": "error", "message": "Team already exists"}), 400
    else:
        team = Team(id=team_id,name=team_name)
        db.session.add(team)
        db.session.commit()
        return jsonify({"status": "success", "team_id": team.id}), 200

@app.route("/submit", methods=["POST"])
def submit():
    agent_id = request.args.get('agent_id')
    agent_link = request.args.get('agent_link')
    if not validate_submission_with_tester(agent_link):
        return jsonify({"status": "error", "message": "Invalid or faulty agent submission"}), 400
    submission_id = (int(agent_id) + uuid.uuid4().int) & 0x0FFFFFFF

    #TODO: run tests
    # assert that the agent_link is a valid link to downloadable python code that we can run
    # assert that when we run the code in a docker container and check that it runs successfully and does not crash
             
    submission = Submission(id=submission_id, team_id=agent_id, blob_url=agent_link)
    db.session.add(submission)
    db.session.commit()
    return jsonify({"status": "success", "submission_id": submission.id}), 200

@app.route("/start_tournament", methods=["GET"])
def start_tournament():
    #TODO: run tests
    # assert that each team has at least 1 successful submission ran against tester
    round = 1
    timeout = 180  # 3 minutes in seconds
    teams = Team.query.all()

    for team in teams:
        if not Submission.query.filter_by(team_id=team.id).first():
            return jsonify({"status": "error", "message": f"Team {team.id} has no valid submissions"}), 400
        
        for submission in Submission.query.filter_by(team_id=team.id).first():
                dangerous_patterns = [r"os\.system\(\s*[\"']rm\s+-rf\s+/[\"']\s*\)", 
                                      r"shutil\.rmtree\(\s*['\"]/\s*['\"]\)", 
                                    r"subprocess\.(call|run|Popen)\(\s*[\"']rm\s+-rf\s+/[\"']"]
            with open(submission.blob_url, "r", encoding = "utf-8") as file:
                submission_code = file.read()
            
        for pattern in dangerous_patterns:
            if re.search(pattern, submission_code):
                return jsonify({
                    "status": "error",
                    "message": f"Submission contains unsafe commands."
                }), 400
                
        
    # Create a tournament object with the teams id
    tournament = Tournament([team.id for team in teams])
    # Get the active matches for the first round
    matches = tournament.get_active_matches()
    # While there are matches to run, we will keep running them
    while len(matches) > 0:
        print(f"========== We have {len(matches)} matches to run in this round ==========")
        num_matches = len(matches)
        # Go through each match of the current round
        for match in matches:
            # Create a match id
            match_id = uuid.uuid4().int & 0x000FFFFF
            setattr(match, "match_id", match_id)
            # Launch the match in a docker container
            container_id = launch_match(match_id,match.get_participants()[0].get_competitor(),match.get_participants()[1].get_competitor())
            
            # If the container fails to launch, we will return an error
            if container_id is None:
                return jsonify({"status": "error", "message": "Failed to launch match"}), 500
            print(f"========== We are running match {match_id} ==========")
            # Add the match to the database
            match_entry = Match(id=match_id,
                                team_id_1=match.get_participants()[0].get_competitor(),
                                team_id_2=match.get_participants()[1].get_competitor(),
                                container_id=container_id,
                                status="running",
                                result="pending",
                                frames_blob_url="na",
                                round_num=round)
            db.session.add(match_entry)
            db.session.commit()
            # Wait for the match to finish
        # We check that the number of matches done is equal to the number of matches we started THIS round
        while len(Match.query.filter_by(status="complete").all()) != num_matches: 
            time.sleep(1)
            print(f"========== We have {len(Match.query.filter_by(status='complete').all())} matches that have finished this round ==========")
            if timeout == 0:
                return jsonify({"status": "error", "message": "Timeout"}), 500
            timeout -= 1
        # Go through each match of the current round
        print(f"========== We have {len(matches)} matches completed this round ==========")

        for match in matches:
            # Get the match entry from the database
            match_entry = Match.query.filter_by(id=match.match_id).first()
            match_entry.status = "done"
            db.session.commit()
            # If the match entry has a result, we will add the win to the tournament
            if match_entry.result == match_entry.team_id_1:
                tournament.add_win(match,match.get_participants()[0].get_competitor())
            else:
                tournament.add_win(match,match.get_participants()[1].get_competitor())
        round += 1
        matches = tournament.get_active_matches()
    print(f"========== We are done with the tournament ==========")
    return jsonify({"status": "success"}), 200


if __name__ == "__main__":
    app.run(
        host="127.0.0.1", 
        port=5000, 
        debug=True, 
        threaded=True,
    )

