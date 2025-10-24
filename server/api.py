import os
import sys
from typing import Optional
from supabase import create_client


def validate_battle(username1, username2) -> bool:
    """Validate two participants for a battle."""
    return check_validation_status(username1) and check_validation_status(username2)

def check_validation_status(username: str) -> bool:
    """Check if a participant has passed validation."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    existing = client.table("ai2_leaderboard").select("validation_status").eq("username", username).execute()
    rows = []
    if hasattr(existing, "data") and isinstance(existing.data, list):
        rows = existing.data
    if rows:
        return bool(rows[0]["validation_status"])
    return False

def update_validation_status(username: str, status: bool) -> None:
    """Update the validation status for a participant."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    resp = client.table("ai2_leaderboard").update({"validation_status": status}).eq("username", username).execute()
    if hasattr(resp, "error") and resp.error:
        # supabase-py may return error object/message
        raise RuntimeError(str(resp.error))

def create_participant(username: str) -> None:
    """Create a participant in the ai2_leaderboard table."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    
    # Exit early if user already exists
    existing = client.table("ai2_leaderboard").select("username").eq("username", username).execute()
    rows = []
    if hasattr(existing, "data") and isinstance(existing.data, list):
        rows = existing.data
    elif hasattr(existing, "data") and isinstance(existing.data, dict):
        # Some client versions may return a dict when using RPCs/single; normalize
        rows = [existing.data]

    if rows:
        return

    resp = client.table("ai2_leaderboard").insert({"username": username, "elo": 1000}).execute()
    if hasattr(resp, "error") and resp.error:
        # supabase-py may return error object/message
        raise RuntimeError(str(resp.error))

def elo_update(elo1, elo2, result, k=32):
    """
    Update two ELO scores based on result.
    :param elo1: Player 1 ELO
    :param elo2: Player 2 ELO
    :param result: 1 if player 1 wins, 0 if player 1 loses, 0.5 for draw
    :param k: K-factor (default 32)
    :return: (new_elo1, new_elo2)
    """
    # Calculate expected scores
    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))
    # Update ratings
    new_elo1 = elo1 + k * (result - expected1)
    new_elo2 = elo2 + k * ((1 - result) - expected2)
    return new_elo1, new_elo2


def get_participant_elo(username: str) -> int:
    """Read ELO for a participant from the ai2_leaderboard table."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    if not key:
        raise RuntimeError("Missing Supabase key in environment (SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY)")

    client = create_client(url, key)

    resp = client.table("ai2_leaderboard").select("elo").eq("username", username).single().execute()
    if getattr(resp, "data", None) and "elo" in resp.data:
        return resp.data["elo"]
    raise ValueError(f"No ELO found for user: {username}")

def update_participant_elo(username: str, elo: int, match_result: Optional[str] = None) -> None:
    """Update ELO for a participant in the ai2_leaderboard table.

    match_result is accepted for compatibility but not used here.
    """
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    resp = client.table("ai2_leaderboard").update({"elo": elo}).eq("username", username).execute()

    if hasattr(resp, "error") and resp.error:
        # supabase-py may return error object/message
        raise RuntimeError(str(resp.error))
        
def main(argv: list[str]) -> int:
    """
    CLI: python server/elo.py <username1> <username2>
    Prints ELO for two users on separate lines (for Actions step usage).
    """
    u1 = argv[1]

    create_participant(u1)
    is_valid = check_validation_status(u1)
    if not is_valid:
        print(f"User {u1} has not passed validation", file=sys.stderr)
        return 1


    # if len(argv) != 3:
    #     print("Usage: python server/elo.py <username1> <username2>", file=sys.stderr)
    #     return 2
    # u1, u2 = argv[1], argv[2]
    # try:
    #     e1 = get_participant_elo(u1)
    #     e2 = get_participant_elo(u2)
    # except Exception as e:
    #     print(f"Error: {e}", file=sys.stderr)
    #     return 1

    # try:
    #     update_participant_elo(u1, 1234)
    #     update_participant_elo(u2, 2345)
    # except Exception as e:
    #     print(f"Error: {e}", file=sys.stderr)
    #     return 1

    # print("ELOs updated successfully")
    # print(f"ELO for {u1}: {get_participant_elo(u1)}")
    # print(f"ELO for {u2}: {get_participant_elo(u2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


