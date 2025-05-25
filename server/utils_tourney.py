"""
utils_tourney.py

Brief description:
------------------
This module provides a simple interface for creating a double-elimination tournament
using the `DoubleEliminationTournament` class. It serves as a wrapper to encapsulate
tournament initialization logic.

Functions:
----------
- create_tournament(teams):
    Initializes and returns a new instance of `DoubleEliminationTournament` with the given list of teams.

Usage:
------
Import this module and call `create_tournament()` with a list of team names or objects.

Example:
--------
    from tournament_initializer import create_tournament

    teams = ["Team A", "Team B", "Team C", "Team D"]
    tournament = create_tournament(teams)

Notes:
------
- The `DoubleEliminationTournament` class must be defined in the `double_elimination_tournament` module.
- Each team should be a unique identifier (string or object) representing a competitor.

Author: Ambrose Ling  
Date: 2025-05-25
"""

from double_elimination_tournament import DoubleEliminationTournament

def create_tournament(teams):
    return DoubleEliminationTournament(teams)

