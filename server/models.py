"""
models.py

Brief description:
------------------
Models for the tournament system, including Teams, Submissions, and Matches.

Classes:
--------
- Team: Represents a participating team.
- Submission: Stores submission metadata for a team.
- Match: Records matches between two teams and their outcomes.

Author: Ambrose Ling
Date: 2025-05-25
"""


from sqlalchemy import Column, String, DateTime, Integer, ForeignKey
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
from __init__ import db
import datetime

class Submission(db.Model):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    status = Column(String, nullable=True, default="pending")
    blob_url = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "team_id": self.team_id,
            "status": self.status,
            "blob_url": self.blob_url,
            "created_at": self.created_at
        }

class Team(db.Model):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.now)
    submissions = db.relationship("Submission", backref="team", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at
        }


class Match(db.Model):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    team_id_1 = Column(Integer, ForeignKey("teams.id"), nullable=False)
    team_id_2 = Column(Integer, ForeignKey("teams.id"), nullable=False)
    container_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    result = Column(String, nullable=False)
    frames_blob_url = Column(String, nullable=False)
    round_num = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "team_id_1": self.team_id_1,
            "team_id_2": self.team_id_2,
            "status": self.status,
            "result": self.result,
            "frames_blob_url": self.frames_blob_url,
            "round_num": self.round_num,
            "created_at": self.created_at
        }
