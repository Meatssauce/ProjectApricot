from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class VoteType(Enum):
    VeryStronglyFor = 3
    StronglyFor = 2
    ModeratelyFor = 1
    Mixed = 0
    ModeratelyAgainst = -1
    StronglyAgainst = -2
    VeryStronglyAgainst = -3


@dataclass
class Party:
    name: str
    simple_name: str
    members: set
    start_date: datetime = None
    end_date: datetime = None


@dataclass
class Politician:
    name: str
    # affiliation_history: list
    electorate: str
    votes: list
    friends: set
    rebellion_rate: float = None
    attendance_rate: float = None
    party: Party = None

