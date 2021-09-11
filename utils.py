from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class VoteType(Enum):
    # html class for each vote type on theyvoteforyou.org
    VeryStronglyFor = 'position-voted-very-strongly-for'
    StronglyFor = 'position-voted-strongly-for'
    ModeratelyFor = 'position-voted-moderately-for'
    Mixed = 'position-voted-a-mixture-of-for-and-against'
    ModeratelyAgainst = 'position-voted-moderately-against'
    StronglyAgainst = 'position-voted-strongly-against'
    VeryStronglyAgainst = 'position-voted-very-strongly-against'


def make_set():
    return set()


@dataclass
class Party:
    name: str
    members: set = field(default_factory=make_set, init=False)
    start_date: datetime = None
    end_date: datetime = None

    def add(self, member) -> None:
        member.party = self
        self.members.add(member)


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

