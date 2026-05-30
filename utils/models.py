from pydantic import BaseModel
from typing import List


class WatchPlan(BaseModel):
    theme: str
    selections: List[str]
    narrative: str


class Pitch(BaseModel):
    title: str
    logline: str
    concept: str
    data_report: str