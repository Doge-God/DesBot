from typing import Literal, Union
from pydantic import BaseModel, TypeAdapter

class SttWordMessage(BaseModel):
    type: Literal["Word"]
    text: str
    start_time: float

class SttEndWordMessage(BaseModel):
    type: Literal["EndWord"]
    stop_time: float

class SttReadyMessage(BaseModel):
    type: Literal["Ready"]

class SttErrorMessage(BaseModel):
    type: Literal["Error"]
    message: str

class SttStepMessage(BaseModel):
    type: Literal["Step"]
    step_idx: int
    prs: list[float]  # Pause prediction scores

class SttMarkerMessage(BaseModel):
    type: Literal["Marker"]
    id: int

SttMessage = Union[SttWordMessage, SttEndWordMessage, SttReadyMessage, SttErrorMessage, SttStepMessage, SttMarkerMessage]
SttMessageAdapter = TypeAdapter(SttMessage)