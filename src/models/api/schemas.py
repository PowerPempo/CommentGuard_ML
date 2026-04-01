from pydantic import BaseModel, Field
from typing import Dict

class LabelResult(BaseModel):
    confidence: float
    flagged: bool

class CommentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class PredictionResponse(BaseModel):
    is_banned: bool
    confidence: float
    labels: Dict[str, LabelResult]