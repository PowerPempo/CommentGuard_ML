from pydantic import BaseModel


class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text : str
    is_banned : bool
    confidence : float

