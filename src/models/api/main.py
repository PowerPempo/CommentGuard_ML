from fastapi import FastAPI, HTTPException
from src.models.api.model_loader import ban_model
from src.models.api.schemas import CommentRequest, PredictionResponse

app = FastAPI(
    Title = 'Banword model',
    description = 'Predicts the ban word comments',
    version='1.0.0'
)



@app.post('/predict', response_model = PredictionResponse)
async def predict(self, request: CommentRequest ):
    if not request.text.strip():
        raise HTTPException(status_code=400 , detail='Could not be empty')


    result = ban_model.predict(request.text)


    return PredictionResponse(
        text = request.text,
        is_banned = result['is_banned'],
        confidence = result['confidence']
    )