import joblib



class BanModel:
    def __init__(self, model_path: str = 'CommentGuard_ML.pkl'):
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> dict:

        proba = self.model.predict_proba([text])[0][1]


        return {
            'is_banned': bool(proba > 0.5),
            'confidence': round(float(proba), 4)
        }


ban_model = BanModel()
