import json
import joblib
import numpy as np


class BanModel:
    def __init__(self,
                 model_path: str = 'src/models/project_models/CommentGuard_ML.pkl',
                 labels_path: str = 'src/models/project_models/labels.json',
                 thresholds_path: str = 'src/models/project_models/thresholds.json'):

        self.model = joblib.load(model_path)

        with open(labels_path) as f:
            self.labels = json.load(f)

        with open(thresholds_path) as f:
            self.thresholds = json.load(f)

    def predict(self, text: str) -> dict:

        proba_vector = self.model.predict_proba([text])[0]  # (n_labels,)

        results = {}
        is_banned = False

        for i, label in enumerate(self.labels):
            prob = float(proba_vector[i])
            thresh = self.thresholds[label]
            flagged = prob >= thresh

            if flagged:
                is_banned = True

            results[label] = {
                'confidence': round(prob, 4),
                'flagged': flagged
            }

        max_confidence = round(float(proba_vector.max()), 4)

        return {
            'is_banned': is_banned,
            'confidence': max_confidence,
            'labels': results
        }


ban_model = BanModel()