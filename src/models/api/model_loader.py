import pickle

class BanModel:
    def __init__(self,model_path: str = 'CommentGuard_ML'):
        with open(model_path , 'rb') as f:
            self.model = pickle.load(model_path)