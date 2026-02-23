from src.models.train import DATA_PATH , RANDOM_STATE , load_data , base_pipeline
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import classification_report


X, y, df, active_labels = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

pipe = base_pipeline()

param_grid = {
    'vect__max_features': [30000, 50000],
    'vect__ngram_range': [(1, 2), (1, 3)],
    'clf__estimator__C': [0.5, 1.0, 5.0, 10.0],
}

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\n=== results ===")
print("best params :", grid.best_params_)
print("F1 best:", grid.best_score_)

y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred, target_names=active_labels, zero_division=0))