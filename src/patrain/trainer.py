import ray
import pandas as pd
import os
import joblib

from patrain.models import ModelFactory


class Trainer:
    """Manages model training, evaluation, and saving using Ray."""

    def __init__(self, X_train, y_train, X_test, y_test, model_dir="models"):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    @staticmethod
    @ray.remote
    def train_model(model_type, params, X_train, y_train, X_test, y_test, model_dir):
        """Train, evaluate, and save a single model."""
        model = ModelFactory.create_model(model_type, params)
        model.fit(X_train, y_train)
        accuracy = ModelFactory.evaluate(model, X_test, y_test)

        # Generate model filename based on type and parameters
        param_str = "_".join(f"{k}={v}" for k, v in params.items())
        filename = f"{model_type}_{param_str}.joblib"
        filepath = os.path.join(model_dir, filename)

        # Save the model
        joblib.dump(model, filepath)

        return {
            'model_type': model_type,
            **params,
            'accuracy': accuracy,
            'model_path': filepath
        }

    def get_param_grid(self):
        """Define simplified hyperparameter grid."""
        return [
            {'model_type': 'RandomForest', 'params': {'n_estimators': n, 'max_depth': d}}
            for n in [50, 100]
            for d in [5, 10]
        ] + [
            {'model_type': 'GradientBoosting', 'params': {'n_estimators': 50, 'learning_rate': lr}}
            for lr in [0.1]
        ] + [
            {'model_type': 'LogisticRegression', 'params': {'C': c}}
            for c in [1.0]
        ]

    def train_all(self):
        """Train all models in parallel and save them."""
        param_grid = self.get_param_grid()
        futures = [
            self.train_model.remote(
                config['model_type'],
                config['params'],
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.model_dir
            )
            for config in param_grid
        ]
        results = ray.get(futures)
        return pd.DataFrame(results)