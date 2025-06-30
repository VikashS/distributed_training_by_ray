from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelFactory:
    """Factory for creating and evaluating models."""
    @staticmethod
    def create_model(model_type, params):
        """Create model instance based on type and parameters."""
        if model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42
            )
        elif model_type == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                random_state=42
            )
        elif model_type == "LogisticRegression":
            return LogisticRegression(
                C=params['C'],
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def evaluate(model, X_test, y_test):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)