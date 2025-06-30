import ray
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the training function for a single model
@ray.remote
def train_model(model_type, params):
    # Initialize model based on type
    if model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42
        )
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=42
        )
    elif model_type == "LogisticRegression":
        model = LogisticRegression(
            C=params['C'],
            max_iter=1000,
            random_state=42
        )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'model_type': model_type,
        **params,
        'accuracy': accuracy
    }


def main():
    # Define hyperparameter grid for different model types
    param_grid = [
                     # RandomForest parameters
                     {'model_type': 'RandomForest', 'params': {'n_estimators': n, 'max_depth': d}}
                     for n in [50, 100, 200]
                     for d in [5, 10, 15, None]
                 ] + [
                     # GradientBoosting parameters
                     {'model_type': 'GradientBoosting',
                      'params': {'n_estimators': n, 'learning_rate': lr, 'max_depth': d}}
                     for n in [50, 100]
                     for lr in [0.01, 0.1]
                     for d in [3, 5]
                 ] + [
                     # LogisticRegression parameters
                     {'model_type': 'LogisticRegression', 'params': {'C': c}}
                     for c in [0.1, 1.0, 10.0]
                 ]

    # Create Ray tasks for each model and hyperparameter combination
    futures = [train_model.remote(config['model_type'], config['params']) for config in param_grid]

    # Collect results
    results = ray.get(futures)

    # Convert results to DataFrame for easy viewing
    results_df = pd.DataFrame(results)

    # Find best model
    best_model = results_df.loc[results_df['accuracy'].idxmax()]

    # Print results
    print("\nAll Results:")
    print(results_df.sort_values('accuracy', ascending=False))
    print("\nBest Model:")
    print(f"Model Type: {best_model['model_type']}")
    print(f"Parameters: { {k: v for k, v in best_model.items() if k not in ['model_type', 'accuracy']} }")
    print(f"Accuracy: {best_model['accuracy']:.4f}")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()