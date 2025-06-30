import ray
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the training function for a single model
@ray.remote
def train_model(params):
    # Initialize model with given parameters
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'accuracy': accuracy
    }



def main():
    # Define hyperparameter grid
    param_grid = [
        {'n_estimators': n, 'max_depth': d}
        for n in [50, 100, 200]
        for d in [5, 10, 15, None]
    ]

    # Create Ray tasks for each hyperparameter combination
    futures = [train_model.remote(params) for params in param_grid]

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
    print(f"Parameters: n_estimators={best_model['n_estimators']}, "
          f"max_depth={best_model['max_depth']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")

    # Shutdown Ray
    ray.shutdown()





if __name__ == "__main__":
    main()