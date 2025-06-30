import ray
import pandas as pd
from patrain.etl import ETL
from patrain.trainer import Trainer


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Run ETL
    etl = ETL()
    X_train, X_test, y_train, y_test = etl.load()

    # Train models and save them
    trainer = Trainer(X_train, y_train, X_test, y_test, model_dir="models")
    results_df = trainer.train_all()

    # Display results
    print("\nModel Training Results:")
    print(results_df.sort_values('accuracy', ascending=False))

    best_model = results_df.loc[results_df['accuracy'].idxmax()]
    print("\nBest Model:")
    print(f"Model Type: {best_model['model_type']}")
    print(f"Parameters: { {k: v for k, v in best_model.items() if k not in ['model_type', 'accuracy', 'model_path']} }")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Model saved at: {best_model['model_path']}")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()