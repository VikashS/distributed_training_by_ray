from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ETL:
    """Handles data extraction, transformation, and loading."""
    def __init__(self, n_samples=5000, n_features=10, n_classes=2, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        self.scaler = StandardScaler()

    def extract(self):
        """Generate synthetic classification dataset."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            random_state=self.random_state
        )
        return X, y

    def transform(self, X, y):
        """Split and scale data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def load(self):
        """Execute ETL pipeline."""
        X, y = self.extract()
        return self.transform(X, y)