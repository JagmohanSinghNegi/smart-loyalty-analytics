class DummyModel:
    """Placeholder model class for prototyping."""
    def predict(self, X):
        # return zeros of appropriate length
        try:
            return [0 for _ in X]
        except TypeError:
            return 0
