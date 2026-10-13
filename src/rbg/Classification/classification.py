from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):
    """Abstract base class for classification models."""

    def __init__(self):
        self._fit_done = False
        self._classes = None

    @abstractmethod
    def fit(self, X, y):
        """Fit the classifier and store its known classes."""
        pass

    @abstractmethod
    def predict(self, X):
        """Return one predicted class per observation."""
        pass
