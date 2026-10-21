from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):
    """Abstract base class for classification models."""

    def __init__(self, classes, reference_class=None):
        """Store class labels and move the reference class to the last position.

        Args:
            classes: Nonempty one-dimensional sequence of unique class labels.
            reference_class: Reference label. If None, use the first supplied
                label, then move it to the end of the stored class order.

        Raises:
            ValueError: If classes is empty, not one-dimensional, contains
                duplicates, or does not contain the reference label.
        """
        self._fit_done = False
        self._classes = np.asarray(classes)
        if self._classes.ndim != 1 or self._classes.size == 0:
            raise ValueError("classes must be a nonempty one-dimensional sequence.")
        if np.unique(self._classes).size != self._classes.size:
            raise ValueError("classes must contain unique labels.")
        if reference_class is None:
            reference_class = self._classes[0]
        if reference_class not in self._classes:
            raise ValueError("The reference class must be present in classes.")
        self._reference_class = reference_class
        self._classes = np.concatenate((self._classes[self._classes != reference_class],self._classes[self._classes == reference_class],))

    @abstractmethod
    def fit(self, X, y):
        """Fit the classifier using features X and original class labels y.

        Args:
            X: Feature matrix of shape (n_samples, p_features).
            y: Class-label vector of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Return one original class label per input observation.

        Args:
            X: Feature matrix of shape (n_new_samples, p_features).

        Returns:
            Predicted label vector of shape (n_new_samples,).
        """
        pass
