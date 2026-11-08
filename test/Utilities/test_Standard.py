import pytest
import numpy as np
from rbg.Preprocessing.Standard import Standard

def test_standard_scaler_workflow():
    np.random.seed(42)
    X_train = np.random.randn(100, 4) * 5 + 10
    X_test = np.random.randn(25, 4) * 5 + 10

    standard_scaler = Standard()

    # 1. Testar se o fit ajusta a média e std sem erro
    standard_scaler.fit(X_train)
    assert standard_scaler._mean is not None
    assert standard_scaler._std is not None

    # 2. Testar transform no treino (média deve virar ~0 e std ~1)
    X_train_scaled = standard_scaler.transform(X_train)
    np.testing.assert_allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-7)

    # 3. Testar transform no teste (deve manter o formato das colunas)
    X_test_scaled = standard_scaler.transform(X_test)
    assert X_test_scaled.shape == (25, 4)

    # 4. Testar reconstrução dos dados originais
    X_reconstructed = standard_scaler.untransform(X_train_scaled)
    np.testing.assert_allclose(X_reconstructed, X_train, atol=1e-7)