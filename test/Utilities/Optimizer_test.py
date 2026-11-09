import pytest 
from rbg.Utilities import AdamOptimizer


class TestOptimizer:
    
    @pytest.fixture
    def optimizer(self):
        learning_rate = 0.1 
        tolerance = 1e-9
        max_iterations = 1000
        return AdamOptimizer(learning_rate, tolerance, max_iterations)

    def test_with_function(self, optimizer):
        f = lambda x: (x + 3)**2 
        
        assert optimizer.getmin(0, f)[1][0] == pytest.approx(-3, abs=1e-3) 

    def test_with_gradient(self, optimizer):
        f = lambda x: (x + 3)**2 
        g = lambda x: 2 * x + 6

        assert optimizer.getmin(0, f, g)[1][0] == pytest.approx(-3, abs=1e-5)