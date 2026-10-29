import pytest 
from rbg.Utilities.Optimizer import AdamOptimizer

class TestOptimizer():
    def test_with_function(self):
        f = lambda x : (x+3)**2 # minimum at x = -3

        learning_rate = 1e-3
        tolerance = 1e-5
        max_iterations = 1000
        
        optimizer = AdamOptimizer(learning_rate, tolerance, max_iterations) #learning rate, tolerance, max iterations

        assert abs(optimizer.getmin(0, f) + 3) < 1e-5 

    def test_with_gradient(self):
        #same as previous test, bu now we also pass on the gradient directly
        f = lambda x : (x+3)**2 # minimum at x = -3
        g = lambda x : 2*x + 6

        learning_rate = 1e-3
        tolerance = 1e-5
        max_iterations = 1000
        optimizer = AdamOptimizer(learning_rate, tolerance, max_iterations) #learning rate, tolerance, max iterations

        assert abs(optimizer.getmin(0, f, g) + 3) < 1e-5 

    def test_(self):
        pass