import gp
import numpy as np
import math

class Individual:
    _tree: gp.Node
    _fitness: float
    _predictions: list[float]

    def __init__(self, tree: gp.Node | None, fitness: float = -np.inf):
        self._fitness = fitness
        if tree:
            self._tree = tree
        else:
            self._tree = None

    def compute_fitness_mse(self, ground_truth: np.ndarray[float]) -> float:
        mse: float = 100*np.square(ground_truth-self._predictions).sum()/len(ground_truth)
        mse = (1e6 if math.isnan(mse) else mse)
        self._fitness = -(mse + (len(self.tree) * 5))
        return mse

    def add_predictions(self, predictions: list[float]):
        self._predictions = predictions
        return

    @property
    def predictions(self) -> list[float]:
        return self._predictions

    @property
    def tree(self) -> gp.Node:
        return self._tree
    
    @property
    def fitness(self) -> float:
        return self._fitness