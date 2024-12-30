import gp
import numpy as np

LEN_TREE_WEIGHT = 0
NAN_PRED_WEIGHT = 0.1

VALUE_NAN_NUMBER = np.inf

class Individual:
    _tree: gp.Node
    _fitness: float
    _predictions: np.ndarray[float]
    _len_tree: int

    def __init__(self, tree: gp.Node | None, fitness: float = -np.inf):
        self._fitness = fitness
        self._len_tree = 0
        if tree:
            self._tree = tree
        else:
            self._tree = None

    def compute_fitness_mse(self, ground_truth: np.ndarray[float]) -> float:
        # Mask of boolean to get the valid prediction
        valid_mask = ~np.isnan(self._predictions)

        # Huge fitness if no values can be computed
        if not valid_mask.any():
            self._fitness = -VALUE_NAN_NUMBER
            return VALUE_NAN_NUMBER
        
        # Compute MSE with only valid predictions
        valid_ground_truth = ground_truth[valid_mask]
        valid_predictions = self._predictions[valid_mask]
        mse: float = 100 * np.square(valid_ground_truth - valid_predictions).sum() / len(valid_ground_truth)

        # Compute penalty for NaN
        nan_penalty = (~valid_mask).sum() * NAN_PRED_WEIGHT

        # Compute penalty for length of the tree
        len_tree_penalty = self._len_tree * LEN_TREE_WEIGHT
        
        # Compute final fitness
        self._fitness = -(mse + len_tree_penalty + nan_penalty)
        
        return mse

    def add_predictions(self, predictions: np.ndarray[float]):
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
    
    @property
    def len_tree(self) -> int:
        return self._len_tree
    
    @len_tree.setter
    def len_tree(self, length: int):
        self._len_tree = length