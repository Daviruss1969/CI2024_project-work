import gp
import numpy as np

LEN_TREE_WEIGHT = 0.0001
NAN_PRED_WEIGHT = 0.001

VALUE_NAN_NUMBER = np.inf

class Individual:
    _tree: gp.Node
    _fitness: tuple[bool, float]
    _predictions: np.ndarray[float]
    _len_tree: int
    _valid: bool

    def __init__(self, tree: gp.Node | None, fitness: tuple[bool, float] = (False, -np.inf)):
        self._fitness = fitness
        self._len_tree = 0
        if tree:
            self._tree = tree
        else:
            self._tree = None

    def compute_fitness_mse(self, ground_truth: np.ndarray[float], min_mse: float | None = None) -> float:
        # Mask of boolean to get the valid prediction
        valid_mask = ~np.isnan(self._predictions)
        
        # Check if all inputs are valids
        if valid_mask.all():
            self._valid = True
        else:
            self._valid = False
            # Huge fitness if no values can be computed
            if not valid_mask.any():
                self._fitness = (False, -VALUE_NAN_NUMBER)
                return VALUE_NAN_NUMBER

        
        # Compute MSE with only valid predictions
        valid_ground_truth = ground_truth[valid_mask]
        valid_predictions = self._predictions[valid_mask]
        mse: float = 100 * np.square(valid_ground_truth - valid_predictions).sum() / len(valid_ground_truth)

        # Compute penalty for NaN
        if min_mse:
            nan_penalty = (~valid_mask).sum() * (NAN_PRED_WEIGHT * min_mse)
        else:
            nan_penalty = 0

        # Compute penalty for length of the tree
        if min_mse:
            len_tree_penalty = self._len_tree * (LEN_TREE_WEIGHT * min_mse)
        else:
            len_tree_penalty = 0
        
        # Compute final fitness
        self._fitness = (self._valid, -(mse + len_tree_penalty + nan_penalty))
        
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
    def fitness(self) -> tuple[bool, float]:
        return self._fitness
    
    @property
    def len_tree(self) -> int:
        return self._len_tree
    
    @property
    def valid(self) -> bool:
        return self._valid
    
    @len_tree.setter
    def len_tree(self, length: int):
        self._len_tree = length