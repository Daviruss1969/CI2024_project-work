import numpy as np
import gp
import random
from copy import deepcopy

TOURNAMENT_SELECTION_SIZE = 3

MUTATION_PROBABILITY = 0.5
LEAF_MUTATION_PROBABILITY = 0.5

CONSTANT_PROBABILITY = 0.3
CONSTANT_RANGE = 10

class Symreg_gp:
    _operators: list[np.ufunc]
    _variables: list[str]
    _train: bool
    _population: list[gp.Individual]
    _ground_truth: np.ndarray[float]
    _population_size: int
    _offspring_size: int
    _max_depth: int


    def __init__(self, input_size: int, ground_truth: np.ndarray[float], population_size: int, offspring_size: int, max_depth: int):
        self._operators = Symreg_gp.get_valid_ufuncs()
        self._variables = [Symreg_gp.formated_variable(i) for i in range(input_size)]
        self._train = False
        self._population_size = population_size
        self._offspring_size = offspring_size
        self._ground_truth = ground_truth
        self._max_depth = max_depth
        self._population = []

        # Init population
        for _ in range(self._population_size):
            # Generate random tree
            tree = self._generate_random_tree(self._max_depth)
            individual = gp.Individual(tree)
            self._population.append(individual)

    def __call__(self, inputs: np.ndarray[float]) -> np.ndarray[float]:
        if self._train:
            return self._train_call(inputs)
        else:
            return self._inference_call(inputs)
        
    def update_mse(self, draw_fittest: bool = False) -> list[float]:
        mean_square_errors: list[float] = list()
        offspring: list[gp.Individual] = list()

        self._population.sort(key=lambda l: l.fitness, reverse=True)
        self._population = self._population[:self._population_size]

        if (draw_fittest):
            self._population[0].tree.draw()

        for individual in self._population:
            mean_square_errors.append(individual.compute_fitness_mse(self._ground_truth))

        for _ in range(self._offspring_size):
            parent1 = self._tournament_selection(TOURNAMENT_SELECTION_SIZE)
            parent2 = self._tournament_selection(TOURNAMENT_SELECTION_SIZE)
            new_individual = self._subtree_crossover(parent1, parent2)
            if random.random() < MUTATION_PROBABILITY:
                if random.random() < LEAF_MUTATION_PROBABILITY:
                    new_individual = self._leaf_mutation(new_individual)
                else:
                    new_individual = self._operator_mutation(new_individual)

            offspring.append(new_individual)
        
        offspring.append(self._population[0]) # Append the fittest individual from last generation (elitist strategies)
        self._population = offspring

        return mean_square_errors

    def _train_call(self, inputs):
        for individual in self._population:
            predictions = [
                individual.tree(**{n: v for n, v in zip(self._variables, row)}) 
                for row in inputs
            ]

            individual.add_predictions(predictions)

        return

    def _inference_call(self, inputs) -> np.ndarray[float]:
        Y_pred: list[float] = list()

        for row in inputs:
            Y_pred.append(self._population[0].tree(**{n: v for n, v in zip(self._variables, row)}))

        return np.array(Y_pred)
    
    def _tournament_selection(self, k: int) -> gp.Individual:
        tournament = random.sample(self._population, k)
        return max(tournament, key=lambda l: l.fitness)
    
    def _random_mutation(self, individual: gp.Individual) -> gp.Individual:
        child = deepcopy(individual.tree)
        child.set_random_subtree(self._generate_random_tree(2))
        return gp.Individual(child)
    
    def _leaf_mutation(self, individual: gp.Individual) -> gp.Individual:
        child = deepcopy(individual.tree)
        leaf = self._create_leaf()
        child.set_random_subtree(leaf, True)
        return gp.Individual(child)
    
    def _operator_mutation(self, individual: gp.Individual) -> gp.Individual:
        child = deepcopy(individual.tree)
        child.set_random_operator(self._operators)
        return gp.Individual(child)
    
    def _subtree_crossover(self, parent1: gp.Individual, parent2: gp.Individual) -> gp.Individual:
        node = deepcopy(parent1.tree)

        new_node = deepcopy(parent2.tree.get_random_subtree())

        node.set_random_subtree(new_node)
        
        return gp.Individual(node)

    def _generate_random_tree(self, max_depth: int, current_depth: int = 0) -> gp.Node:
        """Return a random tree generated according to max_depth"""
        # Forced to return a leaf
        if current_depth == max_depth:
            return self._create_leaf()
        
        # Return a leaf according to a probability that depends on how deep we are
        prob_leaf = current_depth / max_depth
        if random.random() < prob_leaf:
            return self._create_leaf()
        
        # Create a numpy node
        operator = random.choice(self._operators)
        successors = [
            self._generate_random_tree(max_depth, current_depth + 1)
            for _ in range(operator.nin)
        ]
        return gp.Node(operator, successors)
            

    def _create_leaf(self) -> gp.Node:
        """Create a leaf, either a const or a variable"""
        if random.random() < CONSTANT_PROBABILITY:
            return gp.Node(random.uniform(-CONSTANT_RANGE, CONSTANT_RANGE))
        else:
            return gp.Node(random.choice(self._variables))
    
    def train(self, train: bool):
        self._train = train

    @staticmethod
    def formated_variable(i: int) -> str:
        """Return a string according to the input : i => xi"""
        return f"x{i}"

    @staticmethod
    def get_valid_ufuncs() -> list[np.ufunc]:
        """
        Return a list of numpy ufuncs who's:
        - Return value is a float
        - Can handle float values in entries
        """
        funcs: list[np.ufunc] = []
        for _, obj in np.__dict__.items():
            # Check if the entry is a numpy function
            if (isinstance(obj, np.ufunc)):
                # Get the tuple that represent the number of arguments + the return value
                dtypes = ()
                if (obj.nin == 1):
                    dtypes = (np.dtype("float32"), None)
                elif (obj.nin == 2):
                    dtypes = (np.dtype("float32"), np.dtype("float32"), None)
                else:
                    continue

                # Try/catch => to discard functions that cannot handle floats as input
                try:
                    # Get the result type of the function and add it if it's not already there (because of numpy aliases, there can be multiple times the same function)
                    # I also have to discard some functions since they work only on vectors and not scalars
                    result_type = obj.resolve_dtypes(dtypes)[-1]
                    if result_type == np.float32 and obj not in funcs and obj != np.matmul and obj != np.vecdot and obj != np.vecmat and obj != np.matvec:
                        funcs.append(obj)
                except:
                    continue

        return funcs


    