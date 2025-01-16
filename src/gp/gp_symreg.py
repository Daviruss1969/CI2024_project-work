import numpy as np
import gp
import random
from copy import deepcopy
import warnings

TOURNAMENT_SELECTION_SIZE = 3

ONLY_MUTATION_PROBABILITY = .2
MUTATION_PROBABILITY = .7
LEAF_MUTATION_PROBABILITY = .4
OPERATOR_MUTATION_PROBABILITY = .4

CONSTANT_PROBABILITY = .2

MAX_TREE_SIZE = 10

class Symreg_gp:
    _operators: list[np.ufunc]
    _variables: list[str]
    _population: list[gp.Individual]
    _ground_truth: np.ndarray[float]
    _population_size: int
    _offspring_size: int
    _initial_max_depth: int
    _constant_range: float
    _fittest_individual: gp.Individual


    def __init__(self, input_size: int, ground_truth: np.ndarray[float], population_size: int, offspring_size: int, initial_max_depth: int):
        self._operators = Symreg_gp.get_valid_ufuncs()
        self._variables = [Symreg_gp.formated_variable(i) for i in range(input_size)]
        self._population_size = population_size
        self._offspring_size = offspring_size
        self._ground_truth = ground_truth
        self._initial_max_depth = initial_max_depth
        self._population = []
        self._constant_range = np.max(np.abs(ground_truth))
        self._fittest_individual = None

        # Init population
        for _ in range(self._population_size):
            # Generate random tree
            tree = self._generate_random_tree(self._initial_max_depth)
            individual = gp.Individual(tree)
            individual.len_tree = len(tree)
            self._population.append(individual)

    def __call__(self, inputs: np.ndarray[float]) -> np.ndarray[float]:
        # For each individual
        for individual in self._population:

            # Get predictions for each input row
            def safe_tree_evaluation(row):
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")  # Capture all warnings

                        # Evaluate the tree
                        result = individual.tree(**{n: v for n, v in zip(self._variables, row)})
                        
                        # Check if there is a warning
                        if w:
                            return np.nan

                        return result
                except Exception as e:
                    # Handle exceptions
                    return np.nan

            # Get predictions for each inputs
            predictions = np.apply_along_axis(
                safe_tree_evaluation,
                axis=1,
                arr=inputs
            )

            # Add the predictions to the individual
            individual.add_predictions(predictions)
        
    def update_mse(self, min_mse: float | None = None) -> list[float] | gp.Individual:
        mean_square_errors: list[float] = list()
        offspring: list[gp.Individual] = list()

        # Compute the fitness for each individuals on the previous call
        for individual in self._population:
            mean_square_errors.append(individual.compute_fitness_mse(self._ground_truth, min_mse))
            if self._fittest_individual == None or individual.fitness > self._fittest_individual.fitness:
                self._fittest_individual = individual

        # Survivor selection (deterministic)
        self._population.sort(key=lambda l: l.fitness, reverse=True)
        self._population = self._population[:self._population_size]

        # Generate offsprings
        for _ in range(self._offspring_size):
            mutation = False
            if random.random() < ONLY_MUTATION_PROBABILITY:
                # Only mutation
                new_individual = self._tournament_selection(TOURNAMENT_SELECTION_SIZE)
                mutation = True
            else:
                # Crossover with maybe mutation
                parent1 = self._tournament_selection(TOURNAMENT_SELECTION_SIZE)
                parent2 = self._tournament_selection(TOURNAMENT_SELECTION_SIZE)
                new_individual = self._subtree_crossover(parent1, parent2)
                if random.random() < MUTATION_PROBABILITY:
                    mutation = True

            if mutation:
                # Select an mutation operator
                if random.random() < LEAF_MUTATION_PROBABILITY:
                    new_individual = self._leaf_mutation(new_individual)
                elif random.random() < OPERATOR_MUTATION_PROBABILITY:
                    new_individual = self._operator_mutation(new_individual)
                else:
                    new_individual = self._random_mutation(new_individual)

            # Compute the length of the tree and add it if it's not to huge
            new_individual.len_tree = len(new_individual.tree)
            if (new_individual.len_tree < MAX_TREE_SIZE):
                offspring.append(new_individual)
        
        if self._fittest_individual:
            offspring.append(self._fittest_individual)# Append the fittest individual from last generation (elitist strategies)

        self._population = offspring

        return mean_square_errors
    
    def get_fittest_individual(self) -> gp.Individual:
        return self._fittest_individual
    
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
            return gp.Node(random.uniform(-self._constant_range, self._constant_range))
        else:
            return gp.Node(random.choice(self._variables))

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


    