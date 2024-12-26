import numpy as np
from typing import Callable
import numbers
import random
import gp


valueType = str | int | np.ufunc | None

class Node:
    __func: Callable
    __successors: list['Node']
    __arity: int
    __str: str

    def __init__(self, value: valueType, successors: list['Node'] = []):
        if isinstance(value, np.ufunc):
            # NumPy functions
            def _f(*_args, **_kwargs):
                return value(*_args)
            
            # Check arity
            self.__arity = value.nin # since we work with NumPy ufuncs we can access to the nin field to get the number of parameters
            assert len(successors) == self.__arity, (
                "Incorrect number of children."
                + f" Expected {self.__arity} found {len(successors)}"
            )

            # Check types of successors
            assert all(isinstance(s, Node) for s in successors), "Panic: Successors must be `Node`"

            self.__func = _f
            self.__successors = list(successors)
            self.__str = value.__name__
        elif isinstance(value, numbers.Number):
            # Constants
            self.__func = eval(f'lambda **_kw: {value}')
            self.__successors = list()
            self.__arity = 0
            self.__str = f'{value:g}'
        elif isinstance(value, str):
            # Variables
            self.__func = eval(f'lambda *, {value}, **_kw: {value}')
            self.__successors = list()
            self.__arity = 0
            self.__str = str(value)
        else:
            assert False

    def __call__(self, **kwargs):
        return self.__func(*[c(**kwargs) for c in self.__successors], **kwargs)
    
    def __str__(self):
        return self.__str
    
    def __len__(self):
        if (len(self.__successors) == 0):
            return 0
        
        if (len(self.__successors) == 1):
            return 1 + len(self.successors[0])
        
        return 1 + max(len(self.successors[0]), len(self.successors[1]))

    def get_random_subtree(self) -> 'Node':
        if len(self.__successors) == 0 or random.random() < .3:
            return self
        
        next_index = random.choice(range(len(self.__successors)))

        return self.__successors[next_index].get_random_subtree()
    
    def set_random_subtree(self, node: 'Node', leaf: bool = False) -> None:
        next_index = random.choice(range(len(self.__successors)))

        if (not leaf and random.random() < .3) or len(self.__successors[next_index].__successors) == 0:
            self.__successors[next_index] = node
            return
        
        self.__successors[next_index].set_random_subtree(node)
        
    def set_random_operator(self, operators: list[np.ufunc]) -> None:
        next_index = random.choice(range(len(self.__successors)))
        if (random.random() < .3) or len(self.__successors[next_index].__successors) == 0:
            available_operators = [operator for operator in operators if operator.nin == self.arity]
            operator = random.choice(available_operators)
            def _f(*_args, **_kwargs):
                return operator(*_args)
            self.__func = _f
            return
        
        self.__successors[next_index].set_random_operator(operators)

        
    
    @property
    def arity(self):
        return self.__arity

    @property
    def successors(self):
        return self.__successors

    @property
    def is_leaf(self):
        return self.arity == 0

    def draw(self) -> None:
        gp.draw.draw_node(self)