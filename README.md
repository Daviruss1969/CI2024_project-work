# Set cover problem

This repository contains my work for the project work of the computational intelligence course at PoliTo.

It contains a genetic programming algorithm to solve [symbolic regression](https://en.wikipedia.org/wiki/Symbolic_regression).

## Structure

The file [s334726.py](./s334726.py) contains my solutions for the [data](./data/) folder given by the teachers.

Under the [src](./src/) directory there is first the gp implementation in the [gp](./src/gp/) folder. And also a [test.ipynb](./src/test.ipynb) file to run the genetic programming algorithm.

### GP structure

- [node.py](./src/gp/node.py) handle the nodes of the gp.
- [draw.py](./src/gp/draw.py) handle the function needed to draw the gp.
- [individual](./src/gp/individual.py) handle the fitness of the inviduals
- [gp_symreg.py](./src/gp/gp_symreg.py) handle the logic of the gp.


## Run the code

### Dependencies

```
python 3.10.12
numpy 2.2.1
ipykernel = "^6.29.5"
graphviz = "^0.20.3"
matplotlib = "^3.10.0"
tqdm = "^4.67.1"
```

The version of numpy is important to get the numpy functions. Indeed, I discarded some functions so having a older version may result in function that don't exist and newer version may result in new invalid functions.

## Results

Here's the results for the given dataset

```
1: 3.92119e-32
2: 1.43075e+15
3: 20232.1
4: 38.2022
5: 5.57281e-16
6: 7.90409
7: 35085.2
8: 9.085e+07
```