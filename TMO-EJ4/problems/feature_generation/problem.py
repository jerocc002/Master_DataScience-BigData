import random
import sympy as sy
import sys
from sklearn.model_selection import train_test_split
from optimization.genetic import BaseGA
from optimization.programming import UNARY_OPERATORS, BIN_OPERATORS, BinaryTree
import pandas as pd
import numpy as np

class EvolEquation(BaseGA):
    NO_LEAF_MUTATIONS = ["OPERATOR", "SWAP", "GROW"]
    LEAF_MUTATIONS = ["VALUE", "VAR", "CUT"]
    CACHE = []

    def __init__(self, X, y, metric, columns=None, num_equations=100, max_depth=3,
                 mutation_prob=0.1, crossover_prob=0.1, restart_iterations=10000):

        super(EvolEquation, self).__init__(mutation_prob, crossover_prob, restart_iterations)
        self.X = X
        self.y = y
        self.num_equations = num_equations
        self.max_depth = max_depth
        self.columns = columns if columns else self.X.columns
        self.metric = metric

        self.columns = ["X['%s']" % c for c in self.columns]

    def generate_population(self, n=None):
        if n is None: n = self.num_equations
        columns = self.columns

        population = []
        while len(population) < n:
            eq = self._generate_equation(columns, self.max_depth)
            population += [eq]

        return population

    def _generate_equation(self, vars, max_depth=2):
        error = sys.float_info.max

        while error >= 5000:
            tree, t = self.__generate_equation(None, vars, max_depth)
            error = self.__evaluate(tree, self.X, self.y)

        return tree

    def _create_feature(self, dna, X, boolean=True):

        formula = str(dna)
        formula = formula.replace("log", "np.log")
        formula = formula.replace("exp", "np.exp")
        formula = formula.replace("sin", "np.sin")
        try:
            feature = eval(formula)
        except:
            return pd.Series([0] * X.shape[0])

        if type(feature) is pd.Series:
            feature = feature.astype(np.float32)

            if type(feature) == np.float32 or type(feature) == np.float:
                feature = pd.Series([feature] * X.shape[0])

            feature = feature.replace([np.inf], np.finfo(np.float32).max)
            feature = feature.replace([-np.inf], np.finfo(np.float32).min)
            feature[feature.isnull()] = np.finfo(np.float32).max
            if boolean:
                return (feature.astype(np.float32) > 1).fillna('-99999')
            else:
                return feature.astype(np.float32).fillna('-99999')
        else:
            return pd.Series([feature > 1] * X.shape[0])


    def __evaluate(self, dna, X, y):
        return self.metric(y, self._create_feature(dna, X))

    def __generate_equation(self, op, vars, max_depth=2, current_depth=0):

        # Max depth -> leaf (constant or var)
        if current_depth >= max_depth:

            if True or op in UNARY_OPERATORS or random.choice([True, False]):
                return random.choice(vars), 'V'
            else:
                return random.uniform(0, 100), 'C'

        else:

            op = random.choice(BIN_OPERATORS + UNARY_OPERATORS)

            left, tl = self.__generate_equation(op, vars, max_depth,
                                    current_depth + random.choice(range(1, max_depth - current_depth + 1)))
            right = None
            valid = True
            if op in BIN_OPERATORS:
                right, tr = self.__generate_equation(op, vars, max_depth, current_depth + 1)
                valid = False

            while not valid:
                valid = True
                if not isinstance(left, BinaryTree) and not isinstance(right, BinaryTree):

                    # two numbers 3*4, 2-3, .....
                    # same var: x*x, x/x, x**x
                    if tl == tr and (tl == 'C' or left == right):
                        valid = False
                        right, tr = self.__generate_equation(op, vars, max_depth, current_depth + 1)

                    if tr == 1 and op in ["*", "/", "**"]:
                        valid = False
                        right, tr = self.__generate_equation(op, vars, max_depth, current_depth + 1)

            t = BinaryTree(op)
            t.setLeftChild(left)
            t.setRightChild(right)

            return t, None

    def mutate(self, dna):
        ini = dna.clone()

        idx1 = random.choice(range(len(dna.nodes)))
        n1 = dna.nodes[idx1]

        if not isinstance(n1.left, BinaryTree) or not isinstance(n1.right, BinaryTree):
            if random.uniform(0, 1) < 0.7:
                mutation = random.choice(self.LEAF_MUTATIONS + ["OPERATOR"])
            else:
                mutation = random.choice(self.NO_LEAF_MUTATIONS)
        else:
            mutation = random.choice(self.NO_LEAF_MUTATIONS)

        if mutation == "OPERATOR":
            if n1.right is None:
                n1.rootid = random.choice(UNARY_OPERATORS)
            else:
                n1.rootid = random.choice(BIN_OPERATORS)

        elif mutation == "SWAP":
            if not n1.right is None:
                l = n1.left
                r = n1.right
                n1.left = r
                n1.right = l

        elif mutation == "VALUE":
            if not isinstance(n1.left, BinaryTree) and n1.right in self.columns:
                n1.left = random.uniform(0, 100)
            elif not n1.right is None:
                n1.right = random.uniform(0, 100)

        elif mutation == "VAR":

            if not isinstance(n1.left, BinaryTree):
                vars = [v for v in self.columns if n1.left != v]
                if len(vars) > 0:
                    n1.left = random.choice(vars)
            elif not n1.right is None:
                vars = [v for v in self.columns if n1.right != v]
                if len(vars) > 0:
                    n1.right = random.choice(vars)

        elif mutation == "CUT":
            if n1.right is None:
                cand = n1.left
            else:
                cand = random.choice([n1.left, n1.right])

            if n1.parent.right is None:
                n1.parent.left = cand
            else:
                if random.choice([True, False]):
                    n1.parent.left = cand
                else:
                    n1.parent.right = cand

        elif mutation == "GROW":
            if dna.rootid in UNARY_OPERATORS:
                op = random.choice(BIN_OPERATORS)
            else:
                op = random.choice(BIN_OPERATORS + UNARY_OPERATORS)

            l = dna
            r = None
            if op in BIN_OPERATORS:
                r = self._generate_equation(self.columns, self.max_depth)

            t = BinaryTree(op)
            t.setLeftChild(l)
            t.setRightChild(r)
            dna = t

        return dna.clone()

    def crossover(self, dna1, dna2):
        return dna1, dna2
        idx1 = random.choice(range(len(dna1.nodes)))
        idx2 = random.choice(range(len(dna2.nodes)))
        n2 = dna2.nodes[idx2].clone()
        if not dna1.nodes[idx1].parent.left is None:
            dna1.nodes[idx1].parent.setLeftChild(n2)
        else:
            dna1.nodes[idx1].parent.setRightChild(n2)

        idx1 = random.choice(range(len(dna1.nodes)))
        idx2 = random.choice(range(len(dna2.nodes)))
        n1 = dna1.nodes[idx1].clone()

        if not dna2.nodes[idx2].parent.right is None:
            dna2.nodes[idx2].parent.setRightChild(n1)
        else:
            dna2.nodes[idx2].parent.setLeftChild(n1)

        return dna1.clone(), dna2.clone()

    def select(self, items, weights):

        """
        Chooses a random element from items, where items is a list of tuples in
        the form (item, weight). weight determines the probability of choosing its
        respective item.
        """
        weights = [1 / w for w in weights]
        idx = sorted(range(len(items)), key=lambda i: weights[i])
        items = [items[i] for i in idx]
        weights = sorted(weights)
        weight_total = sum(weights)
        n = random.uniform(0, weight_total)
        i = 0
        item = items[0]
        for i, item, weight in zip(idx, items, weights):
            if n < weight:
                return item, i

            n = n - weight
        return item, i

    def _evaluate_list(formula, params):
        # print formula
        expr = sy.sympify(formula)

        result = []
        cache = {}
        for p in params:
            key = '|'.join([str(v) for v in p.values()])
            if key in cache:
                r = cache[key]
            else:
                r = expr.evalf(subs=p, chop=True)
                cache[key] = r

            result.append(r)

        # return [expr.evalf(subs=kwargs) for kwargs in params]
        return result

    def _filtered_params(self, dna):
        vars = dna.getVars()
        params = []
        for p in self.params:
            np = {}
            for v in vars:
                if v in p:
                    np[v] = p[v]

            params.append(np)

        return params

    def fitness(self, dna):

        if isinstance(dna, list):
            return [self.__evaluate(ind, self.X, self.y) for ind in dna]
        else:
            return self.__evaluate(dna, self.X, self.y)

    def pre(self, population_fitness):
        pass

    def post_generation(self, generation):
        if generation > 0 and generation % 10 == 0:
            result_population = []
            for eq in self.population:
                if not str(eq) in self.CACHE:
                    # optmizer = Optimizer(eq, self.y,  self._filtered_params(eq))
                    # eq = optmizer.optimize_equation()

                    self.CACHE.append(str(eq))

                result_population.append(eq)

            self.population = result_population



