import sympy as sy

BIN_OPERATORS = ['+', '-', '*', '/']
UNARY_OPERATORS = ['log', 'exp']

class BinaryTree():
    def __init__(self, rootid):
        self.left = None
        self.right = None
        self.rootid = rootid
        self.parent = self
        self.nodes = []
        self.parent_leaf = []
        self.vars = []

    def expandConstants(self):

        if self.rootid in UNARY_OPERATORS:
            self.right = self.clone()
            self.rootid = "*"
            self.left = "1"

            if isinstance(self.left, BinaryTree):
                self.left.expandConstants()

        else:

            if isinstance(self.left, BinaryTree):
                self.left.expandConstants()

            if isinstance(self.right, BinaryTree):
                self.right.expandConstants()

    def getNumConstants(self):

        n = 0
        if isinstance(self.left, BinaryTree):
            n += self.left.getNumConstants()
        elif not self.left in self.vars and not self.left is None:
            n += 1

        if isinstance(self.right, BinaryTree):
            n += self.right.getNumConstants()
        elif not self.right in self.vars and not self.right is None:
            n += 1

        return n

    def update_constants(self, constants):

        if isinstance(self.left, BinaryTree):
            constants = self.left.update_constants(constants)
        elif not self.left is None and not self.left in self.vars:
            self.left = constants[0]
            constants = constants[1:]

        if isinstance(self.right, BinaryTree):
            constants = self.right.update_constants(constants)
        elif not self.right is None and not self.right in self.vars:
            self.right = constants[0]
            constants = constants[1:]

        return constants

    def setLeftChild(self, value):
        self.left = value

        if isinstance(value, BinaryTree):
            value.parent = self
        elif not str(value).replace(".", "").isdigit() and not value is None:
            self.vars.append(value)

        self.nodes = self.getNodes()

    def getLeftChild(self):
        return self.left

    def setRightChild(self, value):
        self.right = value

        if isinstance(value, BinaryTree):
            value.parent = self
        elif not str(value).replace(".", "").isdigit() and not value is None:
            self.vars.append(value)

        self.nodes = self.getNodes()

    def getRightChild(self):
        return self.right

    def setNodeValue(self, value):
        self.rootid = value

    def getNodeValue(self):
        return self.rootid

    def getVars(self):
        vars = []
        if isinstance(self.left, BinaryTree):
            vars += self.left.getVars()

        if isinstance(self.right, BinaryTree):
            vars += self.right.getVars()

        vars += self.vars

        return list(set(vars))

    def getNodes(self):
        nodes = []
        if isinstance(self.left, BinaryTree):
            nodes += [self.left] + self.left.getNodes()

        if isinstance(self.right, BinaryTree):
            nodes += [self.right] + self.right.getNodes()

        nodes += [self]

        return nodes

    def str(self):

        # binary operator
        if not self.right is None:

            return "(%s%s%s)" % (self.left.str() if isinstance(self.left, BinaryTree) else self.left,
                                 self.rootid,
                                 self.right.str() if isinstance(self.right, BinaryTree) else self.right)

        else:
            return "(%s(%s))" % (self.rootid,
                                 self.left.str() if isinstance(self.left, BinaryTree) else self.left)

    def __str__(self):
        formula = self.str()
        return formula

        expr = sy.sympify(formula)

        return str(sy.simplify(expr))

    def clone(self):
        t = BinaryTree(self.rootid)
        t.setLeftChild(self.left.clone() if isinstance(self.left, BinaryTree) else self.left)
        t.setRightChild(self.right.clone() if isinstance(self.right, BinaryTree) else self.right)

        return t


