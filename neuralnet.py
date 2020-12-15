######################## Created by Sriya Vudata (sv520) #########################
import random
### this file contains classes for the neural network
# represents a node in a single layer
class Node:
    # output of the node
    out = 0.0
    # weight vector
    w = []
    # derivative computed during backpropogation
    derivative = 0.0
    def __init__(self, num_weights):
        self.w = [random.random() for _ in range(num_weights+1)]

# represents a layer
class Layer:
    # number of nodes in the layer
    num_nodes = 0
    # number of inputs to the layer
    num_ins = 0
    # list of nodes in that layer
    nodes = []
    def __init__(self, n, inn):
        self.nodes = [None]*n
        self.num_nodes = n
        self.num_ins = inn
