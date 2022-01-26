import numpy as np

from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activation import ReLU

class MLP0:

    def __init__(self, debug = False):
    
        self.layers = [ Linear(2, 3) ]
        self.f      = [ ReLU() ]

        self.debug = debug

    def forward(self, A0):

        Z0 = None # TODO
        A1 = None # TODO

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1
        
        return NotImplemented

    def backward(self, dLdA1):
    
        dA1dZ0 = None # TODO
        dLdZ0  = None # TODO
        dLdA0  = None # TODO

        if self.debug:

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0
        
        return NotImplemented
        
class MLP1:

    def __init__(self, debug = False):
    
        self.layers = [ Linear(2, 3),
                        Linear(3, 2) ]
        self.f      = [ ReLU(),
                        ReLU() ]

        self.debug = debug

    def forward(self, A0):
    
        Z0 = None # TODO
        A1 = None # TODO
    
        Z1 = None # TODO
        A2 = None # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2
        
        return NotImplemented

    def backward(self, dLdA2):

        dA2dZ1 = None # TODO
        dLdZ1  = None # TODO
        dLdA1  = None # TODO
    
        dA1dZ0 = None # TODO
        dLdZ0  = None # TODO
        dLdA0  = None # TODO

        if self.debug:

            self.dA2dZ1 = dA2dZ1
            self.dLdZ1  = dLdZ1
            self.dLdA1  = dLdA1

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0
        
        return NotImplemented

class MLP4:
    def __init__(self, debug=False):
        
        # Hidden Layers
        self.layers = [
            Linear(2, 4),
            Linear(4, 8),
            Linear(8, 8),
            Linear(8, 4),
            Linear(4, 2)]

        # Activations
        self.f = [
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU()]

        self.debug = debug

    def forward(self, A):

        if self.debug:

            self.Z = []
            self.A = [ A ]

        L = len(self.layers)

        for i in range(L):

            Z = None # TODO
            A = None # TODO

            if self.debug:

                self.Z.append(Z)
                self.A.append(A)

        return NotImplemented

    def backward(self, dLdA):

        if self.debug:

            self.dAdZ = []
            self.dLdZ = []
            self.dLdA = [ dLdA ]

        L = len(self.layers)

        for i in reversed(range(L)):

            dAdZ = None # TODO
            dLdZ = None # TODO
            dLdA = None # TODO

            if self.debug:

                self.dAdZ = [dAdZ] + self.dAdZ
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA

        return NotImplemented
        