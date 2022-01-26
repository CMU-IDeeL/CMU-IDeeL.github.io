import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = None # TODO
        
        return NotImplemented
    
    def backward(self):
    
        dAdZ = None # TODO
        
        return NotImplemented


class Tanh:
    
    def forward(self, Z):
    
        self.A = None # TODO
        
        return NotImplemented
    
    def backward(self):
    
        dAdZ = None # TODO
        
        return NotImplemented


class ReLU:
    
    def forward(self, Z):
    
        self.A = None # TODO
        
        return NotImplemented
    
    def backward(self):
    
        dAdZ = None # TODO
        
        return NotImplemented
        
        
