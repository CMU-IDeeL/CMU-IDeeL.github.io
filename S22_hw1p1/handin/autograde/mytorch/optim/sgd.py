import numpy as np

class SGD:

    def __init__(self, model, lr=0.1, momentum=0):
    
        self.l   = model.layers
        self.L   = len(model.layers)
        self.lr  = lr
        self.mu  = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]
    
    def step(self):
    
        for i in range(self.L):
        
            if self.mu == 0:
        
                self.l[i].W = None # TODO
                self.l[i].b = None # TODO
                
            else:
        
                self.v_W[i] = None # TODO
                self.v_b[i] = None # TODO
                self.l[i].W = None # TODO
                self.l[i].b = None # TODO
    
        return None