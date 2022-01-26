import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     = None # TODO
        sse    = None # TODO
        mse    = sse/(N*C)
        
        return NotImplemented
    
    def backward(self):
    
        dLdA = None
        
        return NotImplemented

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")

        self.softmax     = None # TODO
        crossentropy     = None # TODO
        sum_crossentropy = None # TODO
        L = sum_crossentropy / N
        
        return NotImplemented
    
    def backward(self):
    
        dLdA = None # TODO
        
        return NotImplemented
