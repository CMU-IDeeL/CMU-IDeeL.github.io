import numpy as np

class PdbExample:
    def __init__(self):
        self.batch_B = np.random.rand(10, 3, 2)
        self.batch_C = np.random.rand(10, 3, 4)

    def run(self, A):
        multiplied = A @ self.batch_B  # Get a matrix of (10, 3, 3)
        
        multiplied = multiplied * self.batch_C  # Element-wise multiplication, shape remains (10, 3, 3)

        row_means = np.mean(multiplied, axis=2)  # Correct shape: (10, 3, 1)
        
        centered = multiplied - row_means  # Subtract row mean from each row

        return centered

def main():
    pdb_example = PdbExample()    
    A = np.random.rand(10, 3, 2)

    # Current Pseudo Code:
    #     mat = A_i @ B_i           # shape (3, 2) @ (2, 3) = (3, 3)
    #     mat = mat * C_i           # element-wise multiply, shape stays (3, 3)
    #
    #     row_means = mean(mat, axis=1, keepdims=True)  # shape (3, 1)
    #     mat = mat - row_means      # subtract row mean from each row
    
    result = pdb_example.run(A)

    print("Success!")

if __name__ == "__main__":
    main()
