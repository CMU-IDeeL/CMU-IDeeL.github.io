# -*- coding: utf-8 -*-
"""
────────────────────────────────────────────────────────────────────────────────────
HW1P1 - SOLUTIONS
────────────────────────────────────────────────────────────────────────────────────
"""

from mytorch.nn import Identity, Sigmoid, Tanh, ReLU
from mytorch.nn import MSELoss, CrossEntropyLoss
from mytorch.nn import Linear
from mytorch.nn import BatchNorm1d
from mytorch.optim import SGD
from mytorch.models import MLP0, MLP1, MLP4
from mytorch.hw1p1_autograder_flags import *

import numpy as np
np.set_printoptions(
    suppress  = True,
    precision = 4)

import json


autograder_version = '2.0.0'
print("Autograder version: "+str(autograder_version))

"""
────────────────────────────────────────────────────────────────────────────────────
# Activations
────────────────────────────────────────────────────────────────────────────────────
"""

"""
────────────────────────────────────────────────────────────────────────────────────
## Identity
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_IDENTITY = DEBUG_AND_GRADE_IDENTITY_flag

if DEBUG_AND_GRADE_IDENTITY:

    print("──────────────────────────────────────────")
    print("IDENTITY | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [ 0,  1],
        [ 2,  3]], dtype="f")

    identity = Identity()

    A = identity.forward(Z)
    print("\nA =\n", A, sep="")

    dAdZ = identity.backward()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("IDENTITY | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    dAdZ_solution = np.array([
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]], dtype="f")

    print("\nA =\n", A_solution, sep="")
    print("\ndAdZ =\n", dAdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("IDENTITY | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("           Pass?")

    TEST_identity_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_identity_A)

    TEST_identity_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_identity_dAdZ)

else:

    TEST_identity_A    = False
    TEST_identity_dAdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
## Sigmoid
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_SIGMOID = DEBUG_AND_GRADE_SIGMOID_flag

if DEBUG_AND_GRADE_SIGMOID:

    print("\n──────────────────────────────────────────")
    print("SIGMOID | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [ 0,  1],
        [ 2,  3]], dtype="f")

    sigmoid = Sigmoid()

    A = sigmoid.forward(Z)
    print("\nA =\n", A, sep="")

    dAdZ = sigmoid.backward()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("──────────────────────────────────────────")
    print("SIGMOID | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [0.018,  0.0474],
        [0.1192, 0.2689],
        [0.5   , 0.7311],
        [0.8808, 0.9526]], dtype="f")

    dAdZ_solution = np.array([
        [0.0177, 0.0452],
        [0.105 , 0.1966],
        [0.25  , 0.1966],
        [0.105 , 0.0452]], dtype="f")

    print("\nA =\n", A_solution, sep="")

    print("\ndAdZ =\n", dAdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("SIGMOID | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_sigmoid_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_sigmoid_A)

    TEST_sigmoid_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_sigmoid_dAdZ)

else:

    TEST_sigmoid_A    = False
    TEST_sigmoid_dAdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
## Tanh
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_TANH = DEBUG_AND_GRADE_TANH_flag

if DEBUG_AND_GRADE_TANH:

    print("\n──────────────────────────────────────────")
    print("TANH | STUDENT OUTPUT")
    print("──────────────────────────────────────────")
    
    Z = np.array([
        [-4, -3],
        [-2, -1],
        [ 0,  1],
        [ 2,  3]], dtype="f")

    tanh = Tanh()

    A = tanh.forward(Z)
    print("\nA =\n", A, sep="")

    dAdZ = tanh.backward()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("──────────────────────────────────────────")
    print("TANH | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [-0.9993, -0.9951],
        [-0.964 , -0.7616],
        [ 0.    ,  0.7616],
        [ 0.964 ,  0.9951]], dtype="f")

    dAdZ_solution = np.array([
        [0.0013, 0.0099],
        [0.0707, 0.42  ],
        [1.    , 0.42  ],
        [0.0707, 0.0099]], dtype="f")
    
    print("\nA =\n", A_solution, sep="")
    print("\ndAdZ =\n", dAdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("TANH | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_tanh_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_tanh_A)

    TEST_tanh_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_tanh_dAdZ)

else:

    TEST_tanh_A = False
    TEST_tanh_dAdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
## ReLU
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_RELU = DEBUG_AND_GRADE_RELU_flag

if DEBUG_AND_GRADE_RELU:

    print("\n──────────────────────────────────────────")
    print("RELU | STUDENT OUTPUT")
    print("──────────────────────────────────────────")
    
    Z = np.array([
        [-4, -3],
        [-2, -1],
        [ 0,  1],
        [ 2,  3]], dtype="f")

    relu = ReLU()

    A = relu.forward(Z).copy()
    print("A =\n", A, sep="")

    dAdZ = relu.backward()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("RELU | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [2., 3.]], dtype="f")

    dAdZ_solution = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [1., 1.]], dtype="f")

    print("\nA =\n", A_solution, "\n", sep="")
    print("\ndAdZ =\n", dAdZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("RELU | CLOSENESS TEST RESULT")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_relu_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_relu_A)

    TEST_relu_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_relu_dAdZ)

else:

    TEST_relu_A = False
    TEST_relu_dAdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
# Loss
────────────────────────────────────────────────────────────────────────────────────
"""

"""
────────────────────────────────────────────────────────────────────────────────────
## MSELoss
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_MSELOSS = DEBUG_AND_GRADE_MSELOSS_flag

if DEBUG_AND_GRADE_MSELOSS:

    print("\n──────────────────────────────────────────")
    print("MSELoss | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    Y = np.array([
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.]], dtype="f")

    mse = MSELoss()

    L = mse.forward(A, Y)
    print("\nL =\n", L.round(4), sep="")

    dLdA = mse.backward()
    print("\ndLdA =\n", dLdA, sep="")

    print("\n──────────────────────────────────────────")
    print("MSELOSS | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    L_solution = np.array(6.5, dtype="f")

    dLdA_solution = np.array([
        [-4., -4.],
        [-3., -1.],
        [-1.,  1.],
        [ 2.,  2.]], dtype="f")

    print("\nL =\n", L_solution, "\n", sep="")
    print("\ndLdA =\n", dLdA_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("MSELOSS | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_mseloss_L = np.allclose(L.round(4), L_solution)
    print("Test L:   ", TEST_mseloss_L)

    TEST_mseloss_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_mseloss_dLdA)

else:

    TEST_mseloss_L = False
    TEST_mseloss_dLdA = False

"""
────────────────────────────────────────────────────────────────────────────────────
## CrossEntropyLoss
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_CROSSENTROPYLOSS = DEBUG_AND_GRADE_CROSSENTROPYLOSS_flag

if DEBUG_AND_GRADE_CROSSENTROPYLOSS:

    print("\n──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    Y = np.array([
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.]], dtype="f")

    xent = CrossEntropyLoss()

    L = xent.forward(A, Y)
    print("\nL =\n", L.round(4), sep="")

    dLdA = xent.backward()
    print("\ndLdA =\n", dLdA, sep="")

    print("──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    L_solution = np.array(0.8133, dtype="f")

    dLdA_solution = np.array([
        [ 0.2689, -0.2689],
        [-0.7311,  0.7311],
        [-0.7311,  0.7311],
        [ 0.2689, -0.2689]], dtype="f")

    print("\nL =\n", L_solution, sep="")

    print("\ndLdA =\n", dLdA_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("           Pass?")

    TEST_crossentropyloss_L = np.allclose(L.round(4), L_solution)
    print("Test L:   ", TEST_crossentropyloss_L)

    TEST_crossentropyloss_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_crossentropyloss_dLdA)

else:

    TEST_crossentropyloss_L = False
    TEST_crossentropyloss_dLdA = False

"""
────────────────────────────────────────────────────────────────────────────────────
# Linear Layer
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_LINEAR = DEBUG_AND_GRADE_LINEAR_flag

if DEBUG_AND_GRADE_LINEAR:

    print("──────────────────────────────────────────")
    print("LINEAR | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    W = np.array([
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    b = np.array([
        [-1.],
        [ 0.],
        [ 1.]], dtype="f")

    linear = Linear(2, 3, debug=True)
    linear.W = W
    linear.b = b

    Z = linear.forward(A)
    print("Z =\n", Z.round(4), sep="")

    dLdZ = np.array([
        [-4., -3., -2.],
        [-1., -0.,  1.],
        [ 2.,  3.,  4.],
        [ 5.,  6.,  7.]], dtype="f")

    dLdA = linear.backward(dLdZ)

    dZdA = linear.dZdA
    print("\ndZdA =\n", dZdA, sep="")

    dZdW = linear.dZdW
    print("\ndZdW =\n", dZdW, sep="")

    dZdi = linear.dZdi
    print("\ndZdi =\n", dZdi, sep="")

    dZdb = linear.dZdb
    print("\ndZdb =\n", dZdb, sep="")

    dLdA = linear.dLdA
    print("\ndLdA =\n", dLdA, sep="")

    dLdi = linear.dLdi
    print("\ndLdi =\n", dLdi, sep="")

    dLdA = linear.dLdA
    print("\ndLdA =\n", dLdA, sep="")

    print("\n──────────────────────────────────────────")
    print("LINEAR | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    Z_solution = np.array([
        [ 10.,  -3., -16.],
        [  4.,  -1.,  -6.],
        [ -2.,   1.,   4.],
        [ -8.,   3.,  14.]], dtype="f")

    dZdA_solution = np.array([
        [-2.,  0.,  2.],
        [-1.,  1.,  3.]], dtype="f")

    dZdW_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    dZdi_solution = None

    dZdb_solution = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    dLdA_solution = np.array([
        [ 4., -5.],
        [ 4.,  4.],
        [ 4., 13.],
        [ 4., 22.]], dtype="f")

    dLdi_solution = None

    dLdA_solution = np.array([
        [ 4., -5.],
        [ 4.,  4.],
        [ 4., 13.],
        [ 4., 22.]], dtype="f")
    
    print("\ndZdA =\n", dZdA_solution, sep="")
    print("\ndZdW =\n", dZdW_solution, sep="")
    print("\ndZdi =\n", dZdi_solution, sep="")
    print("\ndZdb =\n", dZdb_solution, sep="")
    print("\ndLdA =\n", dLdA_solution, sep="")
    print("\ndLdi =\n", dLdi_solution, sep="")
    print("\ndLdA =\n", dLdA_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("LINEAR | TEST RESULTS")
    print("──────────────────────────────────────────")
    
    print("\n           Pass?")

    TEST_linear_Z = np.allclose(Z.round(4), Z_solution)
    print("Test Z:   ", TEST_linear_Z)

    TEST_linear_dZdA = np.allclose(dZdA.round(4), dZdA_solution)
    print("Test dZdA:", TEST_linear_dZdA)

    TEST_linear_dZdW = np.allclose(dZdW.round(4), dZdW_solution)
    print("Test dZdW:", TEST_linear_dZdW)

    TEST_linear_dZdi = dZdi is None
    print("Test dZdi:", TEST_linear_dZdi)

    TEST_linear_dZdb = np.allclose(dZdb.round(4), dZdb_solution)
    print("Test dZdb:", TEST_linear_dZdb)

    TEST_linear_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_linear_dLdA)

    TEST_linear_dLdi = dLdi is None
    print("Test dLdi:", TEST_linear_dLdi)

    TEST_linear_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_linear_dLdA)

else:

    TEST_linear_Z    = False
    TEST_linear_dZdA = False
    TEST_linear_dZdW = False
    TEST_linear_dZdi = False
    TEST_linear_dZdb = False
    TEST_linear_dLdA = False
    TEST_linear_dLdi = False

"""
────────────────────────────────────────────────────────────────────────────────────
# SGD
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_SGD = DEBUG_AND_GRADE_SGD_flag

if DEBUG_AND_GRADE_SGD:

    print("\n──────────────────────────────────────────")
    print("SGD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    class PseudoModel:
        def __init__(self):
            self.layers = [ Linear(3,2) ]
            self.f      = [ ReLU() ]
        def forward(self, A):
            return NotImplemented
        def backward(self):
            return NotImplemented

    # Create Example Model
    pseudo_model = PseudoModel()

    pseudo_model.layers[0].W    = np.ones((3,2))
    pseudo_model.layers[0].dLdW = np.ones((3,2))/10
    pseudo_model.layers[0].b    = np.ones((3,1))
    pseudo_model.layers[0].dLdb = np.ones((3,1))/10

    print("\nInitialized Parameters:\n")
    print("W =\n", pseudo_model.layers[0].W, "\n", sep="")
    print("b =\n", pseudo_model.layers[0].b, "\n", sep="")

    # Test Example Models
    optimizer = SGD(pseudo_model, lr=1)
    optimizer.step()

    print("Parameters After SGD (Step=1)\n")

    W_1 = pseudo_model.layers[0].W.copy()
    b_1 = pseudo_model.layers[0].b.copy()
    print("W =\n", W_1, "\n", sep="")
    print("b =\n", b_1, "\n", sep="")

    optimizer.step()

    print("Parameters After SGD (Step=2)\n")

    W_2 = pseudo_model.layers[0].W
    b_2 = pseudo_model.layers[0].b
    print("W =\n", W_2, "\n", sep="")
    print("b =\n", b_2, "\n", sep="")

    print("──────────────────────────────────────────")
    print("SGD | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    W_1_solution = np.array([
        [0.9, 0.9],
        [0.9, 0.9],
        [0.9, 0.9]], dtype="f")

    b_1_solution = np.array([
        [0.9],
        [0.9],
        [0.9]], dtype="f")

    W_2_solution = np.array([
        [0.8, 0.8],
        [0.8, 0.8],
        [0.8, 0.8]], dtype="f")

    b_2_solution = np.array([
        [0.8],
        [0.8],
        [0.8]], dtype="f")

    print("\nParameters After SGD (Step=1)\n")

    print("W =\n", W_1_solution, "\n", sep="")
    print("b =\n", b_1_solution, "\n", sep="")

    print("Parameters After SGD (Step=2)\n")

    print("W =\n", W_2_solution, "\n", sep="")
    print("b =\n", b_2_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("SGD | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("                 Pass?")

    TEST_sgd_W_1 = np.allclose(W_1.round(4), W_1_solution)
    print("Test W (Step 1):", TEST_sgd_W_1)

    TEST_sgd_b_1 = np.allclose(b_1.round(4), b_1_solution)
    print("Test b (Step 1):", TEST_sgd_b_1)

    TEST_sgd_W_2 = np.allclose(W_2.round(4), W_2_solution)
    print("Test W (Step 2):", TEST_sgd_W_2)

    TEST_sgd_b_2 = np.allclose(b_2.round(4), b_2_solution)
    print("Test b (Step 2):", TEST_sgd_b_2)

else:

    TEST_sgd_W_1 = False
    TEST_sgd_b_1 = False
    TEST_sgd_W_2 = False
    TEST_sgd_b_2 = False

"""
────────────────────────────────────────────────────────────────────────────────────
# Multilayer Perceptrons
────────────────────────────────────────────────────────────────────────────────────
"""

"""
────────────────────────────────────────────────────────────────────────────────────
## MLP0
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_MLP0 = DEBUG_AND_GRADE_MLP0_flag

if DEBUG_AND_GRADE_MLP0:

    print("\n──────────────────────────────────────────")
    print("MLP0 | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A0 = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    W0 = np.array([
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    b0 = np.array([
        [-1.],
        [ 0.],
        [ 1.]], dtype="f")

    mlp0 = MLP0(debug=True)
    mlp0.layers[0].W = W0
    mlp0.layers[0].b = b0

    A1 = mlp0.forward(A)
    Z0 = mlp0.Z0

    print("Z0 =\n", Z0.round(4), sep="")

    print("\nA1 =\n", A1.round(4), sep="")

    dLdA1 = np.array([
        [-4., -3., -2.],
        [-1., -0.,  1.],
        [ 2.,  3.,  4.],
        [ 5.,  6.,  7.]], dtype="f")

    mlp0.backward(dLdA1)

    dA1dZ0 = mlp0.dA1dZ0
    print("\ndA1dZ0 =\n", dA1dZ0, sep="")

    dLdZ0 = mlp0.dLdZ0
    print("\ndLdZ0 =\n", dLdZ0, sep="")

    dLdA0 = mlp0.dLdA0
    print("\ndLdA0 =\n", dLdA0, sep="")

    dLdW0 = mlp0.layers[0].dLdW
    print("\ndLdW0 =\n", dLdW0, sep="")

    dLdb0 = mlp0.layers[0].dLdb
    print("\ndLdb0 =\n", dLdb0, sep="")

if DEBUG_AND_GRADE_MLP0:

    print("──────────────────────────────────────────")
    print("MLP0 | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    Z0_solution = np.array([
        [ 10.,  -3., -16.],
        [  4.,  -1.,  -6.],
        [ -2.,   1.,   4.],
        [ -8.,   3.,  14.]], dtype="f")

    A1_solution =np.array([
        [10.,  0.,  0.],
        [ 4.,  0.,  0.],
        [ 0.,  1.,  4.],
        [ 0.,  3., 14.]], dtype="f")

    dA1dZ0_solution = np.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 1.],
        [0., 1., 1.]], dtype="f")

    dLdZ0_solution = np.array([
        [-4., -0., -0.],
        [-1., -0.,  0.],
        [ 0.,  3.,  4.],
        [ 0.,  6.,  7.]], dtype="f")

    dLdA0_solution = np.array([
        [ 8.,  4.],
        [ 2.,  1.],
        [ 8., 15.],
        [14., 27.]], dtype="f")

    dLdW0_solution = np.array([
        [4.5,  3.25],
        [3. ,  5.25],
        [3.5,  6.25]], dtype="f")

    dLdb0_solution = np.array([
        [-1.25],
        [ 2.25],
        [ 2.75]], dtype="f")

if DEBUG_AND_GRADE_MLP0:

    print("\n──────────────────────────────────────────")
    print("MLP0 | TEST RESULTS")
    print("──────────────────────────────────────────")


    print("             Pass?")

    TEST_mlp0_Z0 = np.allclose(Z0.round(4), Z0_solution)
    print("Test Z0:    ", TEST_mlp0_Z0)

    TEST_mlp0_A1 = np.allclose(A1.round(4), A1_solution)
    print("Test A1:    ", TEST_mlp0_A1)

    TEST_mlp0_dA1dZ0 = np.allclose(dA1dZ0.round(4), dA1dZ0_solution)
    print("Test dA1dZ0:", TEST_mlp0_dA1dZ0)

    TEST_mlp0_dLdZ0 = np.allclose(dLdZ0.round(4), dLdZ0_solution)
    print("Test dLdZ0: ", TEST_mlp0_dLdZ0)

    TEST_mlp0_dLdA0 = np.allclose(dLdA0.round(4), dLdA0_solution)
    print("Test dLdA0: ", TEST_mlp0_dLdA0)

    TEST_mlp0_dLdW0 = np.allclose(dLdW0.round(4), dLdW0_solution)
    print("Test dLdW0: ", TEST_mlp0_dLdW0)

    TEST_mlp0_dLdb0 = np.allclose(dLdb0.round(4), dLdb0_solution)
    print("Test dLdb0: ", TEST_mlp0_dLdb0)

else:

    TEST_mlp0_Z0     = False
    TEST_mlp0_A1     = False
    TEST_mlp0_dA1dZ0 = False
    TEST_mlp0_dLdZ0  = False
    TEST_mlp0_dLdA0  = False
    TEST_mlp0_dLdW0  = False
    TEST_mlp0_dLdb0  = False

"""## MPL1"""

DEBUG_AND_GRADE_MLP1 = DEBUG_AND_GRADE_MLP1_flag

if DEBUG_AND_GRADE_MLP1:

    print("──────────────────────────────────────────")
    print("MLP0 | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A0 = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    W0 = np.array([
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    b0 = np.array([
        [-1.],
        [ 0.],
        [ 1.]], dtype="f")

    W1 = np.array([
        [-2., -1., 0],
        [ 1.,  2., 3]], dtype="f")

    b1 = np.array([
        [-1.],
        [ 1.]], dtype="f")

    mlp1 = MLP1(debug=True)
    mlp1.layers[0].W = W0
    mlp1.layers[0].b = b0
    mlp1.layers[1].W = W1
    mlp1.layers[1].b = b1

    A2 = mlp1.forward(A)

    Z0 = mlp1.Z0
    A1 = mlp1.A1
    Z1 = mlp1.Z1

    print("Z0 =\n", Z0.round(4), sep="")

    print("\nA1 =\n", A1.round(4), sep="")

    print("\nZ1 =\n", Z1.round(4), sep="")

    print("\nA2 =\n", A2.round(4), sep="")

    dLdA2 = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.]], dtype="f")

    mlp1.backward(dLdA2)

    dA2dZ1 = mlp1.dA2dZ1
    print("\ndA2dZ1 =\n", dA2dZ1, sep="")

    dLdZ1 = mlp1.dLdZ1
    print("\ndLdZ1 =\n", dLdZ1, sep="")

    dLdA1 = mlp1.dLdA1
    print("\ndLdA1 =\n", dLdA1, sep="")

    dA1dZ0 = mlp1.dA1dZ0
    print("\ndA1dZ0 =\n", dA1dZ0, sep="")

    dLdZ0 = mlp1.dLdZ0
    print("\ndLdZ0 =\n", dLdZ0, sep="")

    dLdA0 = mlp1.dLdA0
    print("\ndLdA0 =\n", dLdA0, sep="")

    dLdW0 = mlp1.layers[0].dLdW
    print("\ndLdW0 =\n", dLdW0, sep="")

    dLdb0 = mlp1.layers[0].dLdb
    print("\ndLdb0 =\n", dLdb0, sep="")

    print("──────────────────────────────────────────")
    print("MLP1 | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    Z0_solution = np.array([
        [ 10.,  -3., -16.],
        [  4.,  -1.,  -6.],
        [ -2.,   1.,   4.],
        [ -8.,   3.,  14.]], dtype="f")

    A1_solution = np.array([
        [10.,  0.,  0.],
        [ 4.,  0.,  0.],
        [ 0.,  1.,  4.],
        [ 0.,  3., 14.]], dtype="f")

    Z1_solution = np.array([
        [-21.,  11.],
        [ -9.,   5.],
        [ -2.,  15.],
        [ -4.,  49.]], dtype="f")

    A2_solution = np.array([
        [ 0., 11.],
        [ 0.,  5.],
        [ 0., 15.],
        [ 0., 49.]], dtype="f")

    dA2dZ1_solution = np.array([
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.]], dtype="f")

    dLdZ1_solution = np.array([
        [-0., -3.],
        [-0., -1.],
        [ 0.,  1.],
        [ 0.,  3.]], dtype="f")

    dLdA1_solution = np.array([
        [-3., -6., -9.],
        [-1., -2., -3.],
        [ 1.,  2.,  3.],
        [ 3.,  6.,  9.]], dtype="f")

    dA1dZ0_solution = np.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 1.],
        [0., 1., 1.]], dtype="f")

    dLdZ0_solution = np.array([
        [-3., -0., -0.],
        [-1., -0., -0.],
        [ 0.,  2.,  3.],
        [ 0.,  6.,  9.]], dtype="f")

    dLdA0_solution = np.array([
        [ 6.,  3.],
        [ 2.,  1.],
        [ 6., 11.],
        [18., 33.]], dtype="f")

    dLdW0_solution = np.array([
        [3.5, 2.5],
        [3. , 5. ],
        [4.5, 7.5]], dtype="f")

    dLdb0_solution = np.array([
        [-1.],
        [ 2.],
        [ 3.]], dtype="f")
    
    print("\ndA2dZ1 =\n", dA2dZ1, sep="")
    print("\ndLdZ1 =\n",  dLdZ1,  sep="")
    print("\ndLdA1 =\n",  dLdA1,  sep="")
    print("\ndA1dZ0 =\n", dA1dZ0, sep="")
    print("\ndLdZ0 =\n",  dLdZ0,  sep="")
    print("\ndLdA0 =\n",  dLdA0,  sep="")
    print("\ndLdW0 =\n",  dLdW0,  sep="")
    print("\ndLdb0 =\n",  dLdb0,  sep="")

    print("\n──────────────────────────────────────────")
    print("MLP1 | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("             Pass?")

    TEST_mlp1_Z0 = np.allclose(Z0.round(4), Z0_solution)
    print("Test Z0:    ", TEST_mlp1_Z0)

    TEST_mlp1_A1 = np.allclose(A1.round(4), A1_solution)
    print("Test A1:    ", TEST_mlp1_A1)

    TEST_mlp1_Z1 = np.allclose(Z1.round(4), Z1_solution)
    print("Test Z1:    ", TEST_mlp1_Z1)

    TEST_mlp1_A2 = np.allclose(A2.round(4), A2_solution)
    print("Test A2:    ", TEST_mlp1_A2)

    TEST_mlp1_dA2dZ1 = np.allclose(dA2dZ1.round(4), dA2dZ1_solution)
    print("Test dA2dZ1:", TEST_mlp1_dA2dZ1)

    TEST_mlp1_dLdZ1 = np.allclose(dLdZ1.round(4), dLdZ1_solution)
    print("Test dLdZ1: ", TEST_mlp1_dLdZ1)

    TEST_mlp1_dLdA1 = np.allclose(dLdA1.round(4), dLdA1_solution)
    print("Test dLdA1: ", TEST_mlp1_dLdA1)

    TEST_mlp1_dA1dZ0 = np.allclose(dA1dZ0.round(4), dA1dZ0_solution)
    print("Test dA1dZ0:", TEST_mlp1_dA1dZ0)

    TEST_mlp1_dLdZ0 = np.allclose(dLdZ0.round(4), dLdZ0_solution)
    print("Test dLdZ0: ", TEST_mlp1_dLdZ0)

    TEST_mlp1_dLdA0 = np.allclose(dLdA0.round(4), dLdA0_solution)
    print("Test dLdA0: ", TEST_mlp1_dLdA0)

    TEST_mlp1_dLdW0 = np.allclose(dLdW0.round(4), dLdW0_solution)
    print("Test dLdW0: ", TEST_mlp1_dLdW0)

    TEST_mlp1_dLdb0 = np.allclose(dLdb0.round(4), dLdb0_solution)
    print("Test dLdb0: ", TEST_mlp1_dLdb0)

else:

    TEST_mlp1_Z0     = False
    TEST_mlp1_A1     = False
    TEST_mlp1_Z1     = False
    TEST_mlp1_A2     = False
    TEST_mlp1_dA2dZ1 = False
    TEST_mlp1_dLdZ1  = False
    TEST_mlp1_dLdA1  = False
    TEST_mlp1_dA1dZ0 = False
    TEST_mlp1_dLdZ0  = False
    TEST_mlp1_dLdA0  = False
    TEST_mlp1_dLdW0  = False
    TEST_mlp1_dLdb0  = False

"""
────────────────────────────────────────────────────────────────────────────────────
## MLP4
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_MLP4 = DEBUG_AND_GRADE_MLP4_flag

if DEBUG_AND_GRADE_MLP4:

    print("\n──────────────────────────────────────────")
    print("MLP4 FORWARD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A0 = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.],
        [ 4.,  5.]], dtype="f")

    W0 = np.array([
        [0., 1.],
        [1., 2.],
        [2., 0.],
        [0., 1.]], dtype="f")

    b0 = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W1 = np.array([
        [0., 2., 1., 0.],
        [1., 0., 2., 1.],
        [2., 1., 0., 2.],
        [0., 2., 1., 0.],
        [1., 0., 2., 1.],
        [2., 1., 0., 2.],
        [0., 2., 1., 0.],
        [1., 0., 2., 1.]], dtype="f")

    b1 = np.array([
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W2 = np.array([
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.],
        [2., 1., 0., 2., 1., 0., 2., 1.],
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.],
        [2., 1., 0., 2., 1., 0., 2., 1.],
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.]], dtype="f")

    b2 = np.array([
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W3 = np.array([
        [0., 1., 2., 0., 1., 2., 0., 1.],
        [1., 2., 0., 1., 2., 0., 1., 2.],
        [2., 0., 1., 2., 0., 1., 2., 0.],
        [0., 1., 2., 0., 1., 2., 0., 1.]], dtype="f")

    b3 = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W4 = np.array([
        [0., 2., 1., 0.],
        [1., 0., 2., 1.]], dtype="f")

    b4 = np.array([
        [1.],
        [1.]], dtype="f")

    mlp4 = MLP4(debug=True)

    mlp4.layers[0].W = W0
    mlp4.layers[0].b = b0
    mlp4.layers[1].W = W1
    mlp4.layers[1].b = b1
    mlp4.layers[2].W = W2
    mlp4.layers[2].b = b2
    mlp4.layers[3].W = W3
    mlp4.layers[3].b = b3
    mlp4.layers[4].W = W4
    mlp4.layers[4].b = b4

    A5 = mlp4.forward(A0)

    Z0 = mlp4.Z[0]; print("\nZ0 =\n", Z0, sep="")
    A1 = mlp4.A[1]; print("\nA1 =\n", A1, sep="")
    Z1 = mlp4.Z[1]; print("\nZ1 =\n", Z1, sep="")
    A2 = mlp4.A[2]; print("\nA2 =\n", A2, sep="")
    Z2 = mlp4.Z[2]; print("\nZ2 =\n", Z2, sep="")
    A3 = mlp4.A[3]; print("\nA3 =\n", A3, sep="")
    Z3 = mlp4.Z[3]; print("\nZ3 =\n", Z3, sep="")
    A4 = mlp4.A[4]; print("\nA4 =\n", A4, sep="")
    Z4 = mlp4.Z[4]; print("\nZ4 =\n", Z4, sep="")

    print("\nA5 =\n", A5, sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 FORWARD | TEST RESULTS")
    print("──────────────────────────────────────────")

    Z0_solution = np.array([
        [-2., -9., -7., -2.],
        [ 0., -3., -3.,  0.],
        [ 2.,  3.,  1.,  2.],
        [ 4.,  9.,  5.,  4.],
        [ 6., 15.,  9.,  6.]], dtype="f")

    A1_solution = np.array([
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 2.,  3.,  1.,  2.],
        [ 4.,  9.,  5.,  4.],
        [ 6., 15.,  9.,  6.]], dtype="f")

    Z1_solution = np.array([
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 8.,  7., 12.,  8.,  7., 12.,  8.,  7.],
        [24., 19., 26., 24., 19., 26., 24., 19.],
        [40., 31., 40., 40., 31., 40., 40., 31.]], dtype="f")

    A2_solution = np.array([
        [  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.],
        [  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.],
        [  8.,   7.,  12.,   8.,   7.,  12.,   8.,   7.],
        [ 24.,  19.,  26.,  24.,  19.,  26.,  24.,  19.],
        [ 40.,  31.,  40.,  40.,  31.,  40.,  40.,  31.]], dtype="f")

    Z2_solution = np.array([
        [  9.,   8.,  10.,   9.,   8.,  10.,   9.,   8.],
        [  9.,   8.,  10.,   9.,   8.,  10.,   9.,   8.],
        [ 67.,  73.,  70.,  67.,  73.,  70.,  67.,  73.],
        [167., 177., 202., 167., 177., 202., 167., 177.],
        [267., 281., 334., 267., 281., 334., 267., 281.]], dtype="f")

    A3_solution = np.array([
        [  9.,   8.,  10.,   9.,   8.,  10.,   9.,   8.],
        [  9.,   8.,  10.,   9.,   8.,  10.,   9.,   8.],
        [ 67.,  73.,  70.,  67.,  73.,  70.,  67.,  73.],
        [167., 177., 202., 167., 177., 202., 167., 177.],
        [267., 281., 334., 267., 281., 334., 267., 281.]], dtype="f")

    Z3_solution = np.array([
        [  65.,   76.,   75.,   65.],
        [  65.,   76.,   75.,   65.],
        [ 500.,  640.,  543.,  500.],
        [1340., 1564., 1407., 1340.],
        [2180., 2488., 2271., 2180.]], dtype="f")

    A4_solution = np.array([
        [  65.,   76.,   75.,   65.],
        [  65.,   76.,   75.,   65.],
        [ 500.,  640.,  543.,  500.],
        [1340., 1564., 1407., 1340.],
        [2180., 2488., 2271., 2180.]], dtype="f")

    Z4_solution = np.array([
        [ 228.,  281.],
        [ 228.,  281.],
        [1824., 2087.],
        [4536., 5495.],
        [7248., 8903.]], dtype="f")

    A5_solution = np.array([
        [ 228.,  281.],
        [ 228.,  281.],
        [1824., 2087.],
        [4536., 5495.],
        [7248., 8903.]], dtype="f")

    print("\nZ0 =\n", Z0_solution, sep="")
    print("\nA1 =\n", A1_solution, sep="")
    print("\nZ1 =\n", Z1_solution, sep="")
    print("\nA2 =\n", A2_solution, sep="")
    print("\nZ2 =\n", Z2_solution, sep="")
    print("\nA3 =\n", A3_solution, sep="")
    print("\nZ3 =\n", Z3_solution, sep="")
    print("\nA4 =\n", A4_solution, sep="")
    print("\nZ4 =\n", Z4_solution, sep="")

    print("\nA5 =\n", A5_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 FORWARD | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("             Pass?")

    TEST_mlp4_Z0 = np.allclose(Z0.round(4), Z0_solution)
    print("Test Z0:    ", TEST_mlp4_Z0)

    TEST_mlp4_A1 = np.allclose(A1.round(4), A1_solution)
    print("Test A1:    ", TEST_mlp4_A1)

    TEST_mlp4_Z1 = np.allclose(Z1.round(4), Z1_solution)
    print("Test Z1:    ", TEST_mlp4_Z1)

    TEST_mlp4_A2 = np.allclose(A2.round(4), A2_solution)
    print("Test A2:    ", TEST_mlp4_A2)

    TEST_mlp4_Z2 = np.allclose(Z2.round(4), Z2_solution)
    print("Test Z2:    ", TEST_mlp4_Z2)

    TEST_mlp4_A3 = np.allclose(A3.round(4), A3_solution)
    print("Test A3:    ", TEST_mlp4_A3)

    TEST_mlp4_Z3 = np.allclose(Z3.round(4), Z3_solution)
    print("Test Z3:    ", TEST_mlp4_Z3)

    TEST_mlp4_A4 = np.allclose(A4.round(4), A4_solution)
    print("Test A4:    ", TEST_mlp4_A4)

    TEST_mlp4_Z4 = np.allclose(Z4.round(4), Z4_solution)
    print("Test Z4:    ", TEST_mlp4_Z4)

    TEST_mlp4_A5 = np.allclose(A5.round(4), A5_solution)
    print("Test A5:    ", TEST_mlp4_A5)

else:

    TEST_mlp4_Z0 = False
    TEST_mlp4_A1 = False
    TEST_mlp4_Z1 = False
    TEST_mlp4_A2 = False
    TEST_mlp4_Z2 = False
    TEST_mlp4_A3 = False
    TEST_mlp4_Z3 = False
    TEST_mlp4_A4 = False
    TEST_mlp4_Z4 = False
    TEST_mlp4_A5 = False

if DEBUG_AND_GRADE_MLP4:

    print("\n──────────────────────────────────────────")
    print("MLP4 BACKWARD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    dLdA5 = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.],
        [ 4.,  5.]], dtype="f")

    mlp4.backward(dLdA5)

    dA1dZ0 = mlp4.dAdZ[0]; print("\ndA1dZ0 =\n", dA1dZ0, sep="")
    dLdZ0  = mlp4.dLdZ[0]; print("\ndLdZ0 =\n",  dLdZ0,  sep="")
    dLdA0  = mlp4.dLdA[0]; print("\ndLdA0 =\n",  dLdA0,  sep="")
    dLdW0  = mlp4.layers[0].dLdW; print("\ndLdW0 =\n",  dLdW0,  sep="")
    dLdb0  = mlp4.layers[0].dLdb; print("\ndLdb0 =\n",  dLdb0,  sep="")

    dA2dZ1 = mlp4.dAdZ[1]; print("\ndA2dZ1 =\n", dA2dZ1, sep="")
    dLdZ1  = mlp4.dLdZ[1]; print("\ndLdZ1 =\n",  dLdZ1,  sep="")
    dLdA1  = mlp4.dLdA[1]; print("\ndLdA1 =\n",  dLdA1,  sep="")
    dLdW1  = mlp4.layers[1].dLdW; print("\ndLdW1 =\n",  dLdW1,  sep="")
    dLdb1  = mlp4.layers[1].dLdb; print("\ndLdb1 =\n",  dLdb1,  sep="")

    dA3dZ2 = mlp4.dAdZ[2]; print("\ndA3dZ2 =\n", dA3dZ2, sep="")
    dLdZ2  = mlp4.dLdZ[2]; print("\ndLdZ2 =\n",  dLdZ2,  sep="")
    dLdA2  = mlp4.dLdA[2]; print("\ndLdA2 =\n",  dLdA2,  sep="")
    dLdW2  = mlp4.layers[2].dLdW; print("\ndLdW2 =\n",  dLdW2,  sep="")
    dLdb2  = mlp4.layers[2].dLdb; print("\ndLdb2 =\n",  dLdb2,  sep="")

    dA4dZ3 = mlp4.dAdZ[3]; print("\ndA4dZ3 =\n", dA4dZ3, sep="")
    dLdZ3  = mlp4.dLdZ[3]; print("\ndLdZ3 =\n",  dLdZ3,  sep="")
    dLdA3  = mlp4.dLdA[3]; print("\ndLdA3 =\n",  dLdA3,  sep="")
    dLdW3  = mlp4.layers[3].dLdW; print("\ndLdW3 =\n",  dLdW3,  sep="")
    dLdb3  = mlp4.layers[3].dLdb; print("\ndLdb3 =\n",  dLdb3,  sep="")

    dA5dZ4 = mlp4.dAdZ[4]; print("\ndA5dZ4 =\n", dA5dZ4, sep="")
    dLdZ4  = mlp4.dLdZ[4]; print("\ndLdZ4 =\n",  dLdZ4,  sep="")
    dLdA4  = mlp4.dLdA[4]; print("\ndLdA4 =\n",  dLdA4,  sep="")
    dLdW4  = mlp4.layers[4].dLdW; print("\ndLdW4 =\n",  dLdW4,  sep="")
    dLdb4  = mlp4.layers[4].dLdb; print("\ndLdb4 =\n",  dLdb4,  sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 BACKWARD | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    dA1dZ0_solution = np.array([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype="f")

    dLdZ0_solution = np.array([
        [  -0.,   -0.,   -0.,   -0.],
        [  -0.,   -0.,   -0.,   -0.],
        [ 204.,  228.,  306.,  204.],
        [1056., 1020., 1326., 1056.],
        [1908., 1812., 2346., 1908.]], dtype="f")

    dLdA0_solution = np.array([
        [   0.,    0.],
        [   0.,    0.],
        [ 840.,  864.],
        [3672., 4152.],
        [6504., 7440.]], dtype="f")

    dLdW0_solution = np.array([
        [1948.8, 2582.4],
        [1857.6, 2469.6],
        [2407.2, 3202.8],
        [1948.8, 2582.4]], dtype="f")

    dLdb0_solution = np.array([
        [633.6],
        [612. ],
        [795.6],
        [633.6]], dtype="f")

    dA2dZ1_solution = np.array([
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.]], dtype="f")

    dLdZ1_solution = np.array([
        [-154., -212., -216., -154., -212., -216., -154., -212.],
        [ -62.,  -88.,  -96.,  -62.,  -88.,  -96.,  -62.,  -88.],
        [  30.,   36.,   24.,   30.,   36.,   24.,   30.,   36.],
        [ 122.,  160.,  144.,  122.,  160.,  144.,  122.,  160.],
        [ 214.,  284.,  264.,  214.,  284.,  264.,  214.,  284.]], dtype="f")

    dLdA1_solution = np.array([
        [-1500., -1356., -1734., -1500.],
        [ -648.,  -564.,  -714.,  -648.],
        [  204.,   228.,   306.,   204.],
        [ 1056.,  1020.,  1326.,  1056.],
        [ 1908.,  1812.,  2346.,  1908.]], dtype="f")

    dLdW1_solution = np.array([
        [ 366.4,  879.6,  513.2,  366.4],
        [ 483.2, 1161.6,  678.4,  483.2],
        [ 441.6, 1065.6,  624. ,  441.6],
        [ 366.4,  879.6,  513.2,  366.4],
        [ 483.2, 1161.6,  678.4,  483.2],
        [ 441.6, 1065.6,  624. ,  441.6],
        [ 366.4,  879.6,  513.2,  366.4],
        [ 483.2, 1161.6,  678.4,  483.2]], dtype="f")

    dLdb1_solution = np.array([
        [30.],
        [36.],
        [24.],
        [30.],
        [36.],
        [24.],
        [30.],
        [36.]], dtype="f")

    dA3dZ2_solution = np.array([
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.]], dtype="f")

    dLdZ2_solution = np.array([
        [-28., -22., -22., -28., -22., -22., -28., -22.],
        [-12., -10.,  -8., -12., -10.,  -8., -12., -10.],
        [  4.,   2.,   6.,   4.,   2.,   6.,   4.,   2.],
        [ 20.,  14.,  20.,  20.,  14.,  20.,  20.,  14.],
        [ 36.,  26.,  34.,  36.,  26.,  34.,  36.,  26.]], dtype="f")

    dLdA2_solution = np.array([
        [-154., -212., -216., -154., -212., -216., -154., -212.],
        [ -62.,  -88.,  -96.,  -62.,  -88.,  -96.,  -62.,  -88.],
        [  30.,   36.,   24.,   30.,   36.,   24.,   30.,   36.],
        [ 122.,  160.,  144.,  122.,  160.,  144.,  122.,  160.],
        [ 214.,  284.,  264.,  214.,  284.,  264.,  214.,  284.]], dtype="f")

    dLdW2_solution = np.array([
        [382.4, 296.8, 393.6, 382.4, 296.8, 393.6, 382.4, 296.8],
        [272. , 210.8, 279.2, 272. , 210.8, 279.2, 272. , 210.8],
        [371.6, 289.2, 384.4, 371.6, 289.2, 384.4, 371.6, 289.2],
        [382.4, 296.8, 393.6, 382.4, 296.8, 393.6, 382.4, 296.8],
        [272. , 210.8, 279.2, 272. , 210.8, 279.2, 272. , 210.8],
        [371.6, 289.2, 384.4, 371.6, 289.2, 384.4, 371.6, 289.2],
        [382.4, 296.8, 393.6, 382.4, 296.8, 393.6, 382.4, 296.8],
        [272. , 210.8, 279.2, 272. , 210.8, 279.2, 272. , 210.8]], dtype="f")

    dLdb2_solution = np.array([
        [4.],
        [2.],
        [6.],
        [4.],
        [2.],
        [6.],
        [4.],
        [2.]], dtype="f")

    dA4dZ3_solution = np.array([
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype="f")

    dLdZ3_solution = np.array([
        [ -3.,  -8., -10.,  -3.],
        [ -1.,  -4.,  -4.,  -1.],
        [  1.,   0.,   2.,   1.],
        [  3.,   4.,   8.,   3.],
        [  5.,   8.,  14.,   5.]], dtype="f")

    dLdA3_solution = np.array([
        [-28., -22., -22., -28., -22., -22., -28., -22.],
        [-12., -10.,  -8., -12., -10.,  -8., -12., -10.],
        [  4.,   2.,   6.,   4.,   2.,   6.,   4.,   2.],
        [ 20.,  14.,  20.,  20.,  14.,  20.,  20.,  14.],
        [ 36.,  26.,  34.,  36.,  26.,  34.,  36.,  26.]], dtype="f")

    dLdW3_solution = np.array([
        [ 373.4,  395.4,  461.2,  373.4,  395.4,  461.2,  373.4,  395.4],
        [ 539.2,  572. ,  672. ,  539.2,  572. ,  672. ,  539.2,  572. ],
        [1016.4, 1076.8, 1258.4, 1016.4, 1076.8, 1258.4, 1016.4, 1076.8],
        [ 373.4,  395.4,  461.2,  373.4,  395.4,  461.2,  373.4,  395.4]], dtype="f")

    dLdb3_solution = np.array([
        [1.],
        [0.],
        [2.],
        [1.]], dtype="f")

    dA5dZ4_solution = np.array([
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]], dtype="f")

    dLdZ4_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.],
        [ 4.,  5.]], dtype="f")

    dLdA4_solution = np.array([
        [ -3.,  -8., -10.,  -3.],
        [ -1.,  -4.,  -4.,  -1.],
        [  1.,   0.,   2.,   1.],
        [  3.,   4.,   8.,   3.],
        [  5.,   8.,  14.,   5.]], dtype="f")

    dLdW4_solution = np.array([
        [2202.,  2524.8, 2289.6, 2202. ],
        [3032.,  3493.6, 3163.8, 3032. ]], dtype="f")

    dLdb4_solution = np.array([
        [0.],
        [1.]], dtype="f")
    
    print("\ndA1dZ0 =\n", dA1dZ0_solution, sep="")
    print("\ndLdZ0 =\n",  dLdZ0_solution,  sep="")
    print("\ndLdA0 =\n",  dLdA0_solution,  sep="")
    print("\ndLdW0 =\n",  dLdW0_solution,  sep="")
    print("\ndLdb0 =\n",  dLdb0_solution,  sep="")

    print("\ndA2dZ1 =\n", dA2dZ1_solution, sep="")
    print("\ndLdZ1 =\n",  dLdZ1_solution,  sep="")
    print("\ndLdA1 =\n",  dLdA1_solution,  sep="")
    print("\ndLdW1 =\n",  dLdW1_solution,  sep="")
    print("\ndLdb1 =\n",  dLdb1_solution,  sep="")

    print("\ndA3dZ2 =\n", dA3dZ2_solution, sep="")
    print("\ndLdZ2 =\n",  dLdZ2_solution,  sep="")
    print("\ndLdA2 =\n",  dLdA2_solution,  sep="")
    print("\ndLdW2 =\n",  dLdW2_solution,  sep="")
    print("\ndLdb2 =\n",  dLdb2_solution,  sep="")

    print("\ndA4dZ3 =\n", dA4dZ3_solution, sep="")
    print("\ndLdZ3 =\n",  dLdZ3_solution,  sep="")
    print("\ndLdA3 =\n",  dLdA3_solution,  sep="")
    print("\ndLdW3 =\n",  dLdW3_solution,  sep="")
    print("\ndLdb3 =\n",  dLdb3_solution,  sep="")

    print("\ndA5dZ4 =\n", dA5dZ4_solution, sep="")
    print("\ndLdZ4 =\n",  dLdZ4_solution,  sep="")
    print("\ndLdA4 =\n",  dLdA4_solution,  sep="")
    print("\ndLdW4 =\n",  dLdW4_solution,  sep="")
    print("\ndLdb4 =\n",  dLdb4_solution,  sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 BACKWARD | TEST RESULTS")
    print("──────────────────────────────────────────")
    
    print("\n             Pass?")

    TEST_mlp4_dA1dZ0 = np.allclose(dA1dZ0.round(4), dA1dZ0_solution)
    print("Test dA1dZ0:", TEST_mlp4_dA1dZ0)

    TEST_mlp4_dLdZ0 = np.allclose(dLdZ0.round(4), dLdZ0_solution)
    print("Test dLdZ0: ", TEST_mlp4_dLdZ0)

    TEST_mlp4_dLdA0 = np.allclose(dLdA0.round(4), dLdA0_solution)
    print("Test dLdA0: ", TEST_mlp4_dLdA0)

    TEST_mlp4_dLdW0 = np.allclose(dLdW0.round(4), dLdW0_solution)
    print("Test dLdW0: ", TEST_mlp4_dLdW0)

    TEST_mlp4_dLdb0 = np.allclose(dLdb0.round(4), dLdb0_solution)
    print("Test dLdb0: ", TEST_mlp4_dLdb0)

    TEST_mlp4_dA2dZ1 = np.allclose(dA2dZ1.round(4), dA2dZ1_solution)
    print("\nTest dA2dZ1:", TEST_mlp4_dA2dZ1)

    TEST_mlp4_dLdZ1 = np.allclose(dLdZ1.round(4), dLdZ1_solution)
    print("Test dLdZ1: ", TEST_mlp4_dLdZ1)

    TEST_mlp4_dLdA1 = np.allclose(dLdA1.round(4), dLdA1_solution)
    print("Test dLdA1: ", TEST_mlp4_dLdA1)

    TEST_mlp4_dLdW1 = np.allclose(dLdW1.round(4), dLdW1_solution)
    print("Test dLdW1: ", TEST_mlp4_dLdW1)

    TEST_mlp4_dLdb1 = np.allclose(dLdb1.round(4), dLdb1_solution)
    print("Test dLdb1: ", TEST_mlp4_dLdb1)

    TEST_mlp4_dA3dZ2 = np.allclose(dA3dZ2.round(4), dA3dZ2_solution)
    print("\nTest dA3dZ2:", TEST_mlp4_dA3dZ2)

    TEST_mlp4_dLdZ2 = np.allclose(dLdZ2.round(4), dLdZ2_solution)
    print("Test dLdZ2: ", TEST_mlp4_dLdZ2)

    TEST_mlp4_dLdA2 = np.allclose(dLdA2.round(4), dLdA2_solution)
    print("Test dLdA2: ", TEST_mlp4_dLdA2)

    TEST_mlp4_dLdW2 = np.allclose(dLdW2.round(4), dLdW2_solution)
    print("Test dLdW2: ", TEST_mlp4_dLdW2)

    TEST_mlp4_dLdb2 = np.allclose(dLdb2.round(4), dLdb2_solution)
    print("Test dLdb2: ", TEST_mlp4_dLdb2)

    TEST_mlp4_dA4dZ3 = np.allclose(dA4dZ3.round(4), dA4dZ3_solution)
    print("\nTest dA4dZ3:", TEST_mlp4_dA4dZ3)

    TEST_mlp4_dLdZ3 = np.allclose(dLdZ3.round(4), dLdZ3_solution)
    print("Test dLdZ3: ", TEST_mlp4_dLdZ3)

    TEST_mlp4_dLdA3 = np.allclose(dLdA3.round(4), dLdA3_solution)
    print("Test dLdA3: ", TEST_mlp4_dLdA3)

    TEST_mlp4_dLdW3 = np.allclose(dLdW3.round(4), dLdW3_solution)
    print("Test dLdW3: ", TEST_mlp4_dLdW3)

    TEST_mlp4_dLdb3 = np.allclose(dLdb3.round(4), dLdb3_solution)
    print("Test dLdb3: ", TEST_mlp4_dLdb3)

    TEST_mlp4_dA5dZ4 = np.allclose(dA5dZ4.round(4), dA5dZ4_solution)
    print("\nTest dA5dZ4:", TEST_mlp4_dA5dZ4)

    TEST_mlp4_dLdZ4 = np.allclose(dLdZ4.round(4), dLdZ4_solution)
    print("Test dLdZ4: ", TEST_mlp4_dLdZ4)

    TEST_mlp4_dLdA4 = np.allclose(dLdA4.round(4), dLdA4_solution)
    print("Test dLdA4: ", TEST_mlp4_dLdA4)

    TEST_mlp4_dLdW4 = np.allclose(dLdW4.round(4), dLdW4_solution)
    print("Test dLdW4: ", TEST_mlp4_dLdW4)

    TEST_mlp4_dLdb4 = np.allclose(dLdb4.round(4), dLdb4_solution)
    print("Test dLdb4: ", TEST_mlp4_dLdb4)

else:

    TEST_mlp4_dA1dZ0 = False
    TEST_mlp4_dLdZ0  = False
    TEST_mlp4_dLdA0  = False
    TEST_mlp4_dLdW0  = False
    TEST_mlp4_dLdb0  = False
    TEST_mlp4_dA2dZ1 = False
    TEST_mlp4_dLdZ1  = False
    TEST_mlp4_dLdA1  = False
    TEST_mlp4_dLdW1  = False
    TEST_mlp4_dLdb1  = False
    TEST_mlp4_dA3dZ2 = False
    TEST_mlp4_dLdZ2  = False
    TEST_mlp4_dLdA2  = False
    TEST_mlp4_dLdW2  = False
    TEST_mlp4_dLdb2  = False
    TEST_mlp4_dA4dZ3 = False
    TEST_mlp4_dLdZ3  = False
    TEST_mlp4_dLdA3  = False
    TEST_mlp4_dLdW3  = False
    TEST_mlp4_dLdb3  = False
    TEST_mlp4_dA5dZ4 = False
    TEST_mlp4_dLdZ4  = False
    TEST_mlp4_dLdA4  = False
    TEST_mlp4_dLdW4  = False
    TEST_mlp4_dLdb4  = False


"""
────────────────────────────────────────────────────────────────────────────────────
## BATCH NORMALIZATION
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_BATCHNORM = DEBUG_AND_GRADE_BATCHNORM_flag

if DEBUG_AND_GRADE_BATCHNORM:

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Eval) | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [ 1., 4.],
        [ 7., 0.],
        [ 1., 0.],
        [ 7., 4.]], dtype="f")

    BW = np.array([
        [ 2., 5.]], dtype="f")

    Bb = np.array([
        [-1., 2.]], dtype="f")

    bn = BatchNorm1d(2)
    bn.BW = BW
    bn.Bb = Bb

    BZ = bn.forward(A, eval=True)
    print("\n(eval) BZ =\n", BZ, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Eval) | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    BZ_solution = np.array([
        [ 1., 22.],
        [13.,  2.],
        [ 1.,  2.],
        [13., 22.]], dtype="f")
    
    print("\n(eval) BZ =\n", BZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Eval) | TEST RESULTS")
    print("──────────────────────────────────────────")

    TEST_batchnorm_eval_BZ = np.allclose(BZ.round(4), BZ_solution)
    print("\nTest (eval) BZ: ", TEST_batchnorm_eval_BZ, "\n", sep="")

else:
    
    print("\n             Pass?")

    TEST_batchnorm_eval_BZ = False

if DEBUG_AND_GRADE_BATCHNORM:

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Train) | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    bn = BatchNorm1d(2)
    bn.BW = BW
    bn.Bb = Bb

    BZ = bn.forward(A, eval=False)
    print("\n(train) BZ =\n", BZ, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Train) | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    BZ_solution = np.array([
        [-3.,  7.],
        [ 1., -3.],
        [-3., -3.],
        [ 1.,  7.]])

    print("\n(train) BZ =\n", BZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Train) | TEST RESULTS")
    print("──────────────────────────────────────────")

    TEST_batchnorm_train_BZ = np.allclose(BZ.round(4), BZ_solution)
    print("\nTest (train) BZ: ", TEST_batchnorm_train_BZ, "\n", sep="")

else:
    
    print("\n             Pass?")

    TEST_batchnorm_train_BZ = False

if DEBUG_AND_GRADE_BATCHNORM:

    print("\n──────────────────────────────────────────")
    print("BATCHNORM BACKWARD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    dLdA = np.array([
        [ -6. ,  2.],
        [ -12., 16.],
        [ -12., 20.],
        [ -6. ,  2.]], dtype="f")

    dLdZ = bn.backward(dLdA)
    print("\ndLdZ =\n", dLdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM BACKWARD | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    dLdZ_solution = np.array([
        [ 2.,  0.],
        [-2., -5.],
        [-2.,  5.],
        [ 2.,  0.]], dtype="f")

    print("\ndLdZ =\n", dLdZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM BACKWARD | TEST RESULTS")
    print("──────────────────────────────────────────")

    TEST_batchnorm_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution)
    print("\nTest dLdZ: ", TEST_batchnorm_dLdZ, "\n", sep="")

else:
    
    print("\n             Pass?")

    TEST_batchnorm_dLdZ = False


"""
────────────────────────────────────────────────────────────────────────────────────
## SCORE AND GRADE TESTS
────────────────────────────────────────────────────────────────────────────────────
"""

TEST_activations = (
    TEST_identity_A    and
    TEST_identity_dAdZ and
    TEST_sigmoid_A     and
    TEST_sigmoid_dAdZ  and
    TEST_tanh_A        and
    TEST_tanh_dAdZ     and
    TEST_relu_A        and
    TEST_relu_dAdZ)

TEST_loss =(
    TEST_mseloss_L             and
    TEST_mseloss_dLdA          and
    TEST_crossentropyloss_L    and
    TEST_crossentropyloss_dLdA )

TEST_linear = (
    TEST_linear_Z    and
    TEST_linear_dZdA and
    TEST_linear_dZdW and
    TEST_linear_dZdi and
    TEST_linear_dZdb and
    TEST_linear_dLdA and
    TEST_linear_dLdi)

TEST_sgd = (
    TEST_sgd_W_1 and
    TEST_sgd_b_1 and
    TEST_sgd_W_2 and
    TEST_sgd_b_2)

TEST_mlp0 = (
    TEST_mlp0_Z0     and
    TEST_mlp0_A1     and
    TEST_mlp0_dA1dZ0 and
    TEST_mlp0_dLdZ0  and
    TEST_mlp0_dLdA0  and
    TEST_mlp0_dLdW0  and
    TEST_mlp0_dLdb0 )

TEST_mlp1 = (
    TEST_mlp0_Z0     and
    TEST_mlp0_A1     and
    TEST_mlp0_dA1dZ0 and
    TEST_mlp0_dLdZ0  and
    TEST_mlp0_dLdA0  and
    TEST_mlp0_dLdW0  and
    TEST_mlp0_dLdb0  and
    TEST_mlp1_Z0     and
    TEST_mlp1_A1     and
    TEST_mlp1_Z1     and
    TEST_mlp1_A2     and
    TEST_mlp1_dA2dZ1 and
    TEST_mlp1_dLdZ1  and
    TEST_mlp1_dLdA1  and
    TEST_mlp1_dA1dZ0 and
    TEST_mlp1_dLdZ0  and
    TEST_mlp1_dLdA0  and
    TEST_mlp1_dLdW0  and
    TEST_mlp1_dLdb0 )

TEST_mlp4 = (
    TEST_mlp4_Z0 and
    TEST_mlp4_A1 and
    TEST_mlp4_Z1 and
    TEST_mlp4_A2 and
    TEST_mlp4_Z2 and
    TEST_mlp4_A3 and
    TEST_mlp4_Z3 and
    TEST_mlp4_A4 and
    TEST_mlp4_Z4 and
    TEST_mlp4_A5 and
    TEST_mlp4_dA1dZ0 and
    TEST_mlp4_dLdZ0  and
    TEST_mlp4_dLdA0  and
    TEST_mlp4_dLdW0  and
    TEST_mlp4_dLdb0  and
    TEST_mlp4_dA2dZ1 and
    TEST_mlp4_dLdZ1  and
    TEST_mlp4_dLdA1  and
    TEST_mlp4_dLdW1  and
    TEST_mlp4_dLdb1  and
    TEST_mlp4_dA3dZ2 and
    TEST_mlp4_dLdZ2  and
    TEST_mlp4_dLdA2  and
    TEST_mlp4_dLdW2  and
    TEST_mlp4_dLdb2  and
    TEST_mlp4_dA4dZ3 and
    TEST_mlp4_dLdZ3  and
    TEST_mlp4_dLdA3  and
    TEST_mlp4_dLdW3  and
    TEST_mlp4_dLdb3  and
    TEST_mlp4_dA5dZ4 and
    TEST_mlp4_dLdZ4  and
    TEST_mlp4_dLdA4  and
    TEST_mlp4_dLdW4  and
    TEST_mlp4_dLdb4 )

TEST_batchnorm = (
    TEST_batchnorm_eval_BZ and 
    TEST_batchnorm_train_BZ and
    TEST_batchnorm_dLdZ )

SCORE_LOGS = {
    "Activation"   : 5  * int(TEST_activations),
    "Loss"         : 5  * int(TEST_loss),
    "Linear Layer" : 15 * int(TEST_linear),
    "SGD"          : 10 * int(TEST_sgd),
    "MLP0"         : 10 * int(TEST_mlp0),
    "MLP1"         : 15 * int(TEST_mlp1),
    "MLP4"         : 20 * int(TEST_mlp4),
    "Batch Norm"   : 20 * int(TEST_batchnorm)
}


print("\n")
print("TEST   | STATUS | POINTS | DESCRIPTION")
print("───────┼────────┼────────┼────────────────────────────────")

for i, (key, value) in enumerate(SCORE_LOGS.items()):
    
    index_str = str(i).zfill(1)
    point_str = str(value).zfill(2) + "     │ "
    
    if value == 0:
        status_str = " │ FAILED │ " 
    else:
        status_str = " │ PASSED │ "
        
    
    print("Test ", index_str, status_str, point_str, key,sep="")
    
print("\n")

"""
────────────────────────────────────────────────────────────────────────────────────
## FINAL AUTOLAB SCORES
────────────────────────────────────────────────────────────────────────────────────
"""

print(json.dumps({'scores': SCORE_LOGS}))