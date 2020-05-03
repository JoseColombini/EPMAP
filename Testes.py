


import math
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime
import os
import sys
import pathlib
from numba import jit


maypath = pathlib.Path(__file__).parent.absolute()

T = 1 #Ex 1
X = 1

lambd = 0.5
N = 5
#N = int(input("Insert the number of x fraction (N): "))
Dx = X/N
#lambd = float(input("Insert lambda: "))
Dt = lambd*Dx*Dx
M = int(round(T/Dt))
#    M = int(input("Insert the number of t fraction (M): "))
#Size of fractions


Dt = T/M
#    lambd = Dt/(Dx*Dx)
print("N = ", N)
print("M = ", M)
print("Dx = ", Dx)
print("Dt = ", Dt)
print("lambda = ", lambd)

#uik_array(2, 5) is the point 5 of the bar at the moment 2
uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)    #uik aproximado
true_uik_array = np.zeros((M + 1, N + 1), dtype = np.float64) #uik real
#matriz de erros
eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)
#matriz de truncamento
tik_array = np.zeros((M + 1, N + 1), dtype = np.float64)
#Talvez possamos eliminar uma linha e 2 colunas de cada uma dos arrays de erro e truncamento, mas resolvi manter para ser diretamente endereçados a matriz de resultado aproximado

A_P_arrary = np.zeros((N - 1), dtype = np.float64)
A_S_arrary = np.zeros((N - 1), dtype = np.float64)
D_arrary = np.zeros((N - 1), dtype = np.float64)
L_arrary = np.zeros((N - 1), dtype = np.float64)


for i in range(N - 1):
    A_P_arrary[i] = 1 + 2*lambd
for i in range(1, N - 1):
    A_S_arrary[i] = -lambd



D_arrary[0] = A_P_arrary[0]
for i in range(1, N - 1):
    L_arrary[i] = A_S_arrary[i]/D_arrary[i - 1]
    D_arrary[i] = A_P_arrary[i] - L_arrary[i]**2*D_arrary[i - 1]



X1 = np.zeros((N - 1), dtype = np.float64)
X2 = np.zeros((N - 1), dtype = np.float64)
X3 = np.zeros((N - 1), dtype = np.float64)
b_array = np.ones((N - 1), dtype = np.float64)

X1[0] = b_array[0]
for i in range(1, N - 1):
    X1[i] = b_array[i] - L_arrary[i]*X1[i - 1]

for i in range(N - 1):
    X2[i] = X1[i]/D_arrary[i]

X3[N - 2] = X2[N - 2]
for i in range(N - 3, -1, -1):
    X3[i] = X2[i] - L_arrary[i + 1]*X3[i + 1]



print('P: ',  A_P_arrary)
print(len(A_P_arrary))
print('S: ',  A_S_arrary)
print(len(A_S_arrary))
print('L: ',  L_arrary)
print(len(L_arrary))
print('D: ',  D_arrary)
print(len(D_arrary))
print('b: ',  b_array)
print(len(b_array))
print('x1: ',  X1)
print(len(X1))
print('x2: ',  X2)
print(len(X2))
print('x3: ',  X3)
print(len(X3))
