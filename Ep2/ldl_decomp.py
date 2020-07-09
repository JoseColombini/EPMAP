import numpy as np
import math
import pprint

def ldl_decomp(n, A_array, L_array, D_array):
	#loop inicial de construção das matrizes L e D
	v = []
	for i in range(0, n):
		v.append(0)
		for j in range(0, n):
			D_array[i][i] = A_array[i][i]
			if j != i:
				D_array[i][j] = 0
			L_array[i][j] = A_array[i][j]/D_array[i][i]
	for i in range(0, n):
		soma = 0
		for j in range(0, i):
			v[j] = L_array[i][j] * D_array[j][j]
			soma = soma + L_array[i][j] * v[j]
		D_array[i][i] = A_array[i][i] - soma
		for j in range(0, n):
			soma1 = 0
			for k in range (0, i):
				soma1 = soma1 + L_array[j][k] * v[k]
			L_array[j][i] = (A_array[j][i] - soma1)/D_array[i][i]


A_array = [[4, 1, 1, 1], [1, 3, -1, 1], [1, -1, 2, 0], [1, 1, 0, 2]]
L_array = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
D_array = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

ldl_decomp(4, A_array, L_array, D_array)

pprint.pprint(L_array)
pprint.pprint(D_array)