import numpy as np
import math

def ldl_decomp(n, A_array, L_array, D_array):
	#loop inicial de construção das matrizes L e D
	for i in A_array:
		D_array[i][i] = A_array[i][i]
		for j in A_array[i]:
			D_array[i][j] = 0
			L_array[i][j] = A_array[i][j]/D_array[i][j]
	for i in range(0, n-1):
		for j in range(0, i-2):
			v[j] = L_array[i][j] * D_array[i][j]
			soma = soma + L_array[i][j] * v[j]
		D_array[i][i] = A_array[i][i] - soma
		for j in range(i, n-1):
			for k in range (0, i-2):
				soma1 = soma1 + L_array[j][k] * v[k]
			L_array[j][i] = (A_array[j][i] - soma1)/D_array[i][i]


