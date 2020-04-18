############Dúvidas###################
#### a matriz da solucao aproximada uik_array tem dimensao (M + 1, N+1), pois a barra segundo o enunciado vai de 0 a N, se ela tivesse só dimensao
# (M, N) a matriz ia ir de 0 a N - 1. Esta certo esse pensamento?
### Raciocinio analago a dimensao M é aplicada pois a linha zero é o tempo inicial e a gente quer no momento M, logo M + 1 dimensao
# portanto o laco deve ir ate M - 1 pois ai descobrimos o tempo M, certo?
#


import math
import numpy as np
import os
import matplotlib as matplot
import time
import datetime
import os
import sys

T = 1 #Ex 1
X = 1

#funcao qeu insere calor
def f (x,t):
    #f = 10*x*x*(x - 1) - 60*x*t + 20*t

    return f

def g0():
    return 0
def gn():
    return 0
def ui():
    return 0


def main():
    #Input of number of divisions
    N = int(input("Insert the number of x fraction (N): "))
    M = int(input("Insert the number of t fraction (M): "))
    #Size of fractions

    Dx = X/N
    Dt = T/M
    lamb = Dt/(Dx*Dx)
    print("Dx = ", N)
    print("Dt = ", M)
    print("lambda = ", lamb)

    #Creat the matix uik that describ all bar in datetime
    #Each line is a bar in one moment
    #So each colum is a position in the bar
    #uik_array(2, 5) is the point 5 of the bar at the moment 2
    uik_array = np.zeros((M + 1, N + 1))
    #matriz de erros
    eik_array = np.zeros((M + 1, N + 1))
    #matriz de truncamento
    tik_array = np.zeros((M + 1, N + 1))
#Talvez possamos eliminar uma linha e 2 colunas de cada uma dos arrays de erro e truncamento, mas resolvi manter para ser diretamente endereçados a matriz de resultado aproximado


    #metodo 11
    i = 1 #iterator for space
    k = 0 #iterator for time

    #para condincao inicial


    #condicoes de fronteiras
    for k in range(1, M):
        uik_array[k][0] = g0()
        uik_array[k][N] = gn()
    #condicao inicial
    for i in range(N):
        uik_array[0][i] = ui()
    #laco para calcular os elementos depois de dada as condicoes inciais
    for k in range(M - 1):
        for i in range(1, N - 1):
            uik_array[k + 1][i] = uik_array[k][i] + Dt((uik_array[k][i - 1] - 2*uik_array[k][i] + uik_array[k][i + 1])/(Dx*Dx) + f(Dx*i, Dt*k))

    #Calculo do erro
    for k in range(M - 1):
        for i in range(1, N - 1):
            eik_array[k + 1][i] = eik_array[k][i] + Dt((eik_array[k][i - 1] - 2*uik_array[k][i] + uik_array[k][i + 1])/(Dx*Dx) + tik_array[k][i] )

    #erro normalizado
    enorm = math.abs(np.amax(eik_array, axis = 0))






























if __name__ == '__main__':
    main()
