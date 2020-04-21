############Dúvidas###################
#### a matriz da solucao aproximada uik_array tem dimensao (M + 1, N+1), pois a barra segundo o enunciado vai de 0 a N, se ela tivesse só dimensao
# (M, N) a matriz ia ir de 0 a N - 1. Esta certo esse pensamento?
### Raciocinio analago a dimensao M é aplicada pois a linha zero é o tempo inicial e a gente quer no momento M, logo M + 1 dimensao
# portanto o laco deve ir ate M - 1 pois ai descobrimos o tempo M, certo?
#


import math
import numpy as np
import os
import matplotlib as plot
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

#funcoes de condicao de contor, g sao as fronteira ui é a inicial
def g0():
    return 0
def gn():
    return 0
def ui():
    return 0

def u (x, t):
    #f =
    f = 0
    return f


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
    uik_array = np.zeros((M + 1, N + 1))    #uik aproximado
    true_uik_array = np.zeros((M + 1, N + 1)) #uik real
    #matriz de erros
    eik_array = np.zeros((M + 1, N + 1))
    #matriz de truncamento
    tik_array = np.zeros((M + 1, N + 1))
#Talvez possamos eliminar uma linha e 2 colunas de cada uma dos arrays de erro e truncamento, mas resolvi manter para ser diretamente endereçados a matriz de resultado aproximado


    #metodo 11
    i = 0 #iterator for space
    k = 0 #iterator for time

    #para condincao inicial

###########
##Calculo aproximada
########
    #condicoes de fronteiras
    for k in range(1, M + 1):
        uik_array[k][0] = g0()
        uik_array[k][N] = gn()
    #condicao inicial
    for i in range(N + 1):
        uik_array[0][i] = ui()
    #laco para calcular os elementos depois de dada as condicoes inciais
    for k in range(M):
        for i in range(1, N):
            uik_array[k + 1][i] = uik_array[k][i] + Dt((uik_array[k][i - 1] - 2*uik_array[k][i] + uik_array[k][i + 1])/(Dx*Dx) + f(Dx*i, Dt*k))

##########
##Calculo exato
##########
    for k in range(M + 1):
        for i in range(N + 1):
            true_uik_array[k][i] = u(Dx*i, Dt*k)

#########
##Calculo dos erros e truncamentos
###########

    #calculo do truncamento
    for k in range(M):
        for i in range(1, N):
            tik_array[k][i] = ((true_uik_array[k+1][i] - true_uik_array[k][i])/Dt) -((true_uik_array[k][i - 1] - 2*true_uik_array[k][i] + true_uik_array[k][i + 1])/(Dx**2)) - (f(Dx*i, Dt*k))

    #Calculo do erro
    for k in range(M):
        for i in range(1, N):
            eik_array[k + 1][i] = eik_array[k][i] + Dt*((eik_array[k][i - 1] - 2*eik_array[k][i] + eik_array[k][i + 1])/(Dx*Dx) + tik_array[k][i] )

    #erro normalizado
    enorm = math.abs(np.amax(eik_array, axis = 0))
































if __name__ == '__main__':
    main()
