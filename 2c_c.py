############Dúvidas###################
#### a matriz da solucao aproximada uik_array tem dimensao (M + 1, N+1), pois a barra segundo o enunciado vai de 0 a N, se ela tivesse só dimensao
# (M, N) a matriz ia ir de 0 a N - 1. Esta certo esse pensamento?
### Raciocinio analago a dimensao M é aplicada pois a linha zero é o tempo inicial e a gente quer no momento M, logo M + 1 dimensao
# portanto o laco deve ir ate M - 1 pois ai descobrimos o tempo M, certo?
#


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

##funcao qeu insere calor
@jit(nopython=True)
def f (x,t, Dx):
    f = r(t) * gh(x, 0.25, Dx)
    return f


#funcao r(t) do item c
@jit(nopython=True)
def r(t):
    r = 10000 * (1-2*(t**2))
    return r

#funcao gh(x) do item c
@jit(nopython=True)
def gh(x, p, h):
    if p-(h/2) <= x and x <= p+(h/2):
        g = (1/h)#*(1 - abs(x - p)*1/(h/2))
        return g
    else:
        return 0


#funcoes de condicao de contor, g sao as fronteira ui é a inicial
# @jit(nopython=True)
# def g0():
#     return 0
# @jit(nopython=True)
# def gn():
#     return 0
# @jit(nopython=True)
# def ui(x):
#     u = x*x*(1 - x)**2
#     return u



#exercicio 2
@jit(nopython=True)
def Decomp_LD(A_P_arrary, A_S_arrary, L_arrary, D_arrary, N):
    D_arrary[0] = A_P_arrary[0]
    for i in range(1, N - 1):
        L_arrary[i] = A_S_arrary[i]/D_arrary[i - 1]
        D_arrary[i] = A_P_arrary[i] - L_arrary[i]**2*D_arrary[i - 1]


@jit(nopython=True)
def Solving_LD(L_arrary, D_arrary, b_array, N, X3):

    X1 = np.zeros((N - 1), dtype = np.float64)
    X2 = np.zeros((N - 1), dtype = np.float64)

    X1[0] = b_array[0]
    for i in range(1, N - 1):
        X1[i] = b_array[i] - L_arrary[i]*X1[i - 1]


    for i in range(N - 1):
        X2[i] = X1[i]/D_arrary[i]


    X3[N - 2] = X2[N - 2]
    for i in range(N - 3, -1, -1):
        X3[i] = X2[i] - L_arrary[i + 1]*X3[i + 1]


@jit(nopython=True)
def crankNicolson(A_P_arrary, A_S_arrary, L_arrary, D_arrary, uik_array, N, M, Dt, Dx, lambd):

    b_array = np.zeros((N - 1), dtype = np.float64)
    X3 = np.zeros((N - 1), dtype = np.float64)

    for i in range(N - 1):
        A_P_arrary[i] = 1 + lambd
    for i in range(1, N - 1):
        A_S_arrary[i] = -lambd/2


    #COndicoes inicias
    for i in range(N + 1):
        uik_array[0][i] = 0
    #condicao de contorno
    for k in range(1, M + 1):
        uik_array[k][0] = 0
        uik_array[k][N] = 0

    #resolvendo euler implicito
    Decomp_LD(A_P_arrary, A_S_arrary, L_arrary, D_arrary, N)

    for k in range(0, M):
        for i in range(N - 1):
            if i == 0:
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), Dx) + f(Dx*(i + 1), Dt*k, Dx)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2]) + (lambd/2)*uik_array[k + 1][0]
            elif i == (N - 2):
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), Dx) + f(Dx*(i + 1), Dt*k, Dx)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2]) + (lambd/2)*uik_array[k + 1][N]
            else:
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), Dx) + f(Dx*(i + 1), Dt*k, Dx)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2])

        Solving_LD(L_arrary, D_arrary, b_array, N, X3)
        for t in range(N - 1):
            uik_array[k + 1][t + 1] = X3[t]




def main():
            temp = time.time()
            #Input of number of divisions
            N = int(input("Insert the number of x fraction (N): "))
            Dx = X/N
            #lambd = float(input("Insert lambda: "))
            #Dt = lambd*Dx*Dx
            #M = int(round(T/Dt))
            M = int(input("Insert the number of t fraction (M): "))
            #Size of fractions


            Dt = T/M
            lambd = Dt/(Dx*Dx)
            print("N = ", N)
            print("M = ", M)
            print("Dx = ", Dx)
            print("Dt = ", Dt)
            print("lambda = ", lambd)
            #return 0
            #Creat the matix uik that describ all bar in datetime
            #Each line is a bar in one moment
            #So each colum is a position in the bar
            #uik_array(2, 5) is the point 5 of the bar at the moment 2
            c_uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)  #euler uik
            true_uik_array = np.zeros((M + 1, N + 1), dtype = np.float64) #uik real
            #matriz de erros
            eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)

            A_P_arrary = np.zeros((N - 1), dtype = np.float64)
            A_S_arrary = np.zeros((N - 1), dtype = np.float64)
            D_arrary = np.zeros((N - 1), dtype = np.float64)
            L_arrary = np.zeros((N - 1), dtype = np.float64)

            #building A_arrary
            crankNicolson(A_P_arrary, A_S_arrary, L_arrary, D_arrary, c_uik_array, N, M, Dt, Dx, lambd)


            yaxis = np.arange(N+1)

            #tudo sobre euler
                #figura 1 temoperatura
            plt.figure(1, figsize = (20, 15))

            plt.plot(yaxis, c_uik_array[0], 'b--', label = 't = 0.0')
            plt.plot(yaxis, c_uik_array[int(M/10)], 'g--', label = 't = 0.1')
            plt.plot(yaxis, c_uik_array[int(2*M/10)], 'r--', label = 't = 0.2')
            plt.plot(yaxis, c_uik_array[int(3*M/10)], 'c--', label = 't = 0.3')
            plt.plot(yaxis, c_uik_array[int(4*M/10)], 'm--', label = 't = 0.4')
            plt.plot(yaxis, c_uik_array[int(5*M/10)], 'y--', label = 't = 0.5')
            plt.plot(yaxis, c_uik_array[int(6*M/10)], 'b:', label = 't = 0.6')
            plt.plot(yaxis, c_uik_array[int(7*M/10)], 'g:', label = 't = 0.7')
            plt.plot(yaxis, c_uik_array[int(8*M/10)], 'r:', label = 't = 0.8')
            plt.plot(yaxis, c_uik_array[int(9*M/10)], 'c:', label = 't = 0.9')
            plt.plot(yaxis, c_uik_array[M], 'r-', label = 't = 1')

            plt.plot(yaxis, true_uik_array[M], 'k-', label = 'exato')
            plt.legend()
            plt.xlabel('Posição na barra')
            plt.ylabel('temperatura')

            plt.show()



            print(temp - time.time())

































if __name__ == '__main__':
    main()
