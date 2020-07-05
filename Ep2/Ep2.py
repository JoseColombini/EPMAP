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
def f (x,t, Dx, pk):
    f = r(t)*ak*gh(x, pk, Dx)
    return f


#funcao r(t) do item c
@jit(nopython=True)
def r(t):
    r = 10*(1 + math.cos(5*t))
    return r

#funcao gh(x) do item c
@jit(nopython=True)
def gh(x, p, h):
    if p-(h/2) <= x and x <= p+(h/2):
        g = (1/h)#*(1 - abs(x - p)*1/(h/2))
        return g
    else:
        return 0


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
def crankNicolson(A_P_arrary, A_S_arrary, L_arrary, D_arrary, uik_array, N, M, Dt, Dx, lambd, pk):

    b_array = np.zeros((N - 1), dtype = np.float64)
    X3 = np.zeros((N - 1), dtype = np.float64)



    for k in range(0, M):
        for i in range(N - 1):
            if i == 0:
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), Dx, pk) + f(Dx*(i + 1), Dt*k, Dx, pk)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2]) + (lambd/2)*uik_array[k + 1][0]
            elif i == (N - 2):
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), Dx, pk) + f(Dx*(i + 1), Dt*k, Dx, pk)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2]) + (lambd/2)*uik_array[k + 1][N]
            else:
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), Dx, pk) + f(Dx*(i + 1), Dt*k, Dx, pk)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2])

        Solving_LD(L_arrary, D_arrary, b_array, N, X3)
        for t in range(N - 1):
            uik_array[k + 1][t + 1] = X3[t]


def normalSystemArrange (p_uik_array, T_uik_array, NS_array, RS_array):
    for i in range(len(RS_array)):
        for t in range(i, len(RS_array)):
            NS_array[i][t] = np.dot(p_uik_array[i], p_uik_array[t])
            NS_array[t][i] = NS_array[i][t]

            RS_array[i] = np.dot(T_uik_array, p_uik_array[i])



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
            nf = int(input("Insert the number of fonts (nf): "))
            pk_array = np.zeros(nf, dtype = np.float64)
            ak_array = np.zeros(nf, dtype = np.float64)

            for i in range(nf):
                pk_array = int(input("Insert p"+ str(i)+" position: "))
            for i in range(nf):
                ak_array = int(input("Insert a"+ str(i)+" weight: "))




            Dt = T/M
            lambd = Dt/(Dx*Dx)
            print("N = ", N)
            print("M = ", M)
            print("Dx = ", Dx)
            print("Dt = ", Dt)
            print("lambda = ", lambd)
            print("nf = ", nf)
            print("pk = ", pk_array)
            #return 0
            #Creat the matix uik that describ all bar in datetime
            #Each line is a bar in one moment
            #So each colum is a position in the bar
            #uik_array(2, 5) is the point 5 of the bar at the moment 2
            uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)  #euler uik
            T_uik_array = np.zeros((1, N + 1), dtype = np.float64) #uik real
            p_uik_array = np.zeros((nf - 1, N + 1), dtype = np.float64)
            #matriz de erros
            eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)

            A_P_arrary = np.zeros((N - 1), dtype = np.float64)
            A_S_arrary = np.zeros((N - 1), dtype = np.float64)
            D_arrary = np.zeros((N - 1), dtype = np.float64)
            L_arrary = np.zeros((N - 1), dtype = np.float64)

            NS_array = np.zeros((nf, nf), dtype = np.float64)
            RS_array = np.zero((nf), dtype = np.float64)



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


            #building A_arrary
            for i in range(nf):
                crankNicolson(A_P_arrary, A_S_arrary, L_arrary, D_arrary, uik_array, N, M, Dt, Dx, lambd, pk_array[i])
                T_uik_array += ak_array[i]*uik_array[M]
                p_uik_array[i] = uik_array[M]


            normalSystemArrange(p_uik_array, T_uik_array, NS_array, RS_array)



            # yaxis = np.arange(N+1)

            #tudo sobre euler
                #figura 1 temoperatura
            # plt.figure(1, figsize = (20, 15))
            #
            # plt.plot(yaxis, c_uik_array[0], 'b--', label = 't = 0.0')
            # plt.plot(yaxis, c_uik_array[int(M/10)], 'g--', label = 't = 0.1')
            # plt.plot(yaxis, c_uik_array[int(2*M/10)], 'r--', label = 't = 0.2')
            # plt.plot(yaxis, c_uik_array[int(3*M/10)], 'c--', label = 't = 0.3')
            # plt.plot(yaxis, c_uik_array[int(4*M/10)], 'm--', label = 't = 0.4')
            # plt.plot(yaxis, c_uik_array[int(5*M/10)], 'y--', label = 't = 0.5')
            # plt.plot(yaxis, c_uik_array[int(6*M/10)], 'b:', label = 't = 0.6')
            # plt.plot(yaxis, c_uik_array[int(7*M/10)], 'g:', label = 't = 0.7')
            # plt.plot(yaxis, c_uik_array[int(8*M/10)], 'r:', label = 't = 0.8')
            # plt.plot(yaxis, c_uik_array[int(9*M/10)], 'c:', label = 't = 0.9')
            # plt.plot(yaxis, c_uik_array[M], 'r-', label = 't = 1')
            #
            # plt.plot(yaxis, true_uik_array[M], 'k-', label = 'exato')
            # plt.legend()
            # plt.xlabel('Posição na barra')
            # plt.ylabel('temperatura')
            #
            # plt.show()



            print(temp - time.time())

































if __name__ == '__main__':
    main()
