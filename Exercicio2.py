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

#funcao qeu insere calor
@jit(nopython=True)
def f (x,t):
    #f = (math.exp(t - x)*(math.cos(5*t*x) - 5*x*math.sin(5*t*x))) - (math.exp(t - x)*((1 - 25*t**2)*math.cos(5*t*x) + 10*t*math.sin(5*t*x)))       #b
    f = 10*math.cos(10*t)*x*x*(1 - x)**2 - (1 + math.sin(10*t))*(12*x*x - 12*x + 2)        #a
    #f = 10*x*x*(x - 1) - 60*x*t + 20*t                                                     #extra
    #f = 0
    return f

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

@jit(nopython=True)
def u (x, t):
    #u = math.exp(t - x)*math.cos(5*t*x)                                         #b
    u = (1 + math.sin(10*t))*x*x*(1 - x)**2                                    #a
    #u = 10*t*x*x*(x - 1)
    #u = 0
    return u



@jit(nopython=True)
def resolution_a(N, M, uik_array, true_uik_array, eik_array, tik_array, Dx, Dt):
    #metodo 11
    #fix = 0. #inserção de calor
    i = 0 #iterator for space
    k = 0 #iterator for time

    #para condincao inicial

###########
##Calculo aproximada
########
    #condicoes de fronteiras
    for k in range(1, M + 1):
        uik_array[k][0] = u(0, k*Dt)
        uik_array[k][N] = u(N*Dx, k*Dt)
    #condicao inicial
    for i in range(N + 1):
        uik_array[0][i] = u(Dx*i, 0)
    #laco para calcular os elementos depois de dada as condicoes inciais
    for k in range(M):
        for i in range(1, N):
            #fix = f(Dx*i, Dt*k)
            uik_array[k + 1][i] = uik_array[k][i] + Dt*((uik_array[k][i - 1] - 2*uik_array[k][i] + uik_array[k][i + 1])/(Dx*Dx) + f(Dx*i, Dt*k))

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



#exercicio 2
def Decomp_LD(A_P_arrary, A_S_arrary, L_arrary, D_arrary, N):
    D_arrary[0] = A_P_arrary[0]
    for i in range(1, N - 1):
        L_arrary[i] = A_S_arrary[i]/D_arrary[i - 1]
        D_arrary[i] = A_P_arrary[i] - L_arrary[i]**2*D_arrary[i]

def Solving_LD(L_arrary, D_arrary, b_array, N):

    X1 = np.zeros((N - 1), dtype = np.float64)
    X2 = np.zeros((N - 1), dtype = np.float64)
    X3 = np.zeros((N - 1), dtype = np.float64)

    X1[0] = b_array[0]
    for i in range(1, N - 1):
        X1[i] = b_array[i] - L_arrary[i]*X1[i - 1]

    for i in range(N - 1):
        X2[i] = X1[i]/D_arrary[i]

    X3[N - 1] = X2[N - 1]
    for i in range(N - 2, -1, -1):
        X3[i] = X2[i] - L[i + 1]*X3[i + 1]

    return X3





def main():
    llist = [0.25, 0.5, 0.51]
    for lambd in llist:#(0.25, 0.51, 0.25):
        N = 10
        while N <= 320:
            temp = time.time()
            #Input of number of divisions
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
            #return 0
            #Creat the matix uik that describ all bar in datetime
            #Each line is a bar in one moment
            #So each colum is a position in the bar
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

            #building A_arrary
            for i in range(N - 1):
                A_P_arrary[i] = 1 + 2*lambd
            for i in range(1, N - 1):
                A_S_arrary[i] = -lambd

            Decomp_LD(A_P_arrary, A_S_arrary, L_arrary, D_arrary, N)


        #    Solving_LD(L_arrary, D_arraryr, uik_array[i])


            resolution_a(N, M, uik_array, true_uik_array, eik_array, tik_array, Dx, Dt)




            #erro normalizado
            enorm = np.zeros((M + 1, 1), dtype = np.float64)
            enorm = np.amax(np.absolute(eik_array), axis = 1) #normalizado esta estranho


            yaxis = np.arange(N+1)
            #figura 1
            plt.figure(1)
            #plot do estado final aproximado
            plt.subplot(131)
            plt.title('Temperatura')
            plt.plot(yaxis, uik_array[M], 'r', label = 'aproximado')
            plt.plot(yaxis, true_uik_array[M], 'b', label= 'exato')
            plt.legend()
            plt.xlabel('Posição na barra')
            plt.ylabel('temperatura')
            #plot do final exato
            # plt.subplot(122)
            # plt.title('Temperatura real')
            #
            # plt.xlabel('Posição na barra')
            # plt.ylabel('temperatura')

            #plot do erro ao longo da barra
            plt.subplot(132)
            plt.title('Erro ao longo da barra no instante T \n')
            plt.plot(yaxis, eik_array[M], 'b', label = 'erro')
            plt.plot(yaxis, tik_array[M - 1], 'r', label = 'truncamento')
            plt.xlabel('Posição na barra')
            plt.ylabel('erro')
            plt.legend()
            #plot do erro normalizado ao longo do tempo
            plt.subplot(133)
            plt.title('Erro normalizado ao longo do tempo \n')
            plt.plot(np.arange(M+1),enorm)
            plt.xlabel('instante')
            plt.ylabel('erro')
            plt.savefig(str(maypath) + '/Images/Grafs N = ' + str(N) + ', L = ' + str(lambd) + '.png')
            plt.close('all')
            #show pyplot
            plt.show()
            N = 2*N
            print(temp - time.time())

































if __name__ == '__main__':
    main()
