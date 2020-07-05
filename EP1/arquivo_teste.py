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
    r = 10*math.cos(10*t)*x*x*(1 - x)**2 - (1 + math.sin(10*t))*(12*x*x - 12*x + 2)
    #r = 10*x*x*(x - 1) - 60*x*t + 20*t
    #r = 0
    return r

#funcoes de condicao de contor, g sao as fronteira ui é a inicial
@jit(nopython=True)
def g0(t):
    g0 = u(0, t)
    #g0 = 0
    return 0
@jit(nopython=True)
def gn(t):
    gn = u(X, t)
    #gn = 0
    return gn
@jit(nopython=True)
def ui(x):

    #u = x*x*(1 - x)**2                         #a
    u = math.exp(-x)                           #b
    return u

@jit(nopython=True)
def u (x, t):
    f = math.exp(t - x)*math.cos(5*t*x)        #b
    #f = (1 + math.sin(10*t))*x*x*(1 - x)**2    #a
    #f = 10*t*x*x*(x - 1)                       #não é para usar
    #f = 0
    return f



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
        uik_array[k][0] = g0()
        uik_array[k][N] = gn()
    #condicao inicial
    for i in range(N + 1):
        uik_array[0][i] = ui(Dx*i)
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

            for i in range(N + 1):
                true_uik_array[0][i] = u(i*Dx,0)
            #condicoes de fronteiras
            for k in range(1, M + 1):
                uik_array[k][0] = g0(k*Dt)
                uik_array[k][N] = gn(k*Dt)
            #condicao inicial
            for i in range(N + 1):
                uik_array[0][i] = ui(Dx*i)
            plt.plot(np.arange(N + 1), true_uik_array[0], 'b', label='real')
            plt.plot(np.arange(N + 1), uik_array[0], 'r', label = 'aproximado')
            plt.legend()
            plt.show()
        #Talvez possamos eliminar uma linha e 2 colunas de cada uma dos arrays de erro e truncamento, mas resolvi manter para ser diretamente endereçados a matriz de resultado aproximado

            # resolution_a(N, M, uik_array, true_uik_array, eik_array, tik_array, Dx, Dt)
            #
            # #erro normalizado
            # enorm = np.zeros((M + 1, 1), dtype = np.float64)
            # enorm = np.absolute(np.amax(eik_array, axis = 1)) #normalizado esta estranho
            #
            #
            # yaxis = np.arange(N+1)
            # #figura 1
            # plt.figure(1)
            # #plot do estado final aproximado
            # plt.subplot(121)
            # plt.title('Temperatura')
            # plt.plot(yaxis, uik_array[M], 'r', label = 'aproximado')
            # plt.plot(yaxis, true_uik_array[M], 'b', label= 'exato')
            # plt.legend()
            # plt.xlabel('Posição na barra')
            # plt.ylabel('temperatura')
            # #plot do final exato
            # # plt.subplot(122)
            # # plt.title('Temperatura real')
            # #
            # # plt.xlabel('Posição na barra')
            # # plt.ylabel('temperatura')
            # plt.savefig(str(maypath) + '/Images/GN = ' + str(N) + ', L = ' + str(lambd) + '.png')
            #
            # #figura 2 (erro)
            # plt.figure(2)
            # #plot do erro ao longo da barra
            # plt.subplot(221)
            # plt.title('Erro ao longo da barra no instante T \n')
            # plt.plot(yaxis, eik_array[M], 'b', label = 'erro')
            # plt.plot(yaxis, tik_array[M - 1], 'r', label = 'truncamento')
            # plt.xlabel('Posição na barra')
            # plt.ylabel('erro')
            # plt.legend()
            # #plot do erro normalizado ao longo do tempo
            # plt.subplot(222)
            # plt.title('Erro normalizado ao longo do tempo \n')
            # plt.plot(np.arange(M+1),enorm)
            # plt.xlabel('instante')
            # plt.ylabel('erro')
            # plt.savefig(str(maypath) + '/Images/ErroN = ' + str(N) + ', L = ' + str(lambd) + '.png')
            # plt.close('all')
            # #show pyplot
            # #plt.show()
            N = 2*N
            print(temp - time.time())

































if __name__ == '__main__':
    main()
