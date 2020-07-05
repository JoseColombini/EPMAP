"""""""""""""""""""""""""""""""""
##### Exercicio 1 do EP1 #######
    Este arquivo realiza o calculo dos valores para a aproximacao da temperatura da barra
utilizando o método 11.
    As funçoes f(x, t) e u(x, t) possuem as descricao tanto para o item a como para o item b,
bastantdo apenas comentar a linha que nao sera utilizada e descomentar a linha que sera utilizada
    Os erros e truncamentos tambem sao calculados neste arquivo.
"""""""""""""""""""""""""""""""""


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
    f = (math.exp(t - x)*(math.cos(5*t*x) - 5*x*math.sin(5*t*x))) - (math.exp(t - x)*((1 - 25*t**2)*math.cos(5*t*x) + 10*t*math.sin(5*t*x)))       #b
    return f


#Estas funcoes foram depreciadas após a utilizacao dos conceitos das equacoes 2, 3, 4 para simplificar o codigo
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
    u = math.exp(t - x)*math.cos(5*t*x)                                         #b
    return u



@jit(nopython=True)
def resolution_a(N, M, uik_array, true_uik_array, eik_array, tik_array, Dx, Dt):
    #metodo 11
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
            #Creat the matix uik that describ all bar in time
            #Each line is a bar in one moment
            #So each colum is a position in the bar
            #uik_array(2, 5) is the point 5 of the bar at the moment 2
            uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)    #uik aproximado
            true_uik_array = np.zeros((M + 1, N + 1), dtype = np.float64) #uik real
            #matriz de erros
            eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)
            #matriz de truncamento
            tik_array = np.zeros((M + 1, N + 1), dtype = np.float64)

            #funcao para resolucao do exercicio
            resolution_a(N, M, uik_array, true_uik_array, eik_array, tik_array, Dx, Dt)

            #erro normalizado
            enorm = np.zeros((M + 1, 1), dtype = np.float64)
            enorm = np.amax(np.absolute(eik_array), axis = 1)


            yaxis = np.arange(N+1)
            #figura 1
            plt.figure(1, figsize = (20, 15))
            #plot aproximado
            plt.subplot(131)
            plt.title('Temperatura aproximada ao longo do tempo')
            plt.plot(yaxis, uik_array[0], 'b--', label = 't = 0.0')
            plt.plot(yaxis, uik_array[int(M/10)], 'g--', label = 't = 0.1')
            plt.plot(yaxis, uik_array[int(2*M/10)], 'r--', label = 't = 0.2')
            plt.plot(yaxis, uik_array[int(3*M/10)], 'c--', label = 't = 0.3')
            plt.plot(yaxis, uik_array[int(4*M/10)], 'm--', label = 't = 0.4')
            plt.plot(yaxis, uik_array[int(5*M/10)], 'y--', label = 't = 0.5')
            plt.plot(yaxis, uik_array[int(6*M/10)], 'b:', label = 't = 0.6')
            plt.plot(yaxis, uik_array[int(7*M/10)], 'g:', label = 't = 0.7')
            plt.plot(yaxis, uik_array[int(8*M/10)], 'r:', label = 't = 0.8')
            plt.plot(yaxis, uik_array[int(9*M/10)], 'c:', label = 't = 0.9')
            plt.plot(yaxis, uik_array[M], 'r-', label = 't = 1')




            plt.plot(yaxis, true_uik_array[M], 'k-', label = 'exato')
            plt.legend()
            plt.xlabel('Posição na barra')
            plt.ylabel('temperatura')
            #plt.savefig(str(maypath) + '/Images/Exercicio1c/Grafs N = ' + str(N) + ', L = ' + str(lambd) + '.png')

            #plot do erro ao longo da barra
            plt.figure(2, figsize = (20, 15))
            plt.subplot(221)
            plt.title('Erro ao longo da barra no instante T \n')
            plt.plot(yaxis, eik_array[M], 'b', label = 'erro')
            plt.plot(yaxis, tik_array[M - 1], 'r', label = 'truncamento')
            plt.xlabel('Posição na barra')
            plt.ylabel('erro')
            plt.legend()
            #plot do erro normalizado ao longo do tempo
            plt.subplot(222)
            plt.title('Erro normalizado ao longo do tempo \n')
            plt.plot(np.arange(M+1),enorm)
            plt.xlabel('instante')
            plt.ylabel('erro')
            #plt.savefig(str(maypath) + '/Images/Exercicio1c/Errs N = ' + str(N) + ', L = ' + str(lambd) + '.png')
            #plt.close('all')
            #show pyplot
            plt.show()
            print(temp - time.time())

































if __name__ == '__main__':
    main()
