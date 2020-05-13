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
def f (x,t, item):
    if item == 'a':
        f = 10*math.cos(10*t)*x*x*(1 - x)**2 - (1 + math.sin(10*t))*(12*x*x - 12*x + 2)

    if item == 'b':
        f = (math.exp(t - x)*(math.cos(5*t*x) - 5*x*math.sin(5*t*x))) - (math.exp(t - x)*((1 - 25*t**2)*math.cos(5*t*x) + 10*t*math.sin(5*t*x)))

    return f

#funcao com a resposta exata da temperatura da barra
@jit(nopython=True)
def u (x, t, item):
    if item == 'a':
        u = (1 + math.sin(10*t))*x*x*(1 - x)**2

    if item == 'b':
        u = math.exp(t - x)*math.cos(5*t*x)

    return u

#resolucao do Exercicio 1, utilizando o metodo 11
@jit(nopython=True)
def resolution_a(N, M, uik_array, true_uik_array, eik_array, tik_array, Dx, Dt, item):

    #i sempre sera iterador de espaco
    #k sempre sera iterador de tempo

###########
##Calculo aproximada
########
    #condicoes de fronteiras
    for k in range(1, M + 1):
        uik_array[k][0] = u(0, k*Dt, item)
        uik_array[k][N] = u(N*Dx, k*Dt, item)
    #condicao inicial
    for i in range(N + 1):
        uik_array[0][i] = u(Dx*i, 0, item)
    #laco para calcular os elementos depois de dada as condicoes inciais
    for k in range(M):
        for i in range(1, N):
            uik_array[k + 1][i] = uik_array[k][i] + Dt*((uik_array[k][i - 1] - 2*uik_array[k][i] + uik_array[k][i + 1])/(Dx*Dx) + f(Dx*i, Dt*k, item))

##########
##Calculo exato
##########
    for k in range(M + 1):
        for i in range(N + 1):
            true_uik_array[k][i] = u(Dx*i, Dt*k, item)

#########
##Calculo dos erros e truncamentos
###########

    #calculo do truncamento
    for k in range(M):
        for i in range(1, N):
            tik_array[k][i] = ((true_uik_array[k+1][i] - true_uik_array[k][i])/Dt) -((true_uik_array[k][i - 1] - 2*true_uik_array[k][i] + true_uik_array[k][i + 1])/(Dx**2)) - (f(Dx*i, Dt*k, item))

    #Calculo do erro
    for k in range(M):
        for i in range(1, N):
            eik_array[k + 1][i] = eik_array[k][i] + Dt*((eik_array[k][i - 1] - 2*eik_array[k][i] + eik_array[k][i + 1])/(Dx*Dx) + tik_array[k][i])




##########################
#######exercicio 2
##########################

#decompoe a matriz tridiagonal salva em 2 vetores em um sistema LDL
@jit(nopython=True)
def Decomp_LD(A_P_arrary, A_S_arrary, L_arrary, D_arrary, N):
    D_arrary[0] = A_P_arrary[0]
    for i in range(1, N - 1):
        L_arrary[i] = A_S_arrary[i]/D_arrary[i - 1]
        D_arrary[i] = A_P_arrary[i] - L_arrary[i]**2*D_arrary[i - 1]


#esta funcao resolve um sistema LDL de forma analoga a um sistema LU
@jit(nopython=True)
def Solving_LD(L_arrary, D_arrary, b_array, N, X3):
    #vetores de resultador intermadiarios
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

#Resolve um sistema pelo metodo de euler implicito
@jit(nopython=True)
def euler(A_P_arrary, A_S_arrary, L_arrary, D_arrary, uik_array, N, M, Dt, Dx, lambd, item):

    b_array = np.zeros((N - 1), dtype = np.float64)
    X3 = np.zeros((N - 1), dtype = np.float64)

    ##Esta secao pode ser utilizada caso voce deseje resolver tudo dentro desta funcao
    #
    # for i in range(N - 1):
    #     A_P_arrary[i] = 1 + 2*lambd
    # for i in range(1, N - 1):
    #     A_S_arrary[i] = -lambd


    ###Esta parte esta comentado pois o sistema a funcao resolution_a ja resolve este problema,
    ### caso deseje usar este metodo antes dela, descomentar esta parte
    #
    # #COndicoes inicias
    # for i in range(N + 1):
    #     uik_array[0][i] = u(Dx*i, 0, item)
    # #condicao de contorno
    # for k in range(1, M + 1):
    #     uik_array[k][0] = u(0, k*Dt, item)
    #     uik_array[k][N] = u(N*Dx, k*Dt, item)

    #resolvendo euler implicito
    ##Esta secao pode ser utilizada caso voce deseje resolver tudo dentro desta funcao
    #
    #Decomp_LD(A_P_arrary, A_S_arrary, L_arrary, D_arrary, N)

    for k in range(0, M):
        for i in range(N - 1):
            if i == 0:
                b_array[i] = Dt*f(Dx*(i + 1), Dt*(k + 1), item) + uik_array[k][i + 1] + lambd*u(0, (k+1)*Dt, item)
            elif i == (N - 2):
                b_array[i] = Dt*f(Dx*(i + 1), Dt*(k + 1), item) + uik_array[k][i + 1] + lambd*uik_array[k + 1][N]
            else:
                b_array[i] = Dt*f(Dx*(i + 1), Dt*(k + 1), item) + uik_array[k][i + 1]

        Solving_LD(L_arrary, D_arrary, b_array, N, X3)
        for t in range(N - 1):
            uik_array[k + 1][t + 1] = X3[t]


#Resolve um sistema pelo metodo de crank-Nicolson
@jit(nopython=True)
def crankNicolson(A_P_arrary, A_S_arrary, L_arrary, D_arrary, uik_array, N, M, Dt, Dx, lambd, item):

    b_array = np.zeros((N - 1), dtype = np.float64)
    X3 = np.zeros((N - 1), dtype = np.float64)

    ##Esta secao pode ser utilizada caso voce deseje resolver tudo dentro desta funcao
    #
    # for i in range(N - 1):
    #     A_P_arrary[i] = 1 + lambd
    # for i in range(1, N - 1):
    #     A_S_arrary[i] = -lambd/2

    ###Esta parte esta comentado pois o sistema a funcao resolution_a ja resolve este problema,
    ### caso deseje usar este metodo antes dela, descomentar esta parte
    #
    # #COndicoes inicias
    # for i in range(N + 1):
    #     uik_array[0][i] = u(Dx*i, 0)
    # #condicao de contorno
    # for k in range(1, M + 1):
    #     uik_array[k][0] = u(0, k*Dt)
    #     uik_array[k][N] = u(N*Dx, k*Dt)

    #resolvendo euler implicito
    ##Esta secao pode ser utilizada caso voce deseje resolver tudo dentro desta funcao
    #
    #Decomp_LD(A_P_arrary, A_S_arrary, L_arrary, D_arrary, N)

    for k in range(0, M):
        for i in range(N - 1):
            if i == 0:
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), item) + f(Dx*(i + 1), Dt*k, item)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2]) + (lambd/2)*uik_array[k + 1][0]
            elif i == (N - 2):
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), item) + f(Dx*(i + 1), Dt*k, item)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2]) + (lambd/2)*uik_array[k + 1][N]
            else:
                b_array[i] = (Dt/2)*(f(Dx*(i + 1), Dt*(k + 1), item) + f(Dx*(i + 1), Dt*k, item)) + (1 - lambd)*uik_array[k][i + 1] + (lambd/2)*(uik_array[k][i] + uik_array[k][i + 2])

        Solving_LD(L_arrary, D_arrary, b_array, N, X3)
        for t in range(N - 1):
            uik_array[k + 1][t + 1] = X3[t]




def main():
    itlist = ['a', 'b']
    llist = [1, 0.51, 0.5, 0.25]

    for lambd in llist:
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

            #Criacao das matrizes
            uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)        #uik aproximado pelo metodo 11
            e_uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)      #euler uik aproximada
            c_uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)      #crankNicolson aproximado
            true_uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)   #uik real
            #matriz de erros
            eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)        #erros do metodo 11
            #matriz de truncamento
            tik_array = np.zeros((M + 1, N + 1), dtype = np.float64)        #erro de truncamento pelo metodo 11

            #matrizes para a resolucao tridiagonal
                #relacionado a euler implicito
            e_A_P_arrary = np.zeros((N - 1), dtype = np.float64)            #Diagonal principal da matriz A
            e_A_S_arrary = np.zeros((N - 1), dtype = np.float64)            #Diagonal secundária da amtriz A
            e_D_arrary = np.zeros((N - 1), dtype = np.float64)              #Matriz diagonal da decomposicao
            e_L_arrary = np.zeros((N - 1), dtype = np.float64)              #Matirz bidiagonal da decomposicao
                #relacionado a C-N
            c_A_P_arrary = np.zeros((N - 1), dtype = np.float64)            #Diagonal principal da matriz A
            c_A_S_arrary = np.zeros((N - 1), dtype = np.float64)            #Diagonal secundária da amtriz A
            c_D_arrary = np.zeros((N - 1), dtype = np.float64)              #Matriz diagonal da decomposicao
            c_L_arrary = np.zeros((N - 1), dtype = np.float64)              #Matirz bidiagonal da decomposicao

            #As decomposicoes podem ser reaproveitadas para cada funcao de calor (f e u distintas)
                #decomposicao de euler
                    #composicao da matriz tridiagonal para resolucao
            for i in range(N - 1):
                e_A_P_arrary[i] = 1 + 2*lambd
            for i in range(1, N - 1):
                e_A_S_arrary[i] = -lambd

            Decomp_LD(e_A_P_arrary, e_A_S_arrary, e_L_arrary, e_D_arrary, N)
                #decomposicao C-N
                    #compsicao da matriz tridiagonal
            for i in range(N - 1):
                c_A_P_arrary[i] = 1 + lambd
            for i in range(1, N - 1):
                c_A_S_arrary[i] = -lambd/2

            Decomp_LD(c_A_P_arrary, c_A_S_arrary, c_L_arrary, c_D_arrary, N)

            for item in itlist:
                #################
                ##RESOLUCOES
                ###############
                    #Exercicio 1
                resolution_a(N, M, uik_array, true_uik_array, eik_array, tik_array, Dx, Dt, item)
                    #Exercicio 2
                        #euler
                euler(e_A_P_arrary, e_A_S_arrary, e_L_arrary, e_D_arrary, e_uik_array, N, M, Dt, Dx, lambd, item)
                        #C-N
                crankNicolson(c_A_P_arrary, c_A_S_arrary, c_L_arrary, c_D_arrary, c_uik_array, N, M, Dt, Dx, lambd, item)

                #erro normalizado
                enorm = np.zeros((M + 1, 1), dtype = np.float64)
                enorm = np.amax(np.absolute(eik_array), axis = 1) #normalizado esta estranho


                yaxis = np.arange(N+1)
                #figura 1
                plt.figure(1)
                #plot do estado final aproximado
                plt.subplot(131)
                plt.title('Temperatura ' + item + str(N) + str(lambd))
                plt.plot(yaxis, uik_array[M], 'r', label = 'aproximado')
                plt.plot(yaxis, e_uik_array[M], 'g', label = 'euler')
                plt.plot(yaxis, c_uik_array[M], 'y', label = 'nicolsol')
                plt.plot(yaxis, true_uik_array[M], 'b', label= 'exato')
                plt.legend()
                plt.xlabel('Posição na barra')
                plt.ylabel('temperatura')
                plt.show()
                #plot do final exato
                # plt.subplot(122)
                # plt.title('Temperatura real')
                #
                # plt.xlabel('Posição na barra')
                # plt.ylabel('temperatura')

                #plot do erro ao longo da barra
                # plt.subplot(132)
                # plt.title('Erro ao longo da barra no instante T \n')
                # plt.plot(yaxis, eik_array[M], 'b', label = 'erro')
                # plt.plot(yaxis, tik_array[M - 1], 'r', label = 'truncamento')
                # plt.xlabel('Posição na barra')
                # plt.ylabel('erro')
                # plt.legend()
                # #plot do erro normalizado ao longo do tempo
                # plt.subplot(133)
                # plt.title('Erro normalizado ao longo do tempo \n')
                # plt.plot(np.arange(M+1),enorm)
                # plt.xlabel('instante')
                # plt.ylabel('erro')
                # plt.savefig(str(maypath) + '/Images/Grafs N = ' + str(N) + ', L = ' + str(lambd) + '.png')
                # plt.close('all')
                # #show pyplot
                # plt.show()


            N = N*2
            print(temp - time.time())


if __name__ == '__main__':
    main()
