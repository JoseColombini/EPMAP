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
import random


T = 1 #Ex 1
X = 1

##funcao qeu insere calor
@jit(nopython=True)
def f (x,t, Dx, pk):
    f = r(t)*gh(x, pk, Dx)
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
def Decomp_LD(A_P_array, A_S_array, L_array, D_array, N):
    D_array[0] = A_P_array[0]
    for i in range(1, N - 1):
        L_array[i] = A_S_array[i]/D_array[i - 1]
        D_array[i] = A_P_array[i] - L_array[i]**2*D_array[i - 1]


@jit(nopython=True)
def Solving_LD(L_array, D_array, b_array, N, X3):

    X1 = np.zeros((N - 1), dtype = np.float64)
    X2 = np.zeros((N - 1), dtype = np.float64)

    X1[0] = b_array[0]
    for i in range(1, N - 1):
        X1[i] = b_array[i] - L_array[i]*X1[i - 1]


    for i in range(N - 1):
        X2[i] = X1[i]/D_array[i]


    X3[N - 2] = X2[N - 2]
    for i in range(N - 3, -1, -1):
        X3[i] = X2[i] - L_array[i + 1]*X3[i + 1]


@jit
def crankNicolson(A_P_array, A_S_array, L_array, D_array, uik_array, N, M, Dt, Dx, lambd, pk):

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

        Solving_LD(L_array, D_array, b_array, N, X3)
        for t in range(N - 1):
            uik_array[k + 1][t + 1] = X3[t]


#######################################################
##### EP 2
########################################################
def normalSystemArrange (p_uik_array, T_uik_array, NS_array, RS_array):
    for i in range(len(RS_array)):
        for t in range(i, len(RS_array)):
            NS_array[i][t] = np.dot(p_uik_array[i], p_uik_array[t])
            NS_array[t][i] = NS_array[i][t]

            RS_array[i] = np.dot(T_uik_array, p_uik_array[i])


def ldl_decomp(n, A_array, L_array, D_array):
	#loop inicial de construção das matrizes L e D
	v = []
	for i in range(0, n):
		v.append(0)
		for j in range(0, n):
			D_array[i] = A_array[i][i]
			L_array[i][j] = A_array[i][j]/D_array[i]
	for i in range(0, n):
		soma = 0
		for j in range(0, i):
			v[j] = L_array[i][j] * D_array[j]
			soma = soma + L_array[i][j] * v[j]
		D_array[i] = A_array[i][i] - soma
		for j in range(0, n):
			soma1 = 0
			for k in range (0, i):
				soma1 = soma1 + L_array[j][k] * v[k]
			L_array[j][i] = (A_array[j][i] - soma1)/D_array[i]


def ldl_solver(L_array, D_array, RS_array, N, X3):

    X1 = np.zeros((N), dtype = np.float64)
    X2 = np.zeros((N), dtype = np.float64)

    X1[0] = RS_array[0]
    for i in range(1, N):
        soma_l = 0
        for j in range(i):
            soma_l += L_array[i][j]*X1[j]
        X1[i] = (RS_array[i] - soma_l)

    for i in range(N):
        X2[i] = X1[i]/D_array[i]


    X3[N - 1] = X2[N - 1]
    print(X3)
    for i in range(N - 2, -1, -1):
        soma_l = 0
        for j in range(i+1, N):
            soma_l += X3[j]*np.transpose(L_array)[i][j]
        X3[i] = X2[i] - soma_l


def E2 (T_uik_array, ak_array, p_uik_array, Dx):
    soma = 0.0
    for i in range(1, len(T_uik_array[0]) - 1):
        somaak = 0.0
        for k in range(len(ak_array)):
            somaak += ak_array[k]*p_uik_array[k][i]
        soma += (T_uik_array[0][i] - somaak)**2
    return math.sqrt(Dx*(soma))




def main():



    temp = time.time()
    a = input("selecione o teste (a, b, c, d) ou livre (l): ")
    if(a == 'a'):
        N = 128
        M = N
        nf = 1
        pk_array = np.zeros(nf, dtype = np.float64)
        ak_array = np.zeros(nf, dtype = np.float64)
        ak_array[0] = 7
        pk_array[0] = 0.35

        Dx = X/N
        Dt = T/M
        lambd = Dt/(Dx*Dx)
        print("N = ", N)
        print("M = ", M)
        print("Dx = ", Dx)
        print("Dt = ", Dt)
        print("lambda = ", lambd)
        print("nf = ", nf)
        print("pk = ", pk_array)
        print("ak = ", ak_array)

        #Creat the matix uik that describ all bar in datetime
        #Each line is a bar in one moment
        #So each colum is a position in the bar
        #uik_array(2, 5) is the point 5 of the bar at the moment 2
        uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)  #euler uik
        T_uik_array = np.zeros((1, N + 1), dtype = np.float64) #uik real
        p_uik_array = np.zeros((nf, N + 1), dtype = np.float64)
        #matriz de erros
        eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)

        A_P_array = np.zeros((N - 1), dtype = np.float64)
        A_S_array = np.zeros((N - 1), dtype = np.float64)
        D_array = np.zeros((N - 1), dtype = np.float64)
        L_array = np.zeros((N - 1), dtype = np.float64)

        DR_array = np.zeros((nf), dtype = np.float64)
        LR_array = np.zeros((nf, nf), dtype = np.float64)
        XR_array = np.zeros((nf), dtype = np.float64)

        NS_array = np.zeros((nf, nf), dtype = np.float64)
        RS_array = np.zeros((nf), dtype = np.float64)



        for i in range(N - 1):
            A_P_array[i] = 1 + lambd
        for i in range(1, N - 1):
            A_S_array[i] = -lambd/2


        #COndicoes inicias
        for i in range(N + 1):
            uik_array[0][i] = 0
        #condicao de contorno
        for k in range(1, M + 1):
            uik_array[k][0] = 0
            uik_array[k][N] = 0

        Decomp_LD(A_P_array, A_S_array, L_array, D_array, N)
        #building A_array
        for i in range(nf):
            crankNicolson(A_P_array, A_S_array, L_array, D_array, uik_array, N, M, Dt, Dx, lambd, pk_array[i])
            T_uik_array += ak_array[i]*uik_array[M]
            p_uik_array[i] = uik_array[M]


        normalSystemArrange(p_uik_array, T_uik_array, NS_array, RS_array)
        print(NS_array)
        print(RS_array)
        ldl_decomp(nf, NS_array, LR_array, DR_array)
        ldl_solver(LR_array, DR_array, RS_array, nf, XR_array)
        print("result")
        print(XR_array)




    elif(a == 'b'):
        N = 128
        M = N
        nf = 4
        pk_array = np.zeros(nf, dtype = np.float64)
        ak_array = np.zeros(nf, dtype = np.float64)
        pk_array = [0.15, 0.3, 0.7, 0.8]
        ak_array = [2.3, 3.7, 0.3, 4.2]

        Dx = X/N
        Dt = T/M
        lambd = Dt/(Dx*Dx)
        print("N = ", N)
        print("M = ", M)
        print("Dx = ", Dx)
        print("Dt = ", Dt)
        print("lambda = ", lambd)
        print("nf = ", nf)
        print("pk = ", pk_array)
        print("ak = ", ak_array)

        #Creat the matix uik that describ all bar in datetime
        #Each line is a bar in one moment
        #So each colum is a position in the bar
        #uik_array(2, 5) is the point 5 of the bar at the moment 2
        uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)  #euler uik
        T_uik_array = np.zeros((1, N + 1), dtype = np.float64) #uik real
        p_uik_array = np.zeros((nf, N + 1), dtype = np.float64)
        #matriz de erros
        eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)

        A_P_array = np.zeros((N - 1), dtype = np.float64)
        A_S_array = np.zeros((N - 1), dtype = np.float64)
        D_array = np.zeros((N - 1), dtype = np.float64)
        L_array = np.zeros((N - 1), dtype = np.float64)

        DR_array = np.zeros((nf), dtype = np.float64)
        LR_array = np.zeros((nf, nf), dtype = np.float64)
        XR_array = np.zeros((nf), dtype = np.float64)

        NS_array = np.zeros((nf, nf), dtype = np.float64)
        RS_array = np.zeros((nf), dtype = np.float64)



        for i in range(N - 1):
            A_P_array[i] = 1 + lambd
        for i in range(1, N - 1):
            A_S_array[i] = -lambd/2


        #COndicoes inicias
        for i in range(N + 1):
            uik_array[0][i] = 0
        #condicao de contorno
        for k in range(1, M + 1):
            uik_array[k][0] = 0
            uik_array[k][N] = 0

        Decomp_LD(A_P_array, A_S_array, L_array, D_array, N)
        #building A_array
        for i in range(nf):
            crankNicolson(A_P_array, A_S_array, L_array, D_array, uik_array, N, M, Dt, Dx, lambd, pk_array[i])
            T_uik_array += ak_array[i]*uik_array[M]
            p_uik_array[i] = uik_array[M]


        normalSystemArrange(p_uik_array, T_uik_array, NS_array, RS_array)
        print(NS_array)
        print(RS_array)
        ldl_decomp(nf, NS_array, LR_array, DR_array)
        ldl_solver(LR_array, DR_array, RS_array, nf, XR_array)
        print("result")
        print(XR_array)




    elif(a == 'c'):
        Nlist = [128]#, 256, 512, 1024]
        f = open("Arquivo teste para o EP2.txt", "r")
        pk_array = f.readline().split()
        ak_array = np.zeros(len(pk_array), dtype = np.float64)
        for l in Nlist:
            N = l
            M = N
            nf = len(pk_array)
            Dx = X/N
            Dt = T/M
            lambd = Dt/(Dx*Dx)
            print("N = ", N)
            print("M = ", M)
            print("Dx = ", Dx)
            print("Dt = ", Dt)
            print("lambda = ", lambd)
            print("nf = ", nf)
            print("pk = ", pk_array)
            uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)  #euler uik
            T_uik_array = np.zeros((1, N + 1), dtype = np.float64) #uik real
            p_uik_array = np.zeros((nf, N + 1), dtype = np.float64)
            #matriz de erros
            eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)

            A_P_array = np.zeros((N - 1), dtype = np.float64)
            A_S_array = np.zeros((N - 1), dtype = np.float64)
            D_array = np.zeros((N - 1), dtype = np.float64)
            L_array = np.zeros((N - 1), dtype = np.float64)

            DR_array = np.zeros((nf), dtype = np.float64)
            LR_array = np.zeros((nf, nf), dtype = np.float64)
            XR_array = np.zeros((nf), dtype = np.float64)

            NS_array = np.zeros((nf, nf), dtype = np.float64)
            RS_array = np.zeros((nf), dtype = np.float64)


            for i in range(N - 1):
                A_P_array[i] = 1 + lambd
            for i in range(1, N - 1):
                A_S_array[i] = -lambd/2


            #COndicoes inicias
            for i in range(N + 1):
                uik_array[0][i] = 0
            #condicao de contorno
            for k in range(1, M + 1):
                uik_array[k][0] = 0
                uik_array[k][N] = 0

            Decomp_LD(A_P_array, A_S_array, L_array, D_array, N)
            #building A_array
            for i in range(nf):
                crankNicolson(A_P_array, A_S_array, L_array, D_array, uik_array, N, M, Dt, Dx, lambd, float(pk_array[i]))
                p_uik_array[i] = uik_array[M]


            t = 0
            i = 0
            T_uik_array[0][i] = float(f.readline())
            i = 1
            t = 1
            for x in f:
                if (t%(2048/l)) == 0:
                    T_uik_array[0][i] = float(x)
                    i = i + 1
                t = t + 1

            normalSystemArrange(p_uik_array, T_uik_array, NS_array, RS_array)
            print(NS_array)
            print(RS_array)
            ldl_decomp(nf, NS_array, LR_array, DR_array)
            ldl_solver(LR_array, DR_array, RS_array, nf, XR_array)
            print("result")
            print(XR_array)

            E = E2(T_uik_array,ak_array, p_uik_array, Dx)
            print("eerr/: ", E)


    elif(a == 'd'):
        Nlist = [128]#, 256, 512, 1024]
        f = open("Arquivo teste para o EP2.txt", "r")
        pk_array = f.readline().split()
        ak_array = np.zeros(len(pk_array), dtype = np.float64)
        for l in Nlist:
            N = l
            M = N
            nf = len(pk_array)
            Dx = X/N
            Dt = T/M
            lambd = Dt/(Dx*Dx)
            print("N = ", N)
            print("M = ", M)
            print("Dx = ", Dx)
            print("Dt = ", Dt)
            print("lambda = ", lambd)
            print("nf = ", nf)
            print("pk = ", pk_array)
            uik_array = np.zeros((M + 1, N + 1), dtype = np.float64)  #euler uik
            T_uik_array = np.zeros((1, N + 1), dtype = np.float64) #uik real
            p_uik_array = np.zeros((nf, N + 1), dtype = np.float64)
            #matriz de erros
            eik_array = np.zeros((M + 1, N + 1), dtype = np.float64)

            A_P_array = np.zeros((N - 1), dtype = np.float64)
            A_S_array = np.zeros((N - 1), dtype = np.float64)
            D_array = np.zeros((N - 1), dtype = np.float64)
            L_array = np.zeros((N - 1), dtype = np.float64)

            DR_array = np.zeros((nf), dtype = np.float64)
            LR_array = np.zeros((nf, nf), dtype = np.float64)
            XR_array = np.zeros((nf), dtype = np.float64)

            NS_array = np.zeros((nf, nf), dtype = np.float64)
            RS_array = np.zeros((nf), dtype = np.float64)


            for i in range(N - 1):
                A_P_array[i] = 1 + lambd
            for i in range(1, N - 1):
                A_S_array[i] = -lambd/2


            #COndicoes inicias
            for i in range(N + 1):
                uik_array[0][i] = 0
            #condicao de contorno
            for k in range(1, M + 1):
                uik_array[k][0] = 0
                uik_array[k][N] = 0

            Decomp_LD(A_P_array, A_S_array, L_array, D_array, N)
            #building A_array
            for i in range(nf):
                crankNicolson(A_P_array, A_S_array, L_array, D_array, uik_array, N, M, Dt, Dx, lambd, float(pk_array[i]))
                p_uik_array[i] = uik_array[M]


            t = 0
            i = 0
            T_uik_array[0][i] = float(f.readline())
            i = 1
            t = 1
            for x in f:
                if (t%(2048/l)) == 0:
                    T_uik_array[0][i] = float(x)*(1.0 + (0.01*(random.random() - 0.5)*2))
                    i = i + 1
                t = t + 1

            normalSystemArrange(p_uik_array, T_uik_array, NS_array, RS_array)
            print(NS_array)
            print(RS_array)
            ldl_decomp(nf, NS_array, LR_array, DR_array)
            ldl_solver(LR_array, DR_array, RS_array, nf, XR_array)
            print("result")
            print(XR_array)

            E = E2(T_uik_array,ak_array, p_uik_array, Dx)
            print("eerr/: ", E)
            yaxis = np.arange(N+1)

            plt.plot(yaxis, T_uik_array[0])
            plt.show()






    else:
        #Input of number of divisions
        N = int(input("Insert the number of x fraction (N): "))
        #lambd = float(input("Insert lambda: "))
        #Dt = lambd*Dx*Dx
        #M = int(round(T/Dt))
        M = int(input("Insert the number of t fraction (M): "))
        #Size of fractions
        nf = int(input("Insert the number of fonts (nf): "))
        pk_array = np.zeros(nf, dtype = np.float64)
        ak_array = np.zeros(nf, dtype = np.float64)

        for i in range(nf):
            pk_array[i] = float(input("Insert p"+ str(i)+" position: "))
        for i in range(nf):
            ak_array[i] = float(input("Insert a"+ str(i)+" weight: "))




    #return 0





    #resolvendo euler implicito




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
