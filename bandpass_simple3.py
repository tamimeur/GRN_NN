import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
import pylab
import random

#############
# parameters:
#############


a1 = 7.7
a2 = 5.1
a3 = 8.6
k1 = 3.9
k2 = 7.1
b0 = 6.1
b1 = 2.9
b2 = 5.8


s = 100
#design = ['p3','g2','p2','g1','p1','g0']
design = ['p3','g0']
#########
# ODE(s):
#########

# Initial conditions:
A0 = 0
B0 = 0
G0 = 0
inits = [A0,B0,G0]
species = ['g0','g1','g2']
deg = [b0,b1,b2]

isthere = 'g0' in design
print isthere
print "INDEX TEST: "
print design.index('g0')

# Equations
def eqns(X):
    global s
    global design
    global species
    global deg
    eqn = []

    for gene in species:
        if gene in design:
            inx = design.index(gene)
            spec = species.index(gene)
            if design[inx-1] == 'p1':
                eqn.append(a1/(1+k1*X[1]) - deg[spec]*X[spec])
            elif design[inx-1] == 'p2':
                eqn.append(a2/(1+k2*X[2]) - deg[spec]*X[spec])
            elif design[inx-1] == 'p3':
                eqn.append((a3*s)/(1+s) - deg[spec]*X[spec])
                #print "found prom3 for gene ", gene, deg[spec]
        else:
            eqn.append(0)

    ################################# REP CASCADE ###################
    #P3G2|P2G1|P1G0 - FINAL DES
    # eqn = [a1/(1+k1*X[1]) - b0*X[0], 
    # a2/(1+k2*X[2]) - b1*X[1], 
    # (a3*s)/(1+s) - b2*X[2]]

    #P3G0
    # eqn = [(a3*s)/(1+s) - b0*X[0], 
    # 0, 
    # 0]



    return eqn


def dX_dt(X, t):
    return(np.array(eqns(X)))


##############
# Integration:
##############
# t = np.linspace(0, 20, 200)
# X0 = np.array(inits)
# X = integrate.odeint(dX_dt, X0, t)


# ########################
# # Steady state solution:
# ########################

# # Using scipy.optimize.fsolve
# G,A,B = X.T


# for j in range(0,1):


#     # a1 = random.uniform(0,10)
#     # a2 = random.uniform(0,10)
#     # a3 = random.uniform(0,10)
#     # k1 = random.uniform(0,10)
#     # k2 = random.uniform(0,10)
#     # b0 = random.uniform(0,10)
#     # b1 = random.uniform(0,10)
#     # b2 = random.uniform(0,10)

#     a1 = 7.7
#     a2 = 5.1
#     a3 = 8.6
#     k1 = 3.9
#     k2 = 7.1
#     b0 = 6.1
#     b1 = 2.9
#     b2 = 5.8

#     print "params= ", a1,a2,a3,k1,k2,b0,b1,b2

#     inducer_vals = [0,0.001,0.01, 0.1,1,10,100,1000,10000,100000] 
#     ss_res = []
#     for i in inducer_vals:
#     	s = i
#     	result = fsolve(dX_dt, X0, 0)
#     	print "s = ", s ," & steady state: ", result
#     	ss_res.append(result[0])

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)

#     sns.set_style("darkgrid")
#     ax.set_xticks(inducer_vals)
#     plt.semilogx(inducer_vals,ss_res)
#     print "RESULTS for design", design, " = ", inducer_vals, ss_res
#     plt.show()

def get_data(cur_des,it):

    global design
    global s
    design = cur_des
    inducer_vals = [0,0.000001,0.0000001,0.00001,0.0001,0.001,0.01, 0.1,1,10,100,1000,10000,100000,1000000,10000000] 
    #inducer_vals = np.arange(0, 100000, 10)
    ss_res = []
    for i in inducer_vals:
        s = i
        #result = fsolve(dX_dt, X0, 0)
        t = np.linspace(0, 20, 200)
        X0 = np.array([0,0,0])
        X = integrate.odeint(dX_dt, X0, t)


        ########################
        # Steady state solution:
        ########################

        # Using scipy.optimize.fsolve
        G,A,B = X.T
        #print "Sim output for ", design, " w/ s = ", i , " : ", G[199]
        ss_res.append(G[199])
        #ss_res.append(result[0])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    sns.set_style("darkgrid")
    ax.set_xticks(inducer_vals)
    plt.semilogx(inducer_vals,ss_res)
    #print "RESULTS for design", design, " = ", inducer_vals, ss_res
    #plt.show()
    #print str(cur_des)
    #fig.savefig('char_data'+str(cur_des)+'.png')

    return (inducer_vals,ss_res)

#fig.savefig('01_kinetics.png')
#pylab.show()
