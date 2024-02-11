# %%
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.optimize
import scipy.sparse
import cvxpy as cp
import relplot as rp
import math

# %%
def prepare_dataset(n, skew_function):
    res = []
    fa = []
    ya = []
    for _ in range(n):
        f = np.random.uniform(0,0.99)
        y = int(np.random.uniform() > 1 - skew_function(f))
        fa.append(f)
        ya.append(y)
    return (np.array(fa), np.array(ya))

# %%
def prepare_and_plot_dataset(N, fun, bins=50):
    S = prepare_dataset(N, fun)
    positive = np.zeros(bins)
    total = np.zeros(bins)
    for (f,y) in zip(S[0], S[1]):
        bin_num = int(f*bins)
        total[bin_num] += 1
        positive[bin_num] += y
        
    x = np.linspace(0, 1, bins+1)
    fig,axs = plt.subplots(1)
    axs.set_xlabel("prediction f")
    axs.set_ylabel("Fraction of yes instances (E y | f)")
    axs.bar(x[:-1], positive/total, width = 1/bins)
    axs.plot(x, list(map(fun, x)), color="red")
    return S

S_calibrated = prepare_and_plot_dataset(2**10+1, lambda x:x+0.01)

# %%
def smCE_LP(S):
    (x_list, y_list) = S
    indices = np.argsort(x_list)
    x_list = x_list[indices]
    y_list = y_list[indices]
    A = np.diag([1]*len(x_list))
    A = (A -np.roll(A, 1, axis = 1))[0:len(x_list)-1]
    A = np.concatenate([A, -A])
    b = (x_list- np.roll(x_list,1,axis=0))[1:len(x_list)]
    b = np.concatenate([b,b])
    c = y_list - x_list
    return c, A, b

# %%
def smCE_uc(S):
    (x_list, y_list) = S
    n = len(x_list)
    indices = np.argsort(x_list)
    x_list = x_list[indices]
    y_list = y_list[indices]
    A = np.diag([1]*len(x_list))
    A = (A -np.roll(A, 1, axis = 1))[0:len(x_list)-1]
    temp = np.concatenate([A, -A])
    counter = temp.shape[0]
    A = np.zeros((counter+ 2*2**int(np.log(n)/np.log(2))-2, n))
    A[0:counter] = temp
    k = int(np.log2(n-1))
    for i in range(1, k+1):
        for j in range(1, 2**(k - i)+1):
            A[counter][(j-1)*2**i] = 1
            A[counter][j*2**i] = -1

            A[counter+1][(j-1)*2**i] = -1         
            A[counter+1][j*2**i] = 1
            counter +=2
    b = np.abs(A@x_list)
    c = y_list - x_list
    return c, A, b



# %%
#c, A, b = smCE_LP(S_calibrated)
#n = len(S_calibrated[0])
#x = cp.Variable(n)
#objective = cp.Minimize(np.array([c/n]) @ x)
#constraints = [-1 <= x, x <= 1, A@x <= b]
#prob = cp.Problem(objective, constraints)
#result = prob.solve()
#print(-result)

# %%
import warnings
warnings.filterwarnings('ignore')
import statistics
# %%
y = np.zeros(7)
error_low = np.zeros(7)
error_high = np.zeros(7)
for i in range(5,12):
    #for eps in [0.1,0.07,0.05,0.03,0.01, 0.007, 0.005,0.003, 0.001]:
    counter = 0
    l = []
    for _ in range(100):
        S_calibrated = prepare_dataset(2**i+1, lambda x:x+0.01)
        c,A,b = smCE_uc(S_calibrated)
        n = len(S_calibrated[0])
        x = cp.Variable(n)
        objective = cp.Minimize(np.array([c/n]) @ x +2*math.log(n)*cp.norm(cp.maximum(A @ x-b, np.zeros(len(A)) ),"inf"))
        constraints = [-1 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)
        result = -prob.solve()
        l.append(result)
    print(i)
    #print(l)
    m = statistics.median(l)
    y[i-5] = statistics.median(l)
    error_low[i-5] = np.quantile(l, 0.25)
    error_high[i-5] = np.quantile(l, 0.75)
print(y)
print(error_low)
print(error_high)
        #if result >= eps:
            #counter+=1
        #if counter>50:
            #break
    #if counter > 50:
    #    print(i)
    #    print(counter)
    #    print(eps)
    #    break

print('smECE')
y = np.zeros(7)
error_low = np.zeros(7)
error_high = np.zeros(7)
# %%
for i in range(5,12):
    #for eps in [0.1,0.07,0.05,0.03,0.01, 0.007, 0.005,0.003, 0.001]:
    counter = 0
    l = []
    for _ in range(100):
        S_calibrated = prepare_dataset(2**i+1, lambda x:x+0.01)
        result = rp.smECE(S_calibrated[0],S_calibrated[1]) 
        l.append(result)
        #if result >= eps:
        #    counter+=1
        #if counter>50:
        #    break
    print(i)
    #print(l)
    y[i-5] = statistics.median(l)
    error_low[i-5] = np.quantile(l, 0.25)
    error_high[i-5] = np.quantile(l, 0.75)
    #if counter > 50:
    #    print(i)
    #    print(counter)
    #    print(eps)
    #    break
print(y)
print(error_low)
print(error_high)

# %%
def LDTC(S, eps):
    (x_list, y_list) = S
    n = len(x_list)
    U = np.array(range(int(1/eps)))*eps
    u = len(U)
    A_1 = np.zeros((n*2,n*2*u))
    A_2 = np.zeros((u,n*2*u))
    for i in range(2*n):
        for j in range(u):
            A_1[i][j*2*n+i] = 1
    for i in range(u):
        A_2[i][2*i*n: (2*i+1)*n] = [-U[i]]*n
        A_2[i][(2*i+1)*n: (2*i+2)*n] = [1-U[i]]*n 

    b_1 = np.zeros(2*n)
    for i in range(n):
        if y_list[i] == 0:
            b_1[i] = 1/n
        else:
            b_1[i+n] = 1/n 
    b_2 = np.zeros(u)
    c = np.zeros(u*n*2)
    for i in range(u):
        for j in range(n):
            c[2*n*i+j] = U[i]- x_list[j]
            c[2*n*i+j+n] = U[i]- x_list[j]
    c = np.abs(c)
    return A_1, A_2, b_1, b_2, c

print('LDTC')
# %%
y = np.zeros(4)
error_low = np.zeros(4)
error_high = np.zeros(4)
for i in range(5,9):
    #for eps in [0.1,0.07,0.05,0.03,0.01, 0.007, 0.005,0.003, 0.001]:
    counter = 0
    eps = 1/(10*2**(i/2))
    l=[]
    
    for _ in range(20):
        S_calibrated = prepare_dataset(2**i+1, lambda x:x+0.01)
        n = len(S_calibrated[0])
        A_1, A_2, b_1, b_2, c = LDTC(S_calibrated, eps)
        x = cp.Variable(n*2*int(1/eps))
        objective = cp.Minimize(c @ x+ 4*cp.norm(A_1 @ x-b_1 ,1) + 4*cp.norm(A_2 @ x ,1))
        constraints = [0 <= x]
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve()
            l.append(result)
        except Exception as e:
            continue
    print(i)
    #print(l)
    y[i-5] = statistics.median(l)
    error_low[i-5] = np.quantile(l, 0.25)
    error_high[i-5] = np.quantile(l, 0.75)
    #    if result >= eps:
    #        counter+=1
    #    if counter>50:
    #        break
    #if counter > 50:
    #    print(i)
    #    print(counter)
    #    print(eps)
    #    break

print(y)
print(error_low)
print(error_high)





