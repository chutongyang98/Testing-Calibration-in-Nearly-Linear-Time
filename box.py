# This file includes our implementation of box-simplex method. Notice that when you call box_simplex with part = True, you are using S_0 and part = False is for S_base.

# %%
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.optimize
import scipy.sparse
import cvxpy as cp
import math
from numpy import linalg as LA

# %%
def prepare_dataset(n, skew_function, e):
    res = []
    fa = []
    ya = []
    for _ in range(n):
        f = np.random.uniform(0,1-e)
        #f = 0.5
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


# %%
# def smCE_uc(S):
#     (x_list, y_list) = S
#     n = len(x_list)
#     indices = np.argsort(x_list)
#     x_list = x_list[indices]
#     y_list = y_list[indices]
#     A = np.diag([1]*len(x_list))
#     A = (A -np.roll(A, 1, axis = 1))[0:len(x_list)-1]
#     temp = np.concatenate([A, -A])
#     counter = temp.shape[0]
#     A = np.zeros((counter+ 2*2**int(np.log(n)/np.log(2))-2, n))
#     A[0:counter] = temp
#     k = int(np.log2(n-1))
#     for i in range(1, k+1):
#         for j in range(1, 2**(k - i)+1):
#             A[counter][(j-1)*2**i] = 1
#             A[counter][j*2**i] = -1   
#             counter +=1
#         for j in range(1, 2**(k - i)+1):
#             A[counter][(j-1)*2**i] = -1                
#             A[counter][j*2**i] = 1
#             counter +=1
#     b = np.abs(A@x_list)
#     c = (y_list - x_list)/n
#     return c, A, b

def smCE_LP(S):
    (x_list, y_list) = S
    n = len(x_list)
    indices = np.argsort(x_list)
    x_list = x_list[indices]
    y_list = y_list[indices]
    A = np.diag([1]*len(x_list))
    A = (A -np.roll(A, 1, axis = 1))[0:len(x_list)-1]
    A = np.concatenate([A, -A])
    b = (x_list- np.roll(x_list,1,axis=0))[1:len(x_list)]
    b = np.concatenate([b,b])
    c = (y_list - x_list)/n
    return c, A, b

# %%


# %%
import cvxpy as cp
import math

# %%
def ATmul(x, part = False):
    n = len(x)
    if part:
        ret = np.zeros(2*(n-1) + 1) 
    else:
        ret = np.zeros(2*(n-1) + 2*2**int(np.log2(n))-1) #should be -2 but -1 for last row with all 0
    ret[0:(n-1)] = x[:-1] - x[1:]
    ret[n-1:2*(n-1)] = -x[:-1] + x[1:]
    if part:
        return ret/2
    k = int(np.log2(n-1))
    counter = 2*(n-1)
    for i in range(1, k+1):
        indices = np.array(range(2**(k - i)+1))
        indices = 2**i * indices
        ret[counter: counter + (2**(k - i))] = (x[indices])[:-1] - (x[indices])[1:]
        ret[counter+ (2**(k - i)): counter + 2*(2**(k - i))] = (-x[indices])[:-1] + (x[indices])[1:]     
        counter += 2 * (2**(k - i))
    return ret/2

def A_absTmul(x, part = False):
    n = len(x)
    if part:
        ret = np.zeros(2*(n-1) + 1) 
    else:
        ret = np.zeros(2*(n-1) + 2*2**int(np.log2(n))-1) #should be -2 but -1 for last row with all 0
    ret[0:(n-1)] = x[:-1] + x[1:]
    ret[n-1:2*(n-1)] = x[:-1] + x[1:]
    if part:
        return ret/2
    k = int(np.log2(n-1))
    counter = 2*(n-1)
    for i in range(1, k+1):
        indices = np.array(range(2**(k - i)+1))
        indices = 2**i * indices
        ret[counter: counter + (2**(k - i))] = (x[indices])[:-1] + (x[indices])[1:]
        ret[counter+ (2**(k - i)): counter + 2*(2**(k - i))] = (x[indices])[:-1] + (x[indices])[1:]     
        counter += 2 * (2**(k - i))
    return ret/2

# test= 2**np.array(range(9))
# print(ATmul(test, True).shape)
#print(A_@test/2)

# test= 2**np.array(range(9))
# print(A_absTmul(test))
# print(np.abs(A_)@test/2)

# %%
def Amul(y, size, part = False):
    n = len(y)
    ret = np.zeros(size) #should be -2 but -1 for last row with all 0
    counter = 0
    k = int(np.log2(size-1))
    for i in range(k):
        ret[0] += y[counter] - y[counter+ (2**(k - i))]
        ret[-1] += -y[counter+ (2**(k - i))-1] + y[counter+ 2*(2**(k - i))-1]
        indices = np.array(range(1, 2**(k - i)))
        indices = 2**i * indices
        ret[indices] += -(y[counter: counter + 2**(k - i)])[:-1] + (y[counter: counter + 2**(k - i)])[1:]
        counter +=(2**(k - i))
        ret[indices] += (y[counter: counter + 2**(k - i)])[:-1] - (y[counter: counter + 2**(k - i)])[1:]      
        counter +=(2**(k - i))
        if part:
            return ret/2
    ret[0] += y[counter] - y[counter+1]
    ret[-1] += -y[counter] + y[counter+1] 
    return ret/2

def A_absmul(y, size, part = False):
    n = len(y)
    ret = np.zeros(size) #should be -2 but -1 for last row with all 0
    counter = 0
    k = int(np.log2(size-1))
    for i in range(k):
        ret[0] += y[counter] + y[counter+ (2**(k - i))]
        ret[-1] += y[counter+ (2**(k - i))-1] + y[counter+ 2*(2**(k - i))-1]
        indices = np.array(range(1, 2**(k - i)))
        indices = 2**i * indices
        ret[indices] += (y[counter: counter + 2**(k - i)])[:-1] + (y[counter: counter + 2**(k - i)])[1:]
        counter +=(2**(k - i))
        ret[indices] += (y[counter: counter + 2**(k - i)])[:-1] + (y[counter: counter + 2**(k - i)])[1:]      
        counter +=(2**(k - i))
        if part:
            return ret/2
    ret[0] += y[counter] + y[counter+1]
    ret[-1] += y[counter] + y[counter+1] 
    return ret/2
# test= 2**np.array(range(16))
# print(Amul(test, 9, True))
# print(A_[0:16].T@test/2)

# test= 2**np.array(range(30))
# print(A_absmul(test, 9))
# print(np.abs(A_.T)@test/2)

# %%
def box_simplex(size, b, c, epsilon, p = False, f = None):
    n = size[0]
    L = 2
    x = np.zeros(size[0])
    y = np.ones(size[1]) / size[1]
    y_bar = np.ones(size[1]) / size[1]
    x_hat = np.zeros(size[0])
    y_hat = np.zeros(size[1]) 
    T = int( 48*np.log(size[1]) * L / epsilon)
    
    #A = A / L
    c = c / (L*np.log2(n))
    #A_abs = np.abs(A)
    
    for t in range(T):
        # Gradient oracle start
        #g_x = (A @ y + c)/3
        g_x = (Amul(y, n,part = p) + c)/3
        #g_y = (b - A.T @ x)/3 
        g_y = (b - ATmul(x,part = p))/3 
        
        # Step 7
        x_star = np.clip( -(g_x - 2 * x * A_absmul(y, n, part = p)) / (2 * A_absmul(y,n,part = p)+0.000001), -1, 1)
        
        # Step 8
        #y_p = y * np.exp((-1 / 2) * (g_y + A_abs.T @ (x_star**2) - A_abs.T @ (x**2)))
        y_p = y * np.exp((-1 / 2) * (g_y + A_absTmul(x_star**2,part = p) - A_absTmul(x**2,part = p)))
        y_p = y_p / y_p.sum()  # Ensure y_star is a probability vector
        
        # Step 9
        #x_p = np.clip( - (g_x - 2 * np.diag(x)@ A_abs @ y) / (2 * A_abs @ y_p+0.000001), -1, 1)
        x_p = np.clip( - (g_x - 2 * x* A_absmul(y, n,part = p)) / (2 * A_absmul(y_p, n,part = p)+0.000001), -1, 1)
        
        # Step 10: Running average maintenance
        x_hat = x_hat  + x_p / T
        y_hat = y_hat + y_p / T
        
        # Step 12: Extrapolation oracle start
        #g_x = (A @ y_p + c)/6
        #g_y = (b-A.T @ x_p)/6
        g_x = (Amul(y_p, n,part = p) + c)/6
        g_y = (b-ATmul(x_p,part = p))/6
        
        # Step 14
        x_bar = np.clip( - (g_x - 2 * x * A_absmul(y,n,part = p)) / (2 * A_absmul(y_bar, n,part = p)+0.000001), -1, 1)
        
        # Step 15
        previous_y = y
        y = y_bar * np.exp((-1 / 2) * (g_y + A_absTmul(x_bar**2,part = p) + 2*(np.log(y_bar) - np.log(y)) - A_absTmul(x**2,part = p)))
        y = y / y.sum()  # Ensure y is a probability vector
        
        # Step 16
        x = np.clip( - (g_x - 2 * x * A_absmul(previous_y, n,part = p)) / (2 * A_absmul(y,n,part = p)+0.000001), -1, 1)
        y_bar = y * np.exp((-1 / 2) * (g_y + A_absTmul(x**2,part = p) + 2*(np.log(y) - np.log(previous_y)) - A_absTmul(x**2,part = p)))
        y_bar = y_bar / y_bar.sum()  # Ensure y is a probability vector    
        #if(t % 200 == 0):
        #    error = (-(y_hat.T)@ATmul(x_hat, part = True)*T/(t+1)/2 - c@x_hat +b@y_hat/2)*T/(t+1)
        #    print(error-0.02)
        #    f.write(str(error-0.02) +'\n')
        #if (t>3000):
        #    break
    return x_hat, y_hat

#A_p = np.zeros((A_.shape[0]+1,A_.shape[1]))
#A_p[0:A_.shape[0]] = A_
# b_p = np.zeros(b_.shape[0]+1)
# b_p[0:b_.shape[0]] = b_

#file = open("output17_0.1.txt", 'w')
#file = open("output1.txt", 'w')
#for i in range(15,16):
for i in range(5,11):
    it = 5
    #for j in [0.01,0.04,0.16,0.64,1]:
    S_calibrated = []
    smCE = np.zeros(it)
    for k in range(it):
        S_calibrated.append(prepare_dataset(2**i+1, lambda x:(x+0.1), 0.1))
        #S_calibrated.append(prepare_dataset(2**i+1, lambda x:(x+0.01), 0.01))
        c_,A_,b_ = smCE_LP(S_calibrated[k])
        n = len(S_calibrated[k][0])
        x = cp.Variable(n)
        objective = cp.Minimize(np.array([c_]) @ x +2*math.log(n)*cp.norm(cp.maximum(A_ @ x-b_, np.zeros(len(A_)) ),"inf"))
        constraints = [-1 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        smCE[k] =-result
    for j in [0.1,0.5,1]:
    #for j in [0.01,0.04,0.16]:
    #for j in [1]:
        error = np.zeros(it)
        for k in range(it):
            (x_list, y_list) = S_calibrated[k]
            n = len(x_list)
            indices = np.argsort(x_list)
            x_list = x_list[indices]
            y_list = y_list[indices]
            b_p = np.abs(ATmul(x_list))
            c_ = (y_list - x_list)/n

            #x, y = box_simplex((n, 4*n-5), b_p, c_, 10/(n**0.5))
            #print(-y.T@ATmul(x)*2 - c_@x +2*b_p@y)
            b_p = np.abs(ATmul(x_list, part = True))
            x, y = box_simplex((n, 2*n-1), b_p, c_, 1/(i*j*n**0.5), p = True, f = None)
            #x, y = box_simplex((n, 2*n-1), b_p, c_, 1/(j), p = True, f = file)
            #error[k] = -y.T@ATmul(x, part = True)*2 - c_@x +b_p@y*2
            #error[k] = -LA.norm((ATmul(x, part = True)-b_p), np.inf)*2 - c_@x
            #error[k] = -np.max(ATmul(x, part = True)-b_p, 0)*2 - c_@x
            error[k] = - c_@x
        print(i)
        print(j)
        print(error)
        print(smCE)
        print(np.mean(np.abs(error-smCE)))
        T = int( 48 * 2 *j*n**0.5)
        print(T)
        #file.write(str(i)+ "\n")
        #file.write(str(j) + "\n")
        #file.write(str(np.mean(error))+ "\n")
print('box simple version 0.1')
