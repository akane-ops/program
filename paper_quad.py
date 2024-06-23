import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import time
import csv
import pprint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

class Lorenz:
    def __init__(self, param):
        self.sigma = param[0]
        self.r = param[1]
        self.b = param[2]
        
    def Lorenz(self, x, y, z):
        return np.array([-self.sigma*x+self.sigma*y, -x*z+self.r*x-y, x*y-self.b*z])
    
    def function(self, vec):
        return self.Lorenz(vec[0], vec[1], vec[2])
    
    def RK(self, x0, dt):
        vec = x0
        k1 = self.function(vec)
        k2 = self.function(vec + dt * k1/2 * np.ones(len(vec)))
        k3 = self.function(vec + dt * k2/2 * np.ones(len(vec)))
        k4 = self.function(vec + dt * k3 * np.ones(len(vec)))
        return vec + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


if __name__ == '__main__':
    def makeWin(K, N, eta, seed):
        np.random.seed(seed=seed)
        Win = np.random.uniform(-eta, eta, (N, K))
        return Win
    
    def makeW(N, rho, seed):
        re_scale = 1.0
        np.random.seed(seed=seed)
        W = np.random.uniform(-re_scale, re_scale, (N, N))
        
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))
        
        W *= rho / sp_radius
        
        return W

    def predict(Win, W, Wout, u, r):
        x = np.dot(Win, u)
        r = act_func(np.dot(W, r) + x)
        s = []
        for i in range(N):
            for j in range(i, N):
                s.append(r[i] * r[j])
        r_quad = np.concatenate([r,np.array(s)])
        y_pred = np.dot(Wout, r_quad)
            
        return y_pred, r


    start = time.time()
    

    T_train = 1e2
    T_test = 1e2
    
    dt = 0.02

    X0 = np.array([1,1,1])

    param = [10.0, 28.0, 8.0/3.0]

    dynamics = Lorenz(param)
    
    N = 10
    K = len(X0)
    L = K
    
    train_len = int(T_train/dt)
    test_len = int(T_test/dt)

    eta = 0.01
    rho = 0.01
    beta_1 = 1e-6
    seed = 0
    act_func = np.tanh

    Win = makeWin(K, N, eta, seed)
    W = makeW(N, rho, seed)
    
    r = np.zeros(N)
    MSEs = []
    train_T = []
    trans_time = int(100/dt)
    
    Rt_R = np.zeros((int(N+N*(N+1)/2), int(N+N*(N+1)/2)))
    R_Yt = np.zeros((int(N+N*(N+1)/2), L))
    
    for _ in range(trans_time):
        X1 = dynamics.RK(X0, dt)
        x = np.dot(Win, X0)
        r = act_func(np.dot(W, r) + x)
        X0 = X1
    for _ in range(train_len):
        X1 = dynamics.RK(X0, dt)
        x = np.dot(Win, X0)
        r = act_func(np.dot(W, r) + x)
        y = X1
        s = []
        for k in range(N):
            for l in range(k, N):
                s.append(r[k] * r[l])
        r_quad = np.concatenate([r,np.array(s)])
        r_quad = np.reshape(r_quad, (1, -1))
        y = np.reshape(y, (1, -1))
        Rt_R += np.dot(r_quad.T, r_quad)
        R_Yt += np.dot(r_quad.T, y)
        
        X0 = X1
        
    beta = beta_1 * train_len
    Rt_R_ridge = Rt_R + beta * np.identity(N+int(N*(N+1)/2))
    Wout_solve = np.linalg.solve(Rt_R_ridge, R_Yt)
    Wout = Wout_solve.T

    y_pred, r = predict(Win, W, Wout, X1, r)
    
    test = dynamics.RK(X1, dt)
    
    error = 0
    Y_pred = np.zeros((test_len, L+1))
    for j in range(test_len):
        Y_pred[j,0:3] = y_pred
        err = y_pred - test
        err = np.dot(err,err) ** 0.5
        Y_pred[j, 3] = err
        test = dynamics.RK(y_pred, dt)
        
        y_pred, r = predict(Win, W, Wout, y_pred, r)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['font.size'] = 13
    mpl.rcParams['axes.xmargin'] = 0.0
    mpl.rcParams['axes.ymargin'] = 0.05
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='3d')
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.grid(False)
    cm = plt.cm.get_cmap("coolwarm")
    for i in range(len(Y_pred)):
        plt.plot(Y_pred[i:i+2,0], Y_pred[i:i+2,1], Y_pred[i:i+2,2], color=cm((np.log10(Y_pred[i,3])+3)/3), lw=0.7)
    mappable = ax.scatter([Y_pred[:len(Y_pred),0]],[Y_pred[:len(Y_pred),1]],[Y_pred[:len(Y_pred),2]],c=Y_pred[:len(Y_pred),3], vmin=-3, vmax=0, s=-1,cmap=cm)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    cax = fig.add_axes((0.9, 0.05, 0.01, 0.3))
    fig.colorbar(mappable, ax=ax, cax=cax)
    
    plt.show()

