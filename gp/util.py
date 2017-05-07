from __future__ import division
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math
import bigfloat
bigfloat.exp(5000, bigfloat.precision(100))
def f_Lgp(X, Y, kernelParams, W):
	'''
	Y is observation
	'''
	alpha, beta, gamma = kernelParams
	K = RBF(X, X, kernelParams)
	K_ = np.linalg.inv(K)
	D = Y.shape[1] # dimensions of feature vector
	N = Y.shape[0]
	a = np.dot(D/2, np.log(np.linalg.det(K)))
	b = 0
	for k in range(0, D):
		Yk = Y[:, k]
		wk = W[k, k]
		wk2 = wk*wk
		b += np.dot(np.dot(np.dot(wk2, np.transpose(Yk)), K_), Yk)
	b *= 0.5
	return np.add(a,b)
def f_Lik(x, y, X, Y, mu, W, kernelParams):
	K = RBF(X,X,kernelParams)
	D = Y.shape[1]
	sqeuclidean = lambda x: np.inner(x,x)
	var = var_x(x,X,K)
	return sqeuclidean(np.dot(W, np.subtract(y, f_x(x,X, K, Y, mu))))/(2*var) + (D/2) * math.log(var) + 0.5 * sqeuclidean(x)	
def f_LpriorX(X):
	ret = 0
	sqeuclidean = lambda x: np.inner(x,x)
	for i in range(0, X.shape[0]):
		ret += sqeuclidean(X[i,:])
	ret *= 0.5
	return ret

def f_Lbeta(W, kernelParams, N, D):
	alpha, beta, gamma = kernelParams
	w = 1
	for k in range(0, D):
		wk = W[k,k]
		w *= math.pow(wk, N)
		#alpha = abs(alpha)
		#beta = abs(beta)
		#gamma = abs(gamma)
	#print alpha, ' ', beta, ' ', gamma, ' ', w
	return math.log((alpha*beta*gamma)/w)

def gradf_LgpK(X, Y, kernelParams, W):
	K = RBF(X,X, kernelParams)
	K_ = np.linalg.inv(K)
	D = Y.shape[1]
	a = np.dot(K_, Y)
	b = np.dot(a, W)
	c = np.dot(b, W)
	d = np.dot(c, np.transpose(Y))
	return np.dot(np.dot(np.dot(np.dot(np.dot(K_, Y), W), W), np.transpose(Y)), K_) + np.dot(K_, D)
'''
Prediction: predicts Y given x
'''
def f_Lx(x,y, X, Y, kernelParams, W, f_x, var_x):
	'''
	Y is mean centered
	'''
	K = RBF(X,Y, kernelParams)
	D = Y.shape[1]
	sqeuclidean = lambda x: np.inner(x,x)
	term1 = sqeuclidean(np.dot(W, np.subtract(y, f_x))) / (2* var_x) 
	term2 = np.dot(D/2, np.log(var_x))
	term3 = np.multipy(0.5, sqeuclidean(x))
	return term1 + term2 + term3


def f_x(x, X,K, Y, mu):
	#mu = Y.mean(0) 
	assert (len(mu) == Y.shape[1])
	K_ = np.linalg.inv(K)
	idx = findIdx(x, X)
	#print 'idx: ', idx
	if idx < 0:
		idx = findClosest(x,X)
		#print 'new idx: ', idx
	kx = K[idx, :]
	return np.add(mu, np.dot(np.dot(np.transpose(Y), K_), kx))

def var_x(x, X, K):
	idx = findIdx(x,X)
	kx = K[:, idx]
	kxx = K[idx, idx]
	K_ = np.linalg.inv(K)
	a = np.dot(np.dot(np.transpose(kx), K_), kx)
	ret = kxx - a
	if ret == 0:
		ret = 0.00000001
	return abs(ret)

def gradf_Lx_x(x, y, Y, W, f_x, gradf_fx_x, var_x, gradf_var_x):
	sqeuclidean = lambda x: np.inner(x,x)
	Wsquared = np.dot(W, W)
	D = Y.shape[1]
	term1 = np.divide(np.substract(y, f_x),var_x)
	term2 = np.dot(np.dot(-np.transpose(gradf_fx_x),Wsquared), term1)
	term3 = sqeuclidean(np.dot(W, np.subtract(y, f_x)))/var_x
	term4 = np.dot(gradf_var_x , (D - term3)/(2 * var_x))
	return term2 + term4 + x
	
def grad_Lx_y(y, W, f_x, var_x):
	Wsquared = np.dot(W,W)
	return np.divide(np.dot(Wsquared, np.subtract(y, f_x)), var_x)

def gradf_fx_x(x, Y, K, gradf_k_x):
	K_ = np.linalg.inv(K)
	return np.dot(np.dot(np.transpose(Y), K_), gradf_k_x)

def gradf_var_x(x, X, K, gradf_k_x):
	idx = findIdx(x,x)
	kx = k[:, idx]
	K_ = np.linalg.inv(K)
	return np.dot(np.dot(np.multiply(-2, np.transpose(kx)), K_),gradf_k_x)

def gradf_K_X(X, Y, K, alpha):
	'''
	gradient of L_gp wrt X
	'''
	K_ = np.linalg.inv(K)
	YY = np.dot(Y,np.transpose(Y))
	D = Y.shape[1]

	grad = np.dot(alpha, np.dot(K_, np.dot(YY, np.dot(K_, X)))) - np.dot(alpha, np.dot(D, np.dot(K_, X)))
	return grad.flatten()
		
def gradf_k_x(x, X, Y, K, gamma):
	N = K.shape[0]
	M = K.shape(1)
	gradf_k_x = np.zeros([N, M])
	for i in (0, N):
		x_ = X[i, :]
		gradf_k_x[i, :] = gradf_k_xx(x, x_, K, gamma)
	return gradf_k_x
	
	
def gradf_k_xx(x1, x2, X, K, gamma):
	idx1 = findIdx(x1, X)
	idx2 = findIdx(x2, X)
	return np.dot(np.dot(-gamma, np.subtract(x1, x2)), K[idx1, idx2])


def gradf_k_alpha(x1, x2, gamma):
	sqeuclidean = lambda x: np.inner(x,x)
	return math.exp( np.dot(-gamma/2, sqeuclidean((np.subtract(x1,x2)))))

def gradf_K_alpha(X, gamma):
	gradf_K_alpha = np.zeros([X.shape[0], X.shape[0]])
	for i in range(0, X.shape[0]):
		for j in range(0, X.shape[0]):
			x1 = X[i, :]
			x2 = X[j, :]
			gradf_K_alpha[i,j] = gradf_k_alpha(x1,x2, gamma)
	return gradf_K_alpha

def gradf_Kalpha(X, Y, kernelParams, W):
	alpha, beta, gamma = kernelParams
	grad_LgpK = gradf_LgpK(X,Y, kernelParams, W)
	grad_K_alpha = gradf_K_alpha(X, gamma)
	grad =  np.dot(grad_LgpK, grad_K_alpha)
	return grad.flatten()


def gradf_k_beta(x1,x2):
	if np.array_equal(x1, x2):
		return 1
	else:
		return 0

def gradf_K_beta(X):
	gradf_K_beta = np.zeros([X.shape[0], X.shape[0]])
	for i in range(0, X.shape[0]):
		for j in range(0, X.shape[0]):
			x1 = X[i, :]
			x2 = X[j, :]
			gradf_K_beta[i,j] = gradf_k_beta(x1,x2)
	return gradf_K_beta

def gradf_Kbeta(X,Y,kernelParams, W):
	grad_K_beta = gradf_K_beta(X)
	grad_LgpK = gradf_LgpK(X,Y,kernelParams, W)
	grad = np.dot(grad_LgpK, grad_K_beta)
	return grad.flatten()

def gradf_k_gamma(x1,x2, kx1x2):
	sqeuclidean = lambda x: np.inner(x,x)
	return -0.5 * np.dot(sqeuclidean((x1-x2)), kx1x2)

def gradf_K_gamma(X, K):
	gradf_K_gamma = np.zeros([X.shape[0], X.shape[0]])
	for i in range(0, X.shape[0]):
		for j in range(0, X.shape[0]):
			x1 = X[i, :]
			x2 = X[j, :]
			kx1x2 = K[i,j]
			gradf_K_gamma[i,j] = gradf_k_gamma(x1,x2, kx1x2)
	return gradf_K_gamma

def gradf_Kgamma(X,Y,kernelParams, W):
	K = RBF(X,X, kernelParams)
	grad_LgpK = gradf_LgpK(X,Y,kernelParams, W)
	grad_K_gamma = gradf_K_gamma(X, K)
	grad = np.dot(grad_LgpK, grad_K_gamma)
	return grad.flatten()

def gradf_beta(X,Y,kernelParams,W):
	alpha = gradf_Kalpha(X,Y,kernelParams, W)
	beta = gradf_Kbeta(X,Y,kernelParams, W)
	gamma = gradf_Kgamma(X,Y,kernelParams, W)
	return np.asarray((alpha, beta, gamma))


'''
Helper functions
'''

def findIdx(x, X):
	'''
	Returns index of x in X
	'''
	for i in range(0,X.shape[0]):
		row = X[i, :]
		if np.array_equal(row, x):
			return i
	return -1
	
	
def RBF(X1, X2, kernelParams):
	alpha, beta, gamma = kernelParams
	X1, X2 = check_pairwise_arrays(X1, X2)
	k = euclidean_distances(X1,X2, squared=True)
	c = gamma * -0.5
	k = np.multiply(k, c)
	return alpha * np.exp(k) + np.dot(delta(X1,X2), 1/beta)


def delta(X1, X2):
	result = np.zeros((X1.shape[0], X2.shape[0]), dtype=X1.dtype)
	for i in range(0, X1.shape[0]):
		if np.array_equal(X1[i,:], X2[i, :]):
			result[i, i] = 1
	return result
	
def findClosest(x,X):
	mindist = 1e8
	idx = 0
	for i in range(0, X.shape[0]):
		xi = X[i, :]
		dist = np.linalg.norm(np.subtract(xi, x))
		if dist < mindist:
			mindist = dist
			idx = i
	return idx	 

