import numpy as np
import util
import sys
sys.path.append('../../DataGlove')
import createDataset
from scipy import optimize
from klampt import resource
from GPy.util.pca import PCA
import math
import GPy
class SGPLVM():
	def __init__(self, Y):
		self.Y = Y
		self.latentDim = 3
	
	def initX(self, Y):
		N = Y[0].shape[0]
		PC = np.empty([N,self.latentDim])
		for Yi in Y:
			p = PCA(Yi)
			PC = np.add(PC, p.project(Yi, min(self.latentDim, Yi.shape[1])))
		PC = np.divide(PC, len(self.Y))
		return PC

	def Lgp(self, x, *args):
		allKernelParams = x
		X, Y, W = args
		ret = 0.
		for i in range(0, len(Y)):
			Yi = Y[i]
			kernelParams = allKernelParams[i*3: (i+1)*3]
			N = Yi.shape[0]
			D = Yi.shape[1]
			Wi = W[i]
			f_Lgp=util.f_Lgp(X, Yi, kernelParams, Wi)
			ret = np.add(ret, f_Lgp)
			f_Lpriorbeta = util.f_Lbeta(Wi, kernelParams, N, D)
			ret = np.add(ret, f_Lpriorbeta)
		f_LpriorX = util.f_LpriorX(X)
		ret = np.add(ret, f_LpriorX)
		return ret

	def Lgp_df(self, x):
		allKernelParams=x
		grad = np.empty([0,0])
		for i in range(0, len(Y)):
			Yi = Y[i]
			kernelParams = allKernelParams[i*3: (i+1)*3]
			D = Yi.shape[1]
			W = np.identity(D)
			grad_beta = util.gradf_beta(X, Yi, kernelParams, W)
			grad = np.append(grad, grad_beta)
		length = len(grad)
		return np.ndarray.flatten(np.asarray(grad))
	
	def Lik(self, x, *args):
		xi = x
		y1i, y2i, X, Y1, Y2, mu_Y1, mu_Y2, W1, W2, kParams1, kParams2 = args
		f_Lik = util.f_Lik(xi,y1i, X, Y1, mu_Y1, W1, kParams1) + util.f_Lik(xi, y2i, X, Y2, mu_Y2, W2, kParams2)
		return f_Lik

	def Lik_df():
		pass

	def optimizeKParams(self, X, Y, W, kernelParams):
		x0 = kernelParams
		args = (X, Y, W)
		delta = 1e-8
		#res1 = optimize.fmin_cg(self.loglikelihood, self.x0, fprime=None)
		cons = ({'type': 'ineq',
				 'fun': lambda x: np.array([x[0] - delta])},
				{'type': 'ineq',
				 'fun': lambda x: np.array([x[1] - delta])},
				{'type': 'ineq',
				 'fun': lambda x: np.array([x[2] - delta])},
				{'type': 'ineq',
				 'fun': lambda x: np.array([x[3] - delta])},
				{'type': 'ineq',
				 'fun': lambda x: np.array([x[4] - delta])},
				{'type': 'ineq',
				 'fun': lambda x: np.array([x[5] - delta])})
		opts = {'maxiter' : None,    # default value.
        'disp' : False,    # non-default value.
        'gtol' : 1e-5,    # default value.
        'norm' : np.inf,  # default value.
        'eps' : 1.4901161193847656e-08}  # default value.
		bnds = ((0, None), (0, None), (0,None), (0, None), (0,None), (0,None))
		res1=optimize.minimize(self.Lgp, x0, jac=None, args=args, method='L-BFGS-B', bounds = bnds, options = opts)
		return res1.x

	def optimizeLatentSpace(self, allKernelParams, YTrain1, YTrain2, mu_Y1, mu_Y2, W1, W2, X):
		N = YTrain2.shape[0]
		newX = np.zeros([N,self.latentDim])
		if X is None:
			X = self.initX([YTrain1, YTrain2])

		for i in range(0, N):
			y1i = YTrain1[i, :]
			y2i = YTrain2[i, :]
			x0 = X[i, :]
			kParams1 = allKernelParams[0:3]
			kParams2 = allKernelParams[3:]
			opts = {'maxiter' : None,    # default value.
        	'disp' : False,    # non-default value.
        	'gtol' : 1e-5,    # default value.
        	'norm' : np.inf,  # default value.
        	'eps' : 1.4901161193847656e-08}  # default value.
			D = YTrain1.shape[1]
			args = (y1i, y2i, X, YTrain1, YTrain2, mu_Y1, mu_Y2, W1, W2, kParams1, kParams2)
			res1 = optimize.minimize(self.Lik, x0, jac=None, args=args, method='L-BFGS-B', options = opts) 
			x_pred = res1.x
			newX[i, :] = x_pred
		return newX
	
	def updateW(self, X, Y, kernelParams):
		N = Y.shape[0]
		D = Y.shape[1]
		K = util.RBF(X,X,kernelParams)
		K_ = np.linalg.inv(K)
		W = np.identity(D)
		for k in range(0, D):
			yk = Y[:, k]
			ykT = np.transpose(yk)
			denom = np.dot(np.dot(ykT, K_), yk)
			val = math.sqrt(N/denom)
			W[k,k] = val
		return W
			
	
	def learn(self, YTrain1, YTrain2, mu_Y1, mu_Y2):
		maxIters = 1
		W1 = np.identity(YTrain1.shape[1])
		W2 = np.identity(YTrain2.shape[1])
		kParams = [1,1,1,1,1,1]
		X = self.initX([YTrain1, YTrain2])
		N = YTrain1.shape[0]
		for i in range(0,maxIters):
			print 'iter: ', i
			Y = [YTrain1, YTrain2]
			W = [W1, W2]
			kParams = self.optimizeKParams(X, Y, W, kParams)
			newX = self.optimizeLatentSpace(kParams, YTrain1, YTrain2, mu_Y1, mu_Y2,  W1, W2, X)
			#W1 = self.updateW(newX, YTrain1, kParams[0:3])
			#W2 = self.updateW(newX, YTrain2, kParams[3:])
			X = newX
		baseConfigs = np.zeros([N, 6])
		YTrain1 = np.concatenate((baseConfigs, YTrain1), axis=1)
		YTrain2 = np.concatenate((baseConfigs, YTrain2), axis=1)
		#TODO: writefile
		resource.set('YTrain1.configs', YTrain1)
		resource.set('YTrain2.configs', YTrain2)

		return kParams, X, W1, W2

	def predict( self, kParams, X, YTrain1, YTest1, YTest2, W1, W2, mu_Y1, mu_Y2):
		newX = self.optimizeLatentSpace(kParams, YTest1, YTest2, mu_Y1, mu_Y2, W1, W2, None)
		N = YTest1.shape[0]
		Pred = []
		m = GPy.models.GPRegression(YTrain1, X)
		m.optimize('bfgs', max_iters=200)
		for i in range(0, N):
			yi = YTest1[i, :].reshape([1,21])
			xi = m.predict(yi)[0][0]
			K = util.RBF(X,X, kParams[3:])
			f_x = util.f_x(xi, X, K, YTrain2, mu_Y2)
			var_x = util.var_x(xi,X, K)
			f_x = np.reshape(f_x, [1, len(f_x)])
			baseConfigs = np.zeros([1, 6])
			f_x = np.concatenate((baseConfigs, f_x), axis=1)
			#print 'f_x: ', f_x
			Pred.append(f_x[0,:])
		#self.addConstraint(Pred)
		resource.set('Pred.configs', Pred)

	def addConstraint(self, Pred):
		for i in range(0, len(Pred)):
			N = len(Pred[0])
			for j in range(6, N):
				Pred[i][j] = self.limit(Pred[i][j], j)

	def limit(self, val, i):
		lim = {6:(0,0), 7:(0,0), 8:(0, math.pi/2), 9:(0, 2.83), 10:(0,0), 11:(0,0), 12:(0,0), 13:(-math.pi/2, 0), 14:(0, 2.83), 15:(0,0), 16:(0,0), 17:(0,0), 18:(-0.34, 2.83), 19:(0,0), 20:(0,0), 21:(0,0)}
		if val < lim[i][0]:
			val = lim[i][0]
		if val > lim[i][1]:
			val = lim[i][1]
		return val
	
def meanCenter(Y):
	mu = Y.mean(0)
	for i in range(0, Y.shape[0]):
		Y[i,:] = Y[i,:] - mu
	return mu	

if __name__ == '__main__':
	YTrain1,YTrain2 = createDataset.makeDataset('train')
	mu_Y1Train = meanCenter(YTrain1)
	mu_Y2Train = meanCenter(YTrain2)	
	Y = [YTrain1, YTrain2]
	model = SGPLVM(Y)
	kParams,X, W1, W2 = model.learn(YTrain1, YTrain2, mu_Y1Train, mu_Y2Train)
	YTest1, YTest2 = createDataset.makeDataset('test')
	mu_Y1Test = meanCenter(YTest1)
	mu_Y2Test = meanCenter(YTest2)
	model.predict(kParams, X, YTrain1, YTest1, YTest2, W1,W2, mu_Y1Test, mu_Y2Test)
	
