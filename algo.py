import numpy as np
import scipy as scipy
from scipy import optimize
from functools import partial


class Analyzer(object):
	def __init__(self, numUser=50, numRating=6, numHiddenFeatures = 5, weightDecay = 1.):
		self.k = 5
		self.U = numUser
		self.V = numRating
		self.Lambda = weightDecay
		
		self.P = self.initParam((self.U, self.k))
		self.W = self.initParam((self.k, self.U, self.V))
		
	def initParam(self, shape):
		return np.random.standard_normal(shape)*0.1
		
	def foldParam(self, P, W):
		return np.concatenate((P.reshape(-1), W.reshape(-1)))
		
	def vecParam(self):
		return np.concatenate((self.P.reshape(-1), self.W.reshape(-1)))

	def unvecParam(self, Theta):
		P, W = self.unfoldParam(Theta)
		self.P = P
		self.W = W
		
	def unfoldParam(self, Theta):
		ind=0
		P = Theta[ind:ind+(self.U*self.k)].reshape(self.U, self.k)
		ind = ind + self.U*self.k
		W = Theta[ind:ind+(self.k*self.U*self.V)].reshape(self.k, self.U, self.V)
		
		return P, W
		
	def prodM(self, R1, R2):
		shape1 = R1.shape
		shape2 = R2.shape
		R = np.zeros((shape1[0],shape2[1],shape2[2]))
		for i in range(shape2[2]):
			R[:,:,i] = R1.dot(R2[:,:,i])
			
		return R
		
	def prodMM(self, R1, R2):
		shape1 = R1.shape
		shape2 = R2.shape
		R = np.zeros((shape1[0], shape2[1], shape1[2]))
		for i in range(shape1[2]):
			R[:,:,i] = R1[:,:,i].dot(R2[:,:,i])
		
		return R
		
		
	def costFunc(self, Theta, R, mask):
		P,W = self.unfoldParam(Theta)
		
		diff = (self.prodM(P,W) - R)*mask
		J = 1./2.*(diff**2).sum() + self.Lambda/2.*((P**2).sum() + (W**2).sum())
		
		P_grad = self.prodMM(diff, W.transpose(1,0,2)).sum(2) + self.Lambda*P
		W_grad = self.prodM(P.T, diff) + self.Lambda*W
	
		Theta_grad = self.foldParam(P_grad, W_grad)
		
		return J, Theta_grad
		
	def getPDMatrix(self):
		D = np.zeros((self.U, self.U))
		
		for i in range(self.U):
			D[i] = np.sqrt(((self.P - self.P[i])**2).sum(1))
			
		return D


	def trainModel(self, R, mask, maxfun=100, iprint=1.):
		f = partial(self.costFunc, R=R, mask=mask)
		Theta, cost, d = optimize.fmin_l_bfgs_b(f, 
											 self.vecParam(), 
											 maxfun=maxfun, 
											 iprint = iprint)
		
		self.unvecParam(Theta)
		return cost




###################################################
## utility



def computeNumericalGradient(J, Theta):
    numgrad = np.zeros(Theta.shape)
    perturb = np.zeros(Theta.shape)
    e = 1e-4
    for p in range(Theta.shape[0]):
        # Set perturbation vector
        perturb[p] = e;
        loss1,tmp = J(Theta - perturb);
        loss2,tmp = J(Theta + perturb);
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;

    return numgrad

def checkgrad(func, Theta0):
    cost, grad = func(Theta0)

    #grad = gnp.as_numpy_array(grad)

    numgrad = computeNumericalGradient(func, Theta0)

    diff = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)

    print 'diff : ',diff
    return grad,numgrad



		
		