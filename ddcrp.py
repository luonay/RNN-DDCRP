import random
from scipy.spatial import distance
from scipy import special
import numpy as np
from itertools import repeat

batch_size = 10 #number of hidden representations fed into ddcrp
alpha = 0.5 #concentration parameter
#distance matrix for all hidden representations
dim = 15 #dimenson of hid rep
h_vec = np.random.rand(batch_size, dim)
h_mean = h_vec.mean(1)
dists = scipy.spatial.distance.pdist(h_vec, 'euclidean')

class DdcrpSampler(object):

#1: initialize hyperparameters
  def __init__ (self, h_vec):
    dim = h_vec.shape[1]
    mu_0 = np.empty(dim)
    mu_0.fill(0)
    kappa_0 = 1
    diagEntry = np.empty(dim)
    diagEntry.fill(0.1)
    Lambda_0 = np.diag(diagEntry)
    nu_0 = -0.5

#2: define marginal distribution of the samples from a particular cluster 
# h_k: matrix of h_i from kth cluster. dim1: index i  dim2: h_k[i] 
  def ClusterLoglh(self, n_k, h_k)
    assert n_k = h_k.shape[0]
    const = np.log(1 / np.power(np.pi, n_k*dim/2))
    nu_n = nu_0 + n_k
    kappa_n = kappa_0 + n_k
    Lambda_n = Lambda_0 + CSSP(h_k) + kappa_0 * n_k / (kappa_0+n_k) * np.outer(h_k.mean(0) - mu_0)
    loglh = const + Gamma(nu_n / 2, dim) - Gamma(nu_0 / 2, dim) + nu_0 / 2 * np.log(np.linalg.det(Lambda_0)) - nu_n / 2 * np.log(np.linalg.det(Lambda_n)) + dim / 2 * (np.log(kappa_0) - np.log(kappa_n))
    return loglh
    
#helper function
  def CSSP(self, h):
    sum = np.zeros((dim, dim))
    for i in np.arange(dim):
      sum += np.outer(h[i], h[i]) 
    return sum

  def Gamma(self, d, n):
    return scipy.special.multigammaln(n, d)

  def DecayedDist(self, i, j): # i < j
    index = (batch_size - i/2 - 3/2) * i + j
    return np.exp(-dists[index])

  def GetRoot(i):
    while True:
      if parent[i] = i:
        return i
      i = parent[i]

  def GetChild(i):
    if c_child[i] = []:
      return []
    else:
      return c_child[i] + [item for sublist in [GetChild[x] for x in c_child] for item in sublist]
      
  def GetCluster(i):
    root = GetRoot(i)
    return [root] + GetChild(root)

   

#Total decayed distance: denominator of prior probability
  totalDecayDist = np.zeros(batch_size)

  for i in range(0, batch_size):
    for j in range(i):
      totalDecayDist[i] += DecayedDist(j,i)

  def Run(self):
#3: Sampling step 1: chinese restaurant process to initialize assignments c
    c_parent = range(batch_size) #link of each element
    c_child = [[] for i in repeat(None, batch_size)] #list of children linked to element
    for i in range(0,batch_size):
      rand = np.random.rand()
      prob = alpha
      if rand < prob / (totalDecayedDist[i] + alpha):
        continue
      for j in range(i):
        prob += DecayedDist(j,i) 
        if rand < prob / (totalDecayedDist[i] + alpha):
          c_parent[i] = j
          c_child[j].append(i)
        break

#3: Sampling step 2: Gippings sampling with full conditional distribution 
    for Iter in range(max_iter):
      for i in range(batch_size):
        rand = np.random.rand()
        prob = alpha
        log_cond_prob = np.empty(i+1)
        log_cond_prob[i] = np.log(alpha)
        #delete the link from i
        c_child[c_parent[i]].remove(i)
        c_parent[i] = i
        #calculate conditional probability
        for j in range(i):
          log_cond_prob[i] = np.log(DecayedDist(j,i)) + ClusterLoglh(GetCluster(i) + GetCluster(j)) - ClusterLoglh(GetCluster(i)) - ClusterLoglh(GetCluster(j))
          normal_const = np.exp(log_cond_prob).sum()
        #sample c_i
        if rand < prob / normal_const: #linked to itself
          continue
        else:
          for j in range(i):
            prob += np.exp(log_cond_prob[i])
            if rand < prob / normal_const:
              c_parent[i] = j
              c_child[j].append[i]
              continue





















