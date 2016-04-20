import random
from scipy.spatial import distance
from scipy import special
import numpy as np
from itertools import repeat

class Ddcrp(object):
#1: initialize hyperparameters
#FIXME: read parameters from prototxt
  def __init__ (self, h_mat, alpha):
    self.h_mat = h_mat
    self.alpha = alpha
    self.dim = self.h_mat.shape[1]
    self.batch_size = self.h_mat.shape[0]
    self.mu_0 = np.empty(self.dim)
    self.mu_0.fill(0)
    self.kappa_0 = 1
    diagEntry = np.empty(self.dim)
    diagEntry.fill(0.1)
    self.Lambda_0 = np.diag(diagEntry)
    self.nu_0 = self.dim + 1 #Wishart distribution requires that d.f. > dim - 1
    self.dists = distance.pdist(self.h_mat, 'euclidean')
    self.decayedDists = np.exp(-self.dists)
    self.c_parent = range(self.batch_size) #link of each element
    self.c_child = [[] for i in repeat(None, self.batch_size)] #list of children linked to element
    #log full conditional distribution for each element's assignment
    self.log_fullcond = []
    for i in range(self.batch_size):
      self.log_fullcond.append(np.empty(i+1))
      self.log_fullcond[i][i] = np.log(self.alpha)
    self.normal_const = np.empty(self.batch_size)
    self.totalDecayedDists = np.full(self.batch_size, self.alpha)
#helper function
  def CSSP(self, h):
    assert h.shape[1] == self.dim
    cssp = np.zeros((self.dim, self.dim))
    mean = h.mean(0)
    for i in np.arange(h.shape[0]):
      cssp += np.outer(h[i] - mean, h[i] - mean) 
    return cssp

  def Gamma(self, n, d):
    return special.multigammaln(n, d)

  def GetDist(self, i, j): # i < j
    assert i <= j
    if i == j:
      return -np.log(self.alpha)
    else:
      index = (self.batch_size - i/2.0 - 1/2.0) * i + j - i - 1
      return self.dists[index]

  def GetDecayedDist(self, i, j): # i < j
    assert i <= j
    if i == j:
      return self.alpha
    else:
      index = (self.batch_size - i/2.0 - 1/2.0) * i + j - i - 1
      return self.decayedDists[index]

  def GetRoot(self, i):
    while True:
      if self.c_parent[i] == i:
        return i
      i = self.c_parent[i]

  def GetChild(self, i):
    if self.c_child[i] == []:
      return []
    else:
      return self.c_child[i] + [item for sublist in [self.GetChild(x) for x in self.c_child[i]] for item in sublist]
      
  def GetCluster(self, i):
    root = self.GetRoot(i)
    return [root] + self.GetChild(root)

#Total decayed distance: denominator of prior probability
  def UpdateTotalDecayedDists(self):
    for i in range(self.batch_size):
      for j in range(i + 1):
        self.totalDecayedDists[i] += self.GetDecayedDist(j, i)

#2: define marginal distribution of the samples from a particular cluster 
# h_k: matrix of h_i from kth cluster. dim1: index i  dim2: h_k[i] 
  def ClusterLoglh(self, h, cluster):
    h_mat_k = h[cluster]
    n_k = h_mat_k.shape[0]
    const = np.log(1 / np.power(np.pi, n_k*self.dim/2))
    nu_n = self.nu_0 + n_k
    kappa_n = self.kappa_0 + n_k
    Lambda_n = self.Lambda_0 + self.CSSP(h_mat_k) + self.kappa_0 * n_k / (self.kappa_0+n_k) * np.outer(h_mat_k.mean(0) - self.mu_0, h_mat_k.mean(0) - self.mu_0)
    loglh = const + self.Gamma(nu_n / 2, self.dim) - self.Gamma(self.nu_0 / 2, self.dim) + self.nu_0 / 2 * np.log(np.linalg.det(self.Lambda_0)) - nu_n / 2 * np.log(np.linalg.det(Lambda_n)) + self.dim / 2 * (np.log(self.kappa_0) - np.log(kappa_n))
    return loglh

  def UpdateFullCond(self, i):
    for j in range(i):
      self.log_fullcond[i][j] = np.log(self.GetDecayedDist(j,i)) + self.ClusterLoglh(self.h_mat, self.GetCluster(i) + self.GetCluster(j)) - self.ClusterLoglh(self.h_mat, self.GetCluster(i)) - self.ClusterLoglh(self.h_mat, self.GetCluster(j))
    self.normal_const[i] = np.exp(self.log_fullcond[i]).sum()
 
  def PrintClusters(self):
    count = 1
    for i in range(self.batch_size):
      if self.c_parent[i] != i:
        pass
      else:
        cur_cluster = self.GetCluster(i)
        print "cluster ", count, ":", cur_cluster
        count += 1
    print count - 1, " clusters in total."

    
  def InitializeClusters(self):
#3: Sampling step 1: chinese restaurant process to initialize assignments c
    for i in range(self.batch_size):
      rand = np.random.rand()
      prob = self.alpha
      if rand < prob / self.totalDecayedDists[i]:
        continue
      for j in range(i):
        prob += self.GetDecayedDist(j,i) 
        if rand < prob / self.totalDecayedDists[i]:
          self.c_parent[i] = j
          self.c_child[j].append(i)
        break
    self.PrintClusters()
#FIXME: Gradient checking to be finised.

  '''
  def ObjectiveFunction(self, h_mat):
    cur_dists = distance.pdist(h_mat, 'euclidean')
    cur_decayedDists = np.exp(-cur_dists)
    def GetDist(self, i, j): # i < j
      assert i <= j
      if i == j:
        return -np.log(self.alpha)
      else:
        index = (self.batch_size - i/2.0 - 1/2.0) * i + j - i - 1
        return cur_dists[index]
    def GetDecayedDist(self, i, j): # i < j
      assert i <= j
      if i == j:
        return self.alpha
      else:
        index = (self.batch_size - i/2.0 - 1/2.0) * i + j - i - 1
        return cur_decayedDists[index]
    sum_1 = 0
    for i in range(self.batch_size):
      sum_i = 0
      for j in range(i):
        sum_i += GetDecayedDist(j, i)
      sum_1 += - GetDist(i, self.c_parent[i]) - np.log(sum_i)
    sum_2 = 0
    for i in range(self.batch_size):
      if self.c_parent[i] != i:
        pass
      else:
        cur_cluster = self.GetCluster(i)
        sum_2 += self.ClusterLoglh(h_mat, cur_cluster)
    total = sum_1 + sum_2
    return total

  def CheckGradient(self, i, j):
    h = self.h_mat
    epsilon = 0.0000001
    h[i][j] += epsilon
    of1 = self.ObjectiveFunction(self.h_mat)
    of2 = self.ObjectiveFunction(h)
    deriv = (of2 - of1) / epsilon
    return deriv
  '''
#3: Sampling step 2: Gippings sampling with full conditional distribution 
  def GippsSampling(self):
    for i in range(self.batch_size):
      #delete the link from i
      if self.c_child[self.c_parent[i]].count(i) == 0:
        pass
      else:
        self.c_child[self.c_parent[i]].remove(i)
      self.c_parent[i] = i
      #update full conditional probability
      self.UpdateFullCond(i)
      #sample c_i
      rand = np.random.rand()
      prob = self.alpha
      if rand < prob / self.normal_const[i]: #linked to itself
        continue
      else:
        for j in range(i):
          prob += np.exp(self.log_fullcond[i][j])
          if rand < prob / self.normal_const[i]:
            self.c_parent[i] = j
            self.c_child[j].append(i)
            break 
    self.PrintClusters()

#Helper function for gradient descent
  def D2hDeriv(self, q, p, i, j):
    if (q != i and q != j) or (q == i and q == j):
      return 0
    elif q == i:
      return 2 * (self.h_mat[i][p] - self.h_mat[j][p]) * self.h_mat[i][p] / self.GetDist(min(i,j), max(i,j))
    else:
      return 2 * (self.h_mat[j][p] - self.h_mat[i][p]) * self.h_mat[j][p] / self.GetDist(min(i,j), max(i,j))
  
  """
  def L1Deriv(self, q, p):
    deriv = 0
    for i in np.arange(self.batch_size):
      local_sum = 0
      for j in np.arange(i + 1):
        local_sum += self.GetDecayedDist(j, i) * self.D2hDeriv(q, p, i, j)
      deriv += -self.D2hDeriv(q, p, i, self.c_parent[i]) + local_sum / self.totalDecayedDists[i]
    return deriv
  """

  def L1Deriv(self):
    deriv1 = np.zeros((self.batch_size, self.dim))
    for q in range(self.batch_size):
      for p in range(self.dim):
        for i in np.arange(self.batch_size):
          local_sum = 0
          for j in np.arange(i + 1):
            local_sum += self.GetDecayedDist(j, i) * self.D2hDeriv(q, p, i, j)
          deriv1[q][p] += -self.D2hDeriv(q, p, i, self.c_parent[i]) + local_sum / self.totalDecayedDists[i]
    return deriv1


  def L2Deriv(self):
    deriv2 = np.zeros((self.batch_size, self.dim))
    for i in range(self.batch_size):
      if self.GetRoot(i) != i:
        pass
      else:
        cluster_cur = self.GetCluster(i)
        n_k = len(cluster_cur)
        nu_n_k = self.nu_0 + n_k
        h_cur = self.h_mat[cluster_cur]
        h_cur_mean = h_cur.mean(0)
        Lambda_n_k = self.Lambda_0 + self.CSSP(h_cur) + self.kappa_0 * n_k / (self.kappa_0+n_k) * np.outer(h_cur_mean - self.mu_0, h_cur_mean - self.mu_0)
        for p in range(self.dim):
          B = np.zeros((self.dim, self.dim))
          B[p] += (h_cur_mean - self.mu_0) / n_k
          B[:, p] += (h_cur_mean - self.mu_0) / n_k
          for q in cluster_cur:
            A = np.zeros((self.dim, self.dim))
            A[p] += self.h_mat[q] - h_cur_mean
            A[:, p] += self.h_mat[q] - h_cur_mean
            LambdaDeriv = A + (self.kappa_0 * n_k) / (self.kappa_0 + n_k) * B
            deriv2[q][p] = -nu_n_k / 2 * np.trace(np.dot(np.linalg.inv(Lambda_n_k), LambdaDeriv))      
    return deriv2               
    
  def BatchGD(self):
    deriv_total = np.zeros(self.dim)
    l1deriv = np.zeros((self.batch_size, self.dim))
    l2deriv = np.zeros((self.batch_size, self.dim))
    l1deriv = self.L1Deriv()
    l2deriv = self.L2Deriv()
    deriv_total = l1deriv + l2deriv
    print "l1deriv: ", l1deriv
    print "l2deriv: ", l2deriv
    return deriv_total
    
  def EM(self, max_iter):
    self.UpdateTotalDecayedDists()
    self.InitializeClusters()
    for Iter in range(max_iter):
      self.GippsSampling()
      deriv = self.BatchGD()
      print "ttderiv: ", deriv
      #FIXME: send deriv to RNN interface


def main():
  batch_size = 30 #number of hidden representations fed into ddcrp
  alpha = 0.5 #concentration parameter
  #distance matrix for all hidden representations
  dim = 5 #dimenson of hid rep
  h_mat = np.random.rand(batch_size, dim)
  print "h_mat:\n", h_mat
  ddcrp = Ddcrp(h_mat, alpha)
  ddcrp.EM(15)
  #for i in range(50):
   # ddcrp.GippsSampling()

if __name__ == '__main__':
  main()




